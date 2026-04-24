import atexit
import os
import signal
import time

import torch.multiprocessing as mp

from dataclasses import fields
from time import perf_counter

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from diffulex.config import Config
from diffulex.distributed.parallel_state import get_world_size
from diffulex.engine.model_runner import AutoModelRunner
from diffulex.engine.request import AutoReq
from diffulex.engine.scheduler import AutoScheduler, DataParallelScheduler, SchedulerBase
from diffulex.logger import get_logger
from diffulex.mixin.async_serving.engine import DiffulexAsyncEngineMixin
from diffulex.sampling_params import SamplingParams
from diffulex.utils.output import GenerationOutputs

logger = get_logger(__name__)


def _set_parent_death_signal(sig: int = signal.SIGTERM) -> None:
    if os.name != "posix":
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        pr_set_pdeathsig = 1
        libc.prctl(pr_set_pdeathsig, sig)
    except Exception:
        logger.debug("Failed to set parent-death signal for worker process.", exc_info=True)


def _run_model_runner_worker(config: Config, rank: int, event) -> None:
    _set_parent_death_signal()
    if os.getppid() == 1:
        raise SystemExit("Diffulex worker parent exited before worker initialization.")
    AutoModelRunner.from_config(config, rank, event)


class DiffulexEngine(DiffulexAsyncEngineMixin):
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = config = Config(model, **config_kwargs)
        self.model_parallel_world_size = get_world_size(
            config.tensor_parallel_size,
            config.expert_parallel_size,
            dp_size=config.data_parallel_size,
        )
        if len(config.device_ids) < self.model_parallel_world_size:
            raise ValueError(
                "Not enough CUDA devices for the requested topology, "
                f"need {self.model_parallel_world_size}, got device_ids={config.device_ids}."
            )
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, self.model_parallel_world_size):
            event = ctx.Event()
            process = ctx.Process(target=_run_model_runner_worker, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self._exited = False
        atexit.register(self.exit)
        self._install_signal_handlers()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, trust_remote_code=True)
            config.tokenizer_vocab_size = len(self.tokenizer)
            config.eos = self.tokenizer.eos_token_id

            if (
                getattr(self.tokenizer, "mask_token_id", None) is not None
                and config.mask_token_id != self.tokenizer.mask_token_id
            ):
                logger.warning(
                    "Overriding mask_token_id from %s to tokenizer mask_token_id %s.",
                    config.mask_token_id,
                    self.tokenizer.mask_token_id,
                )
                config.mask_token_id = self.tokenizer.mask_token_id

            self.model_runner = AutoModelRunner.from_config(config, 0, self.events)
            self.scheduler: SchedulerBase | DataParallelScheduler = AutoScheduler.from_config(config)
        except BaseException:
            self.exit()
            raise

    def _install_signal_handlers(self) -> None:
        if getattr(self, "_signal_handlers_installed", False):
            return
        self._signal_handlers_installed = True
        self._previous_signal_handlers = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous = signal.getsignal(sig)
                self._previous_signal_handlers[sig] = previous

                def handler(signum, frame, *, _previous=previous):
                    self.exit()
                    if callable(_previous):
                        _previous(signum, frame)
                    else:
                        raise SystemExit(128 + signum)

                signal.signal(sig, handler)
            except Exception:
                logger.debug("Failed to install signal handler for %s.", sig, exc_info=True)

    @staticmethod
    def _join_or_stop_process(process, *, timeout: float = 5.0) -> None:
        try:
            process.join(timeout=timeout)
        except Exception:
            logger.debug("Failed to join worker process %s.", getattr(process, "pid", None), exc_info=True)
        if not process.is_alive():
            return

        logger.warning("Terminating stale worker process pid=%s.", process.pid)
        try:
            process.terminate()
        except Exception:
            logger.debug("Failed to terminate worker process %s.", process.pid, exc_info=True)
        try:
            process.join(timeout=timeout)
        except Exception:
            logger.debug("Failed to join terminated worker process %s.", process.pid, exc_info=True)
        if not process.is_alive():
            return

        logger.warning("Killing stale worker process pid=%s.", process.pid)
        try:
            process.kill()
        except Exception:
            logger.debug("Failed to kill worker process %s.", process.pid, exc_info=True)
        try:
            process.join(timeout=timeout)
        except Exception:
            logger.debug("Failed to join killed worker process %s.", process.pid, exc_info=True)

    def exit(self):
        if getattr(self, "_exited", False):
            return
        self._exited = True
        if hasattr(self, "model_runner") and self.model_runner is not None:
            try:
                self.model_runner.call("exit")
            except Exception:
                pass
            try:
                del self.model_runner
            except Exception:
                pass
        for p in getattr(self, "ps", []):
            self._join_or_stop_process(p)
        time.sleep(0)

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)

        req = AutoReq.create(self.config, prompt, sampling_params)
        req.page_size = self.config.kv_cache_page_size
        self.scheduler.add(req)
        return req.req_id

    def step(self):
        reqs, is_prefill = self.scheduler.schedule()
        self._prepare_reqs_for_execution(reqs)
        try:
            sample_output = self.model_runner.call("run", reqs)
        finally:
            self._clear_execution_prepared(reqs)
        self.scheduler.postprocess(reqs, sample_output)
        finished_req_ids = [req.req_id for req in reqs if (req.is_completed or req.is_finished)]
        if finished_req_ids:
            self.model_runner.call("evict_sampler_state", finished_req_ids)
        return reqs, is_prefill

    @staticmethod
    def _prepare_reqs_for_execution(reqs):
        for req in reqs:
            step_fn = getattr(req, "step", None)
            if callable(step_fn):
                step_fn()
            mark_fn = getattr(req, "mark_execution_prepared", None)
            if callable(mark_fn):
                mark_fn()

    @staticmethod
    def _clear_execution_prepared(reqs):
        for req in reqs:
            clear_fn = getattr(req, "clear_execution_prepared", None)
            if callable(clear_fn):
                clear_fn()

    def is_finished(self):
        return self.scheduler.is_finished()

    def abort_request(self, req_id: int) -> bool:
        return self.scheduler.abort_request(req_id)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Diffulex Generating", dynamic_ncols=True)

        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        req_id_to_prompt_id = {}
        for prompt_id, (prompt, sp) in tqdm(
            enumerate(zip(prompts, sampling_params)),
            total=len(prompts),
            desc="Adding Requests to Scheduler",
            dynamic_ncols=True,
        ):
            req_id = self.add_request(prompt, sp)
            req_id_to_prompt_id[req_id] = prompt_id

        step = 0
        outputs = GenerationOutputs(len(prompts))
        while not self.is_finished():
            step += 1

            start = perf_counter()
            reqs, is_prefill = self.step()
            step_time = perf_counter() - start

            outputs.record_step(reqs, step_time, req_id_to_prompt_id)

            if use_tqdm:
                pbar.set_postfix(outputs.postfix())

            for req in reqs:
                if (req.is_completed or req.is_finished) and use_tqdm:
                    pbar.update(1)

        if use_tqdm:
            pbar.close()

        outputs.log_summary()
        outputs.convert_to_text(self.tokenizer)

        return outputs


__all__ = ["DiffulexEngine"]
