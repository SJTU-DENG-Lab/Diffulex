import atexit
import os
import signal
import time

import torch.multiprocessing as mp

from dataclasses import fields
from time import perf_counter

from tqdm.auto import tqdm

from diffulex.config import Config
from diffulex.distributed.parallel_state import get_world_size
from diffulex.engine.model_runner import AutoModelRunner
from diffulex.engine.request import AutoReq
from diffulex.engine.scheduler import AutoScheduler, DataParallelScheduler, SchedulerBase
from diffulex.logger import get_logger
from diffulex.mixin.async_serving.engine import DiffulexAsyncEngineMixin
from diffulex.profiling import CudaStageTimer, TorchProfileSession, record_function
from diffulex.sampling_params import SamplingParams
from diffulex.utils.output import GenerationOutputs
from diffulex.utils.tokenizer import auto_tokenizer_from_pretrained

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
        self.profile_session = TorchProfileSession("engine", config=config.profiler_config)
        atexit.register(self.exit)
        self._install_signal_handlers()

        try:
            self.tokenizer = auto_tokenizer_from_pretrained(config.model, use_fast=True, trust_remote_code=True)
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
        if hasattr(self, "profile_session"):
            self.profile_session.stop()
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

    def start_profile(self, profile_prefix: str | None = None) -> None:
        self.profile_session.start(profile_prefix)
        if hasattr(self, "model_runner") and self.model_runner is not None:
            self.model_runner.call("start_profile", profile_prefix)

    def stop_profile(self) -> None:
        if hasattr(self, "model_runner") and self.model_runner is not None:
            self.model_runner.call("stop_profile")
        self.profile_session.stop()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            with record_function("diffulex.engine.tokenizer_encode"):
                prompt = self.tokenizer.encode(prompt)

        with record_function("diffulex.engine.add_request"):
            req = AutoReq.create(self.config, prompt, sampling_params)
        req.page_size = self.config.kv_cache_page_size
        with record_function("diffulex.engine.scheduler_add"):
            self.scheduler.add(req)
        return req.req_id

    def step(self):
        timer = CudaStageTimer.from_env("engine")
        try:
            with timer.stage("total"):
                with timer.stage("scheduler_schedule"):
                    with record_function("diffulex.engine.scheduler_schedule"):
                        reqs, is_prefill = self.scheduler.schedule()
                with timer.stage("prepare_reqs_for_execution"):
                    with record_function("diffulex.engine.prepare_reqs_for_execution"):
                        self._prepare_reqs_for_execution(reqs)
                try:
                    with timer.stage("model_runner_run"):
                        with record_function("diffulex.engine.model_runner_run"):
                            sample_output = self.model_runner.call("run", reqs)
                finally:
                    with timer.stage("clear_execution_prepared"):
                        with record_function("diffulex.engine.clear_execution_prepared"):
                            self._clear_execution_prepared(reqs)
                with timer.stage("scheduler_postprocess"):
                    with record_function("diffulex.engine.scheduler_postprocess"):
                        self.scheduler.postprocess(reqs, sample_output)
                finished_req_ids = [req.req_id for req in reqs if (req.is_completed or req.is_finished)]
                if finished_req_ids:
                    with timer.stage("evict_sampler_state"):
                        with record_function("diffulex.engine.evict_sampler_state"):
                            self.model_runner.call("evict_sampler_state", finished_req_ids)
                self.profile_session.step()
                return reqs, is_prefill
        finally:
            timer.flush()

    @staticmethod
    def _prepare_reqs_for_execution(reqs):
        for req in reqs:
            step_fn = getattr(req, "step", None)
            if callable(step_fn):
                step_fn()
            execution_is_prefill = bool(getattr(req, "is_prefilling", False))
            running_sequence = getattr(req, "running_sequence", None)
            setattr(req, "_last_execution_is_prefill", execution_is_prefill)
            setattr(
                req,
                "_last_execution_prefill_tokens",
                len(running_sequence or []) if execution_is_prefill else 0,
            )
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

    @staticmethod
    def _format_duration(seconds: float | None) -> str:
        if seconds is None or seconds < 0:
            return "?"
        seconds = int(seconds)
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours}h{minutes:02d}m{secs:02d}s"
        if minutes:
            return f"{minutes}m{secs:02d}s"
        return f"{secs}s"

    @staticmethod
    def _status_name(req) -> str:
        status = getattr(req, "status", None)
        return getattr(status, "name", str(status or "?"))

    def _log_generation_progress(
        self,
        *,
        outputs: GenerationOutputs,
        reqs,
        req_id_to_prompt_id: dict[int, int],
        completed_count: int,
        total_count: int,
        total_expected_tokens: int,
        start_wall_time: float,
        step: int,
        step_time: float,
    ) -> None:
        elapsed = perf_counter() - start_wall_time
        eta = None
        if total_count > 0 and completed_count >= total_count:
            eta = 0.0
        elif completed_count > 0:
            eta = elapsed / completed_count * max(total_count - completed_count, 0)
        elif total_expected_tokens > 0 and outputs.throughput > 0:
            remaining_tokens = max(total_expected_tokens - outputs.total_generated_tokens, 0)
            eta = remaining_tokens / outputs.throughput

        logger.info(
            "{generation %s/%s done} {step=%s step_time=%.3fs} "
            "{time elapsed=%s eta=%s} {tpf=%.2f tok/step} "
            "{avg e2e=%.2f tok/s decode=%.2f tok/s} "
            "{aggregate e2e=%.2f tok/s decode=%.2f tok/s}",
            completed_count,
            total_count,
            step,
            step_time,
            self._format_duration(elapsed),
            self._format_duration(eta),
            outputs.tpf,
            outputs.avg_e2e_tps,
            outputs.avg_decode_tps,
            outputs.e2e_throughput,
            outputs.decode_throughput,
        )

        active = []
        max_active_to_log = 8
        for req in reqs[:max_active_to_log]:
            prompt_idx = req_id_to_prompt_id.get(req.req_id, req.req_id)
            if prompt_idx >= len(outputs.trajectories):
                continue
            metrics = outputs.request_metrics(prompt_idx)
            active.append(
                "{req=%s prompt=%s status=%s new=%s tokens=%s tpf=%.2f "
                "e2e=%.2f tok/s decode=%.2f tok/s}"
                % (
                    req.req_id,
                    prompt_idx,
                    self._status_name(req),
                    getattr(req, "new_tokens", 0),
                    metrics["tokens"],
                    metrics["tpf"],
                    metrics["e2e_tps"],
                    metrics["decode_tps"],
                )
            )
        if active:
            omitted = len(reqs) - max_active_to_log
            suffix = f" | ... +{omitted} active" if omitted > 0 else ""
            logger.info("{active requests} %s%s", " | ".join(active), suffix)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = False,
        progress_log_interval: float = 5.0,
    ) -> list[str]:
        with record_function("diffulex.engine.generate"):
            pbar = None
            if use_tqdm:
                pbar = tqdm(
                    total=len(prompts),
                    desc="Diffulex Generating",
                    dynamic_ncols=True,
                    mininterval=1.0,
                )

            if not isinstance(sampling_params, list):
                sampling_params = [sampling_params] * len(prompts)
            total_expected_tokens = sum(max(getattr(sp, "max_tokens", 0) or 0, 0) for sp in sampling_params)

            req_id_to_prompt_id = {}
            for prompt_id, (prompt, sp) in tqdm(
                enumerate(zip(prompts, sampling_params)),
                total=len(prompts),
                desc="Adding Requests to Scheduler",
                dynamic_ncols=True,
                disable=not use_tqdm,
                leave=False,
                mininterval=1.0,
            ):
                req_id = self.add_request(prompt, sp)
                req_id_to_prompt_id[req_id] = prompt_id

            step = 0
            start_wall_time = None
            next_log_time = None
            next_progress_refresh = perf_counter()
            completed_req_ids = set()
            last_step_time = 0.0
            outputs = GenerationOutputs(len(prompts))
            while not self.is_finished():
                step += 1

                start = perf_counter()
                if start_wall_time is None:
                    start_wall_time = start
                    next_log_time = start
                reqs, is_prefill = self.step()
                step_time = perf_counter() - start
                last_step_time = step_time

                with record_function("diffulex.engine.record_outputs"):
                    outputs.record_step(reqs, step_time, req_id_to_prompt_id)

                if use_tqdm:
                    pbar.set_postfix(outputs.fast_postfix(), refresh=False)

                completed = 0
                for req in reqs:
                    if (req.is_completed or req.is_finished) and req.req_id not in completed_req_ids:
                        completed_req_ids.add(req.req_id)
                        completed += 1
                if completed and use_tqdm:
                    pbar.update(completed)
                if use_tqdm:
                    now = perf_counter()
                    if now >= next_progress_refresh:
                        pbar.refresh()
                        next_progress_refresh = now + 1.0
                elif progress_log_interval > 0:
                    now = perf_counter()
                    if now >= next_log_time:
                        self._log_generation_progress(
                            outputs=outputs,
                            reqs=reqs,
                            req_id_to_prompt_id=req_id_to_prompt_id,
                            completed_count=len(completed_req_ids),
                            total_count=len(prompts),
                            total_expected_tokens=total_expected_tokens,
                            start_wall_time=start_wall_time,
                            step=step,
                            step_time=step_time,
                        )
                        next_log_time = now + progress_log_interval

            if not use_tqdm and progress_log_interval > 0 and step > 0 and start_wall_time is not None:
                self._log_generation_progress(
                    outputs=outputs,
                    reqs=[],
                    req_id_to_prompt_id=req_id_to_prompt_id,
                    completed_count=len(completed_req_ids),
                    total_count=len(prompts),
                    total_expected_tokens=total_expected_tokens,
                    start_wall_time=start_wall_time,
                    step=step,
                    step_time=last_step_time,
                )

            if use_tqdm:
                pbar.close()

            outputs.log_summary()
            with record_function("diffulex.engine.convert_outputs_to_text"):
                outputs.convert_to_text(self.tokenizer)

            return outputs


__all__ = ["DiffulexEngine"]
