import atexit

import torch.multiprocessing as mp

from tqdm.auto import tqdm
from time import perf_counter
from dataclasses import fields
from transformers import AutoTokenizer

from diffulex.config import Config
from diffulex.sampling_params import SamplingParams
from diffulex.engine.request import AutoReq
from diffulex.engine.scheduler import AutoScheduler, SchedulerBase
from diffulex.engine.model_runner import AutoModelRunner
from diffulex.mixin.async_engine.engine.serving_worker import DiffulexServingWorkerMixin
from diffulex.utils.output import GenerationOutputs
from diffulex.utils.parallelism import get_world_size
from diffulex.logger import get_logger

logger = get_logger(__name__)


class DiffulexTPWorker(DiffulexServingWorkerMixin):
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = config = Config(model, **config_kwargs)
        self.model_parallel_world_size = get_world_size(
            config.tensor_parallel_size,
            config.expert_parallel_size,
        )
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, self.model_parallel_world_size):
            event = ctx.Event()
            process = ctx.Process(target=AutoModelRunner.from_config, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, trust_remote_code=True)
        config.eos = self.tokenizer.eos_token_id
        
        if getattr(self.tokenizer, "mask_token_id", None) is not None and config.mask_token_id != self.tokenizer.mask_token_id:
            logger.warning(
                "Overriding mask_token_id from %s to tokenizer mask_token_id %s.",
                config.mask_token_id,
                self.tokenizer.mask_token_id,
            )
            config.mask_token_id = self.tokenizer.mask_token_id
            
        self.model_runner = AutoModelRunner.from_config(config, 0, self.events)
        self.scheduler: SchedulerBase = AutoScheduler.from_config(config)
        self._exited = False
        atexit.register(self.exit)

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
            try:
                p.join()
            except Exception:
                pass

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)

        req = AutoReq.create(self.config, prompt, sampling_params)
        req.page_size = self.config.kv_cache_page_size
        self.scheduler.add(req)
        return req.req_id

    def step(self):
        reqs, is_prefill = self.scheduler.schedule()
        sample_output = self.model_runner.call("run", reqs)
        self.scheduler.postprocess(reqs, sample_output)
        return reqs, is_prefill

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
