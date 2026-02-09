import atexit
import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch.multiprocessing as mp

from tqdm.auto import tqdm
from time import perf_counter
from dataclasses import fields
from transformers import AutoTokenizer

from diffulex.config import Config
from diffulex.sampling_params import SamplingParams
from diffulex.engine.sequence import AutoSequence
from diffulex.engine.scheduler import AutoScheduler, SchedulerBase
from diffulex.engine.model_runner import AutoModelRunner
from diffulex.logger import get_logger

logger = get_logger(__name__)


class DiffulexTPWorker:
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=AutoModelRunner.from_config, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = AutoModelRunner.from_config(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, trust_remote_code=True)
        config.eos = self.tokenizer.eos_token_id
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
        seq = AutoSequence.create(self.config, prompt, sampling_params)
        seq.block_size = self.config.kvcache_block_size
        self.scheduler.add(seq)
        # Return seq_id so caller can build a stable mapping
        return seq.seq_id

    async def add_request_async(self, prompt: str | list[int], sampling_params: SamplingParams):
        """Async version of add_request (currently synchronous but provided for API consistency)."""
        # Tokenization and sequence creation are fast, but we make it async for consistency
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.add_request, prompt, sampling_params)

    def step(self):
        # Clear step-local activation quant cache (W8A8/W4A8, etc.) so we only reuse within a single step.
        try:
            from diffulex.utils.quantization.context import clear_act_quant_cache
            clear_act_quant_cache()
        except Exception:
            # Quantization context may not be initialized in some paths; ignore.
            pass
        seqs, is_prefill = self.scheduler.schedule()
        sample_output = self.model_runner.call("run", seqs, is_prefill)
        n_diff_steps = self.scheduler.postprocess(seqs, sample_output)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(seq.num_tokens for seq in seqs) if is_prefill else sum(seq.new_tokens for seq in seqs)
        # Diffusion decoding modifies tokens in-place; we currently don't stream intermediate edits
        deltas = []
        return outputs, num_tokens, is_prefill, n_diff_steps, deltas

    async def step_async(self):
        """Async version of step that runs model inference in a thread pool."""
        loop = asyncio.get_event_loop()
        executor = getattr(self, '_step_executor', None)
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=1)
            self._step_executor = executor
        
        def _step():
            seqs, is_prefill = self.scheduler.schedule()
            sample_output = self.model_runner.call("run", seqs, is_prefill)
            n_diff_steps = self.scheduler.postprocess(seqs, sample_output)
            outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
            num_tokens = sum(seq.num_tokens for seq in seqs) if is_prefill else sum(seq.new_tokens for seq in seqs)
            deltas = []
            return outputs, num_tokens, is_prefill, n_diff_steps, deltas
        
        return await loop.run_in_executor(executor, _step)

    def is_finished(self):
        return self.scheduler.is_finished()

    async def is_finished_async(self):
        """Async version of is_finished (currently synchronous but provided for API consistency)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.is_finished)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # Map internal seq_id -> input index to keep output order stable
        seqid_to_idx = {}
        for idx, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
            sid = self.add_request(prompt, sp)
            seqid_to_idx[sid] = idx
        outputs = [None] * len(prompts)
        # Track token/time totals for correct average throughput reporting.
        prefill_total_tokens = 0
        decode_total_tokens = 0
        prefill_total_time = 0.0
        decode_total_time = 0.0
        prefill_steps = 0
        decode_steps = 0
        n_steps = 0
        n_diff_steps = [-1] * len(prompts)
        while not self.is_finished():
            n_steps += 1
            t = perf_counter()
            output, num_tokens, is_prefill, cur_n_diff_steps, _ = self.step()
            dt = perf_counter() - t

            # Accumulate totals to compute average throughput correctly.
            if is_prefill:
                prefill_steps += 1
                prefill_total_tokens += int(num_tokens)
                prefill_total_time += float(dt)
            else:
                decode_steps += 1
                decode_total_tokens += int(num_tokens)
                decode_total_time += float(dt)

            if use_tqdm:
                avg_prefill_throughput = (
                    prefill_total_tokens / prefill_total_time if prefill_total_time > 0 else 0.0
                )
                avg_decode_throughput = (
                    decode_total_tokens / decode_total_time if decode_total_time > 0 else 0.0
                )
                pbar.set_postfix({
                    "Prefill(avg)": f"{int(avg_prefill_throughput)}tok/s",
                    "Decode(avg)": f"{int(avg_decode_throughput)}tok/s",
                })
            if cur_n_diff_steps:
                for seq_id, n_step in cur_n_diff_steps.items():
                    if seq_id in seqid_to_idx and n_step >= 0:
                        n_diff_steps[seqid_to_idx[seq_id]] = n_step
            for seq_id, token_ids in output:
                if seq_id in seqid_to_idx:
                    outputs[seqid_to_idx[seq_id]] = token_ids
                if use_tqdm:
                    pbar.update(1)
                    
        avg_prefill_throughput = (
            prefill_total_tokens / prefill_total_time if prefill_total_time > 0 else 0.0
        )
        avg_decode_throughput = (
            decode_total_tokens / decode_total_time if decode_total_time > 0 else 0.0
        )
        avg_prefill_step_ms = (
            (prefill_total_time / prefill_steps) * 1000.0 if prefill_steps > 0 else 0.0
        )
        avg_decode_step_ms = (
            (decode_total_time / decode_steps) * 1000.0 if decode_steps > 0 else 0.0
        )
        logger.info(
            "Finished in %d steps (prefill=%d, decode=%d). "
            "Prefill: %d tok in %.2fs (avg %.2f tok/s, %.2f ms/step). "
            "Decode: %d tok in %.2fs (avg %.2f tok/s, %.2f ms/step).",
            n_steps,
            prefill_steps,
            decode_steps,
            prefill_total_tokens,
            prefill_total_time,
            avg_prefill_throughput,
            avg_prefill_step_ms,
            decode_total_tokens,
            decode_total_time,
            avg_decode_throughput,
            avg_decode_step_ms,
        )
        # Ensure all outputs are present
        assert all(toks is not None for toks in outputs), "Some sequences did not produce outputs"
        outputs = [{
            "text": self.tokenizer.decode(token_ids).split(self.tokenizer.eos_token)[0],
            "token_ids": token_ids[:token_ids.index(self.config.eos)] if self.config.eos in token_ids else token_ids,
            "n_diff_steps": n_diff_step,
        } for token_ids, n_diff_step in zip(outputs, n_diff_steps)]
        if use_tqdm:
            pbar.close()
        return outputs

    async def generate_async(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """Async version of generate that allows concurrent request handling."""
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # Map internal seq_id -> input index to keep output order stable
        seqid_to_idx = {}
        # Add requests synchronously to avoid race conditions with scheduler
        # The actual async benefit comes from the inference steps, not request addition
        for idx, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
            sid = self.add_request(prompt, sp)
            seqid_to_idx[sid] = idx
        outputs = [None] * len(prompts)
        prefill_throughput = decode_throughput = 0.
        n_steps = 0
        n_diff_steps = [-1] * len(prompts)
        while not await self.is_finished_async():
            t = perf_counter()
            n_steps += 1
            output, num_tokens, is_prefill, cur_n_diff_steps, _ = await self.step_async()
            if use_tqdm:
                if is_prefill:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            if cur_n_diff_steps:
                for seq_id, n_step in cur_n_diff_steps.items():
                    if seq_id in seqid_to_idx and n_step >= 0:
                        n_diff_steps[seqid_to_idx[seq_id]] = n_step
            for seq_id, token_ids in output:
                if seq_id in seqid_to_idx:
                    outputs[seqid_to_idx[seq_id]] = token_ids
                if use_tqdm:
                    pbar.update(1)
            await asyncio.sleep(0)
                    
        print(f"Finished in {n_steps} steps, prefill throughput: {prefill_throughput:.2f} tok/s, decode throughput: {decode_throughput:.2f} tok/s")
        assert all(toks is not None for toks in outputs), "Some sequences did not produce outputs"
        outputs = [{
            "text": self.tokenizer.decode(token_ids).split(self.tokenizer.eos_token)[0],
            "token_ids": token_ids[:token_ids.index(self.config.eos)] if self.config.eos in token_ids else token_ids,
            "n_diff_steps": n_diff_step,
        } for token_ids, n_diff_step in zip(outputs, n_diff_steps)]
        if use_tqdm:
            pbar.close()
        return outputs
