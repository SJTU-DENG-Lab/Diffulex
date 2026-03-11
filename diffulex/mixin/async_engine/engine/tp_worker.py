import asyncio

from tqdm import tqdm
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor

from diffulex.sampling_params import SamplingParams


class DiffulexTPWorkerAsyncMixin:
    async def add_request_async(self, prompt: str | list[int], sampling_params: SamplingParams):
        """Async version of add_request (currently synchronous but provided for API consistency)."""
        # Tokenization and req creation are fast, but we make it async for consistency
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.add_request, prompt, sampling_params)

    async def step_async(self):
        """Async version of step that runs model inference in a thread pool."""
        loop = asyncio.get_running_loop()
        executor = getattr(self, "_step_executor", None)
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=1)
            self._step_executor = executor

        def _step():
            reqs, is_prefill = self.scheduler.schedule()
            sample_output = self.model_runner.call("run", reqs)
            n_diff_steps = self.scheduler.postprocess(reqs, sample_output)
            outputs = [(req.req_id, req.completion_token_ids) for req in reqs if req.is_finished]
            num_tokens = sum(req.num_tokens for req in reqs) if is_prefill else sum(req.new_tokens for req in reqs)
            deltas = []
            return outputs, num_tokens, is_prefill, n_diff_steps, deltas

        return await loop.run_in_executor(executor, _step)

    async def is_finished_async(self):
        """Async version of is_finished (currently synchronous but provided for API consistency)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.is_finished)

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
        # Map internal req_id -> input index to keep output order stable
        reqid_to_idx = {}
        # Add requests synchronously to avoid race conditions with scheduler
        # The actual async benefit comes from the inference steps, not request addition
        for idx, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
            rid = self.add_request(prompt, sp)
            reqid_to_idx[rid] = idx
        outputs = [None] * len(prompts)
        prefill_throughput = decode_throughput = 0.0
        n_steps = 0
        n_diff_steps = [-1] * len(prompts)
        while not await self.is_finished_async():
            t = perf_counter()
            n_steps += 1
            (
                output,
                num_tokens,
                is_prefill,
                cur_n_diff_steps,
                _,
            ) = await self.step_async()
            if use_tqdm:
                if is_prefill:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = num_tokens / (perf_counter() - t)
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
            if cur_n_diff_steps:
                for req_id, n_step in cur_n_diff_steps.items():
                    if req_id in reqid_to_idx and n_step >= 0:
                        n_diff_steps[reqid_to_idx[req_id]] = n_step
            for req_id, token_ids in output:
                if req_id in reqid_to_idx:
                    outputs[reqid_to_idx[req_id]] = token_ids
                if use_tqdm:
                    pbar.update(1)
            await asyncio.sleep(0)

        print(
            f"Finished in {n_steps} steps, prefill throughput: {prefill_throughput:.2f} tok/s, decode throughput: {decode_throughput:.2f} tok/s"
        )
        assert all(toks is not None for toks in outputs), "Some reqs did not produce outputs"
        outputs = [
            {
                "text": self.tokenizer.decode(token_ids).split(self.tokenizer.eos_token)[0],
                "token_ids": token_ids[: token_ids.index(self.config.eos)]
                if self.config.eos in token_ids
                else token_ids,
                "n_diff_steps": n_diff_step,
            }
            for token_ids, n_diff_step in zip(outputs, n_diff_steps)
        ]
        if use_tqdm:
            pbar.close()
        return outputs
