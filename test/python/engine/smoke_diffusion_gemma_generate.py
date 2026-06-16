from __future__ import annotations

import time

from diffulex import Diffulex
from diffulex.sampling_params import SamplingParams


def main() -> None:
    model = "/data1/ckpts/google/diffusiongemma-26B-A4B-it"
    print("constructing engine...", flush=True)
    t0 = time.time()
    llm = Diffulex(
        model,
        model_name="diffusion_gemma",
        tensor_parallel_size=2,
        data_parallel_size=1,
        expert_parallel_size=1,
        master_port=29575,
        distributed_backend="gloo",
        device_ids=[0, 1],
        gpu_memory_utilization=0.70,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_reqs=1,
        num_pages=2,
        enforce_eager=True,
        enable_prefill_cudagraph=False,
        enable_full_static_runner=False,
        enable_torch_compile=False,
        enable_cudagraph_torch_compile=False,
        enable_prefix_caching=False,
        kv_cache_layout="unified",
        moe_dispatcher_backend="standard",
        moe_gemm_impl="triton",
    )
    print(f"engine constructed in {time.time() - t0:.2f}s", flush=True)
    try:
        prompt = "What is 2+2?"
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=16,
            max_nfe=2,
            max_repetition_run=32,
            ignore_eos=False,
        )
        print("generating...", flush=True)
        t1 = time.time()
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        print(f"generated in {time.time() - t1:.2f}s", flush=True)
        print(type(outputs), flush=True)
        print(outputs, flush=True)
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
