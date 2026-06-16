from __future__ import annotations

import argparse
import time

import torch

from test.python.kernel.test_vllm_attention_perf import (
    _bench_cuda,
    _default_cases,
    _diffulex_attention,
    _diffulex_attention_fixed,
    _diffulex_store_kv,
    _make_data,
    _vllm_attention,
    _vllm_store_kv,
)
from diffulex_kernel.python.chunked_prefill_grouped_triton import chunked_prefill_attn_grouped_unified


def _select_case(case_filter: str):
    matches = [case for case in _default_cases() if case_filter.lower() in case.name.lower()]
    if not matches:
        raise ValueError(f"No case matched {case_filter!r}")
    if len(matches) > 1:
        names = ", ".join(case.name for case in matches)
        raise ValueError(f"Case filter {case_filter!r} matched multiple cases: {names}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile one Diffulex/vLLM paged attention kernel shape.")
    parser.add_argument("--backend", choices=("diffulex", "diffulex-autotune", "grouped", "vllm"), required=True)
    parser.add_argument("--case-filter", default="prefix-4k")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    case = _select_case(args.case_filter)
    data = _make_data(
        case,
        page_size=args.page_size,
        block_size=args.block_size,
        device=device,
        dtype=torch.bfloat16,
    )

    if args.backend == "vllm":
        _vllm_store_kv(data)
        fn = lambda: _vllm_attention(data, case)
    elif args.backend == "grouped":
        _diffulex_store_kv(data)
        scale = 1.0 / case.head_dim**0.5
        fn = lambda: chunked_prefill_attn_grouped_unified(
            data.q,
            data.k,
            data.v,
            data.diffulex_k_cache,
            data.diffulex_v_cache,
            data.metadata,
            softmax_scale=scale,
        )
    elif args.backend == "diffulex-autotune":
        _diffulex_store_kv(data)
        fn = lambda: _diffulex_attention(data, case)
    else:
        _diffulex_store_kv(data)
        fn = lambda: _diffulex_attention_fixed(data, case)

    # Compile and warm up before the profiled loop. Nsight Compute still sees
    # only launches inside the process, but launch-skip can ignore this prefix.
    ms = _bench_cuda(fn, warmup=args.warmup, iters=max(1, min(args.iters, 5)))
    print(f"ready backend={args.backend} case={case.name!r} warm_ms={ms:.4f}", flush=True)
    time.sleep(0.25)

    torch.cuda.nvtx.range_push("attention_profile")
    try:
        for _ in range(args.iters):
            fn()
        torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()


if __name__ == "__main__":
    main()
