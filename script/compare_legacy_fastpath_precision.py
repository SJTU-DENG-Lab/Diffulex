#!/usr/bin/env python3
"""Compare chunked-prefill legacy page/block fastpath precision.

Runs the Triton kernel twice on identical inputs:
  1. current generic cache path
  2. legacy PAGE_SIZE == DLLM_BLOCK_SIZE cache fastpath

Both outputs are compared against the Python reference used by the kernel tests,
and against each other.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

os.environ.setdefault("DIFFULEX_DISABLE_CHUNKED_PREFILL_AUTOTUNE", "1")

import torch

from test.python.kernel.test_dllm_flash_attn_chunked_prefill_unified_kernel import (
    build_test_data,
    call_chunked_prefill_kernel,
    naive_chunked_prefill_ref,
)


@dataclass(frozen=True)
class Case:
    name: str
    q_lens: list[int]
    valid_q_lens: list[int]
    ctx_lens: list[int]
    page_size: int
    dllm_block_size: int
    is_block_causal: bool
    is_prefix_full: bool = False
    statuses: list[int] | None = None
    prefix_lens: list[int] | None = None
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128


CASES = [
    Case(
        name="prefix_cache_block_causal",
        q_lens=[128, 96, 64, 160],
        valid_q_lens=[128, 96, 64, 160],
        ctx_lens=[64, 128, 32, 96],
        page_size=32,
        dllm_block_size=32,
        is_block_causal=True,
    ),
    Case(
        name="mixed_all_have_cache",
        q_lens=[160, 128, 128, 128],
        valid_q_lens=[160, 128, 96, 32],
        ctx_lens=[32, 96, 256, 192],
        page_size=32,
        dllm_block_size=32,
        is_block_causal=True,
    ),
    Case(
        name="prefix_full_with_cache",
        q_lens=[128, 96, 64, 160],
        valid_q_lens=[128, 96, 64, 160],
        ctx_lens=[64, 128, 32, 96],
        page_size=32,
        dllm_block_size=32,
        is_block_causal=True,
        is_prefix_full=True,
        statuses=[0, 0, 0, 0],
        prefix_lens=[32, 20, 16, 48],
    ),
    Case(
        name="decode_cache_growth_shape",
        q_lens=[128],
        valid_q_lens=[96],
        ctx_lens=[256],
        page_size=32,
        dllm_block_size=32,
        is_block_causal=True,
        statuses=[1],
        prefix_lens=[0],
    ),
    Case(
        name="dbs64_prefix_full",
        q_lens=[128, 192],
        valid_q_lens=[128, 192],
        ctx_lens=[0, 64],
        page_size=64,
        dllm_block_size=64,
        is_block_causal=True,
        is_prefix_full=True,
        statuses=[0, 0],
        prefix_lens=[30, 80],
    ),
]


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    diff = (a.float() - b.float()).abs()
    denom = torch.maximum(a.float().abs(), b.float().abs()).clamp_min(1e-12)
    rel = diff / denom
    return float(diff.max().item()), float(diff.mean().item()), float(rel.max().item())


def _valid_rows(case: Case, cu: torch.Tensor, valid_slices: torch.Tensor) -> torch.Tensor:
    rows = []
    for i in range(len(case.q_lens)):
        start = int(cu[i].item())
        end = int(valid_slices[i].item())
        rows.extend(range(start, end))
    return torch.tensor(rows, dtype=torch.long, device=cu.device)


def run_case(case: Case, seed: int, dtype: torch.dtype) -> dict[str, float | str | int]:
    if case.page_size != case.dllm_block_size:
        raise ValueError(f"legacy fastpath only applies when page_size == block_size: {case}")

    torch.manual_seed(seed)
    statuses = case.statuses or [0] * len(case.q_lens)
    prefix_lens = case.prefix_lens or [0] * len(case.q_lens)
    padded_prefix_lens = [
        ((p + case.dllm_block_size - 1) // case.dllm_block_size) * case.dllm_block_size
        for p in prefix_lens
    ]
    data = build_test_data(
        q_lens=case.q_lens,
        valid_q_lens=case.valid_q_lens,
        ctx_lens=case.ctx_lens,
        num_heads=case.num_heads,
        num_kv_heads=case.num_kv_heads,
        head_dim=case.head_dim,
        page_size=case.page_size,
        statuses=statuses,
        prefix_lens=prefix_lens,
        dllm_block_size=case.dllm_block_size,
        dtype=dtype,
    )
    q, k, v, k_cache, v_cache, pt, st, cl, cu, vs, pl, ppl = data
    scale = 1.0 / case.head_dim**0.5

    new = call_chunked_prefill_kernel(
        q, k, v, k_cache, v_cache, pt, st, cl, cu, vs, pl, ppl, scale,
        case.dllm_block_size, case.is_block_causal, case.is_prefix_full,
        use_legacy_page_block_cache_fastpath=False,
    )
    legacy = call_chunked_prefill_kernel(
        q, k, v, k_cache, v_cache, pt, st, cl, cu, vs, pl, ppl, scale,
        case.dllm_block_size, case.is_block_causal, case.is_prefix_full,
        use_legacy_page_block_cache_fastpath=True,
    )
    ref = naive_chunked_prefill_ref(
        q, k, v, k_cache, v_cache, pt, statuses, cl, cu, vs,
        prefix_lens, padded_prefix_lens, scale, case.page_size,
        case.dllm_block_size, case.is_block_causal, case.is_prefix_full,
    )
    torch.cuda.synchronize()

    rows = _valid_rows(case, cu, vs)
    new_valid = new.index_select(0, rows)
    legacy_valid = legacy.index_select(0, rows)
    ref_valid = ref.index_select(0, rows)

    new_abs_max, new_abs_mean, new_rel_max = _diff_stats(new_valid, ref_valid)
    legacy_abs_max, legacy_abs_mean, legacy_rel_max = _diff_stats(legacy_valid, ref_valid)
    nl_abs_max, nl_abs_mean, nl_rel_max = _diff_stats(new_valid, legacy_valid)

    return {
        "case": case.name,
        "seed": seed,
        "new_ref_max": new_abs_max,
        "new_ref_mean": new_abs_mean,
        "new_ref_rel_max": new_rel_max,
        "legacy_ref_max": legacy_abs_max,
        "legacy_ref_mean": legacy_abs_mean,
        "legacy_ref_rel_max": legacy_rel_max,
        "new_legacy_max": nl_abs_max,
        "new_legacy_mean": nl_abs_mean,
        "new_legacy_rel_max": nl_rel_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Triton kernel comparison.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    print(f"device={torch.cuda.get_device_name(0)} dtype={dtype} seeds={args.seeds}")
    print(
        "case seed "
        "new_ref_max new_ref_mean new_ref_rel_max "
        "legacy_ref_max legacy_ref_mean legacy_ref_rel_max "
        "new_legacy_max new_legacy_mean new_legacy_rel_max"
    )

    totals = []
    for case in CASES:
        for seed in args.seeds:
            row = run_case(case, seed, dtype)
            totals.append(row)
            print(
                f"{row['case']} {row['seed']} "
                f"{row['new_ref_max']:.6g} {row['new_ref_mean']:.6g} {row['new_ref_rel_max']:.6g} "
                f"{row['legacy_ref_max']:.6g} {row['legacy_ref_mean']:.6g} {row['legacy_ref_rel_max']:.6g} "
                f"{row['new_legacy_max']:.6g} {row['new_legacy_mean']:.6g} {row['new_legacy_rel_max']:.6g}"
            )

    print("\nworst:")
    for key in ("new_ref_max", "legacy_ref_max", "new_legacy_max"):
        worst = max(totals, key=lambda row: float(row[key]))
        print(f"{key}={worst[key]:.6g} case={worst['case']} seed={worst['seed']}")


if __name__ == "__main__":
    main()
