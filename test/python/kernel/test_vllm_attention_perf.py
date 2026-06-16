from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import pytest
import torch
import triton

from diffulex.strategy.multi_bd.attention.metadata import MultiBDAttnMetaData
from diffulex_kernel.python.chunked_prefill_triton import (
    _chunked_prefill_attn_unified_kernel,
    chunked_prefill_attn_unified,
)
from diffulex_kernel.python.kv_cache_kernels import store_kv_cache_unified


pytestmark = [
    pytest.mark.vllm_attention_perf,
    pytest.mark.skipif(
        os.getenv("DIFFULEX_RUN_VLLM_ATTN_PERF", "0") != "1",
        reason="set DIFFULEX_RUN_VLLM_ATTN_PERF=1 to run vLLM attention microbenchmarks",
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@dataclass(frozen=True)
class AttentionBenchCase:
    name: str
    q_lens: tuple[int, ...]
    ctx_lens: tuple[int, ...]
    q_heads: int
    kv_heads: int
    head_dim: int
    status: int
    vllm_causal: bool
    prefix_causal: bool
    is_prefix_full: bool = False
    prefix_lens: tuple[int, ...] | None = None
    compare_outputs: bool = True
    note: str = ""


@dataclass
class AttentionBenchData:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    diffulex_k_cache: torch.Tensor
    diffulex_v_cache: torch.Tensor
    vllm_k_cache: torch.Tensor
    vllm_v_cache: torch.Tensor
    page_tables: torch.Tensor
    slot_mapping: torch.Tensor
    metadata: MultiBDAttnMetaData
    seq_lens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    vllm_out: torch.Tensor
    softmax_segm_output: torch.Tensor
    softmax_segm_max: torch.Tensor
    softmax_segm_expsum: torch.Tensor


@dataclass(frozen=True)
class AttentionBenchResult:
    name: str
    shape: str
    diffulex_attn_ms: float
    diffulex_total_ms: float
    vllm_attn_ms: float
    vllm_total_ms: float
    max_abs_err: float
    compared: bool
    note: str


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def _env_dtype(name: str, default: torch.dtype) -> torch.dtype:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.lower()
    if normalized in ("bf16", "bfloat16"):
        return torch.bfloat16
    if normalized in ("fp16", "float16", "half"):
        return torch.float16
    if normalized in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype for {name}: {raw}")


def _bench_cuda(fn: Callable[[], object], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iters


def _make_data(
    case: AttentionBenchCase,
    *,
    page_size: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> AttentionBenchData:
    num_seqs = len(case.q_lens)
    assert len(case.ctx_lens) == num_seqs

    cu = [0]
    for q_len in case.q_lens:
        cu.append(cu[-1] + int(q_len))
    cu_seqlens_q = torch.tensor(cu, dtype=torch.int32, device=device)
    total_q = cu[-1]

    q = torch.randn(total_q, case.q_heads, case.head_dim, device=device, dtype=dtype)
    k = torch.randn(total_q, case.kv_heads, case.head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)

    seq_lens_host = [int(ctx) + int(q_len) for ctx, q_len in zip(case.ctx_lens, case.q_lens, strict=True)]
    pages_per_seq = [max(1, (seq_len + page_size - 1) // page_size) for seq_len in seq_lens_host]
    max_pages = max(pages_per_seq)
    total_pages = sum(pages_per_seq)

    base_k_cache = torch.randn(total_pages, page_size, case.kv_heads, case.head_dim, device=device, dtype=dtype)
    base_v_cache = torch.randn_like(base_k_cache)
    diffulex_k_cache = base_k_cache.clone()
    diffulex_v_cache = base_v_cache.clone()
    vllm_k_cache = base_k_cache.clone()
    vllm_v_cache = base_v_cache.clone()

    page_tables = torch.full((num_seqs, max_pages), -1, dtype=torch.int32, device=device)
    page_offset = 0
    for seq_id, num_pages in enumerate(pages_per_seq):
        for rel_page in range(num_pages):
            page_tables[seq_id, rel_page] = page_offset + rel_page
        page_offset += num_pages

    slots: list[int] = []
    for seq_id, (ctx_len, q_len) in enumerate(zip(case.ctx_lens, case.q_lens, strict=True)):
        for rel_token in range(q_len):
            pos = int(ctx_len) + rel_token
            rel_page = pos // page_size
            abs_page = int(page_tables[seq_id, rel_page].item())
            slots.append(abs_page * page_size + pos % page_size)
    slot_mapping = torch.tensor(slots, dtype=torch.long, device=device)

    if case.prefix_lens is None:
        prefix_lens_host = [0 if case.status else int(ctx) for ctx in case.ctx_lens]
    else:
        prefix_lens_host = [int(v) for v in case.prefix_lens]
        assert len(prefix_lens_host) == num_seqs
    padded_prefix_lens_host = [
        ((prefix_len + block_size - 1) // block_size) * block_size for prefix_len in prefix_lens_host
    ]

    metadata = MultiBDAttnMetaData(
        is_prefill=[case.status == 0] * num_seqs,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_q,
        max_seqlen_q=max(case.q_lens),
        max_seqlen_k=max(case.q_lens),
        slot_mapping=slot_mapping,
        context_lens=torch.tensor(case.ctx_lens, dtype=torch.int32, device=device),
        page_tables=page_tables,
        page_size=page_size,
        block_size=block_size,
        kv_cache_layout="unified",
    )
    metadata.init_multi_block(
        valid_slices=torch.tensor([cu[i] + case.q_lens[i] for i in range(num_seqs)], dtype=torch.int32, device=device),
        buffer_size=1,
        is_prefix_full=case.is_prefix_full,
        status_table=torch.full((num_seqs,), case.status, dtype=torch.int32, device=device),
        prefix_lens=torch.tensor(prefix_lens_host, dtype=torch.int32, device=device),
        padded_prefix_lens=torch.tensor(padded_prefix_lens_host, dtype=torch.int32, device=device),
        mask_prefix_hole=False,
        prefix_causal=case.prefix_causal,
    )
    num_par_softmax_segments = 16
    seq_threshold_3d = max(1, 128 // case.kv_heads)
    head_dim_padded = 1 << (case.head_dim - 1).bit_length()

    return AttentionBenchData(
        q=q,
        k=k,
        v=v,
        diffulex_k_cache=diffulex_k_cache,
        diffulex_v_cache=diffulex_v_cache,
        vllm_k_cache=vllm_k_cache,
        vllm_v_cache=vllm_v_cache,
        page_tables=page_tables,
        slot_mapping=slot_mapping,
        metadata=metadata,
        seq_lens=torch.tensor(seq_lens_host, dtype=torch.int32, device=device),
        cu_seqlens_q=cu_seqlens_q,
        vllm_out=torch.empty_like(q),
        softmax_segm_output=torch.empty(
            (seq_threshold_3d, case.q_heads, num_par_softmax_segments, head_dim_padded),
            dtype=torch.float32,
            device=device,
        ),
        softmax_segm_max=torch.empty(
            (seq_threshold_3d, case.q_heads, num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        ),
        softmax_segm_expsum=torch.empty(
            (seq_threshold_3d, case.q_heads, num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        ),
    )


def _vllm_store_kv(data: AttentionBenchData) -> None:
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash

    scale = torch.tensor(1.0, dtype=torch.float32, device=data.q.device)
    triton_reshape_and_cache_flash(
        data.k,
        data.v,
        data.vllm_k_cache,
        data.vllm_v_cache,
        data.slot_mapping,
        "auto",
        scale,
        scale,
    )


def _vllm_attention(data: AttentionBenchData, case: AttentionBenchCase) -> torch.Tensor:
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    num_par_softmax_segments = 16
    seq_threshold_3d = max(1, 128 // case.kv_heads)
    unified_attention(
        q=data.q,
        k=data.vllm_k_cache,
        v=data.vllm_v_cache,
        out=data.vllm_out,
        cu_seqlens_q=data.cu_seqlens_q,
        max_seqlen_q=max(case.q_lens),
        seqused_k=data.seq_lens,
        max_seqlen_k=int(data.seq_lens.max().item()),
        softmax_scale=1.0 / case.head_dim**0.5,
        causal=case.vllm_causal,
        window_size=(-1, -1),
        block_table=data.page_tables,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=seq_threshold_3d,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=data.softmax_segm_output,
        softmax_segm_max=data.softmax_segm_max,
        softmax_segm_expsum=data.softmax_segm_expsum,
    )
    return data.vllm_out


def _diffulex_store_kv(data: AttentionBenchData) -> None:
    store_kv_cache_unified(
        data.k,
        data.v,
        data.diffulex_k_cache,
        data.diffulex_v_cache,
        data.slot_mapping,
    )


def _diffulex_attention(data: AttentionBenchData, case: AttentionBenchCase) -> torch.Tensor:
    return chunked_prefill_attn_unified(
        data.q,
        data.k,
        data.v,
        data.diffulex_k_cache,
        data.diffulex_v_cache,
        data.metadata,
        softmax_scale=1.0 / case.head_dim**0.5,
    )


def _diffulex_attention_fixed(data: AttentionBenchData, case: AttentionBenchCase) -> torch.Tensor:
    out = torch.empty_like(data.q)
    metadata = data.metadata
    head_dim = data.q.shape[-1]
    head_dim_padded = 1 << (head_dim - 1).bit_length()
    grid = (
        len(case.q_lens),
        case.q_heads,
        triton.cdiv(int(metadata.max_seqlen_q), 8),
    )
    _chunked_prefill_attn_unified_kernel[grid](
        data.q,
        data.k,
        data.v,
        out,
        data.diffulex_k_cache,
        data.diffulex_v_cache,
        metadata.page_tables,
        metadata.status_table,
        metadata.context_lens,
        metadata.cu_seqlens_q,
        metadata.valid_slices,
        metadata.prefix_lens,
        metadata.padded_prefix_lens,
        1.0 / case.head_dim**0.5,
        *data.q.stride(),
        *data.k.stride(),
        *out.stride(),
        *data.diffulex_k_cache.stride(),
        *data.diffulex_v_cache.stride(),
        *metadata.page_tables.stride(),
        NUM_GROUPS=case.q_heads // case.kv_heads,
        HEAD_DIM=head_dim,
        HEAD_DIM_PADDED=head_dim_padded,
        PAGE_SIZE=data.diffulex_k_cache.shape[1],
        BLOCK_M=8,
        BLOCK_N=16,
        DLLM_BLOCK_SIZE=metadata.block_size,
        IS_BLOCK_CAUSAL=metadata.is_block_causal,
        IS_PREFIX_FULL=metadata.is_prefix_full,
        MASK_PREFIX_HOLE=bool(getattr(metadata, "mask_prefix_hole", False)),
        PREFIX_CAUSAL=bool(getattr(metadata, "prefix_causal", False)),
    )
    return out


def _bench_case(
    case: AttentionBenchCase,
    *,
    page_size: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    diffulex_autotune: bool,
) -> AttentionBenchResult:
    data = _make_data(case, page_size=page_size, block_size=block_size, device=device, dtype=dtype)
    diffulex_attn_fn = _diffulex_attention if diffulex_autotune else _diffulex_attention_fixed

    _vllm_store_kv(data)
    torch.cuda.synchronize()
    diffulex_out = diffulex_attn_fn(data, case)
    vllm_out = _vllm_attention(data, case)
    torch.cuda.synchronize()
    max_abs_err = float((diffulex_out - vllm_out).abs().max().item())
    if case.compare_outputs:
        torch.testing.assert_close(diffulex_out, vllm_out, rtol=6e-2, atol=6e-2)

    diffulex_attn_ms = _bench_cuda(lambda: diffulex_attn_fn(data, case), warmup=warmup, iters=iters)
    vllm_attn_ms = _bench_cuda(lambda: _vllm_attention(data, case), warmup=warmup, iters=iters)
    diffulex_total_ms = _bench_cuda(
        lambda: (_diffulex_store_kv(data), diffulex_attn_fn(data, case)),
        warmup=warmup,
        iters=iters,
    )
    vllm_total_ms = _bench_cuda(lambda: (_vllm_store_kv(data), _vllm_attention(data, case)), warmup=warmup, iters=iters)

    shape = (
        f"B={len(case.q_lens)},q={list(case.q_lens)},ctx={list(case.ctx_lens)},"
        f"QH={case.q_heads},KVH={case.kv_heads},D={case.head_dim},"
        f"status={case.status},vllm_causal={case.vllm_causal}"
    )
    return AttentionBenchResult(
        name=case.name,
        shape=shape,
        diffulex_attn_ms=diffulex_attn_ms,
        diffulex_total_ms=diffulex_total_ms,
        vllm_attn_ms=vllm_attn_ms,
        vllm_total_ms=vllm_total_ms,
        max_abs_err=max_abs_err,
        compared=case.compare_outputs,
        note=case.note,
    )


def _default_cases() -> list[AttentionBenchCase]:
    cases = [
        AttentionBenchCase(
            name="full causal prefill no-prefix",
            q_lens=(256,),
            ctx_lens=(0,),
            q_heads=16,
            kv_heads=2,
            head_dim=512,
            status=0,
            vllm_causal=True,
            prefix_causal=True,
        ),
        AttentionBenchCase(
            name="full causal prefill prefix-4k",
            q_lens=(256,),
            ctx_lens=(4096,),
            q_heads=16,
            kv_heads=2,
            head_dim=512,
            status=0,
            vllm_causal=True,
            prefix_causal=True,
        ),
        AttentionBenchCase(
            name="full block generation prefix-4k",
            q_lens=(256,),
            ctx_lens=(4096,),
            q_heads=16,
            kv_heads=2,
            head_dim=512,
            status=1,
            vllm_causal=True,
            prefix_causal=False,
            compare_outputs=False,
            note=(
                "installed vLLM unified_attention is causal-only; this is a paged token-causal timing proxy, "
                "not an equivalent block-visible mask"
            ),
        ),
        AttentionBenchCase(
            name="full causal prefill mixed-batch",
            q_lens=(256, 256, 256, 256),
            ctx_lens=(0, 1024, 4096, 8192),
            q_heads=16,
            kv_heads=2,
            head_dim=512,
            status=0,
            vllm_causal=True,
            prefix_causal=True,
        ),
        AttentionBenchCase(
            name="sliding-shape block generation prefix-1k",
            q_lens=(256,),
            ctx_lens=(1024,),
            q_heads=16,
            kv_heads=8,
            head_dim=512,
            status=1,
            vllm_causal=True,
            prefix_causal=False,
            compare_outputs=False,
            note=(
                "uses sliding layer head shape only; Diffulex runs block-visible attention, "
                "vLLM side is causal-only timing proxy, not SWA masking"
            ),
        ),
    ]
    if os.getenv("DIFFULEX_VLLM_ATTN_PERF_LONG", "0") == "1":
        cases.append(
            AttentionBenchCase(
                name="full block generation prefix-16k",
                q_lens=(256,),
                ctx_lens=(16384,),
                q_heads=16,
                kv_heads=2,
                head_dim=512,
                status=1,
                vllm_causal=True,
                prefix_causal=False,
                compare_outputs=False,
                note="long-prefix single block generation; vLLM side is causal-only timing proxy",
            )
        )
    return cases


def _print_results(
    results: list[AttentionBenchResult],
    *,
    device: torch.device,
    dtype: torch.dtype,
    iters: int,
) -> None:
    print(f"\nvLLM attention microbench: device={device}, dtype={dtype}, iters={iters}")
    print(
        f"{'case':<42} {'Diff attn':>10} {'Diff total':>11} {'vLLM attn':>10} "
        f"{'vLLM total':>11} {'attn ratio':>11} {'total ratio':>12} {'max err':>9} {'cmp':>4}"
    )
    for item in results:
        print(
            f"{item.name:<42} "
            f"{item.diffulex_attn_ms:>10.4f} "
            f"{item.diffulex_total_ms:>11.4f} "
            f"{item.vllm_attn_ms:>10.4f} "
            f"{item.vllm_total_ms:>11.4f} "
            f"{item.diffulex_attn_ms / item.vllm_attn_ms:>11.3f} "
            f"{item.diffulex_total_ms / item.vllm_total_ms:>12.3f} "
            f"{item.max_abs_err:>9.4f} "
            f"{'yes' if item.compared else 'no':>4}"
        )
        print(f"    {item.shape}")
        if item.note:
            print(f"    note: {item.note}")


def test_vllm_attention_microbenchmarks() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = _env_dtype("DIFFULEX_VLLM_ATTN_PERF_DTYPE", torch.bfloat16)
    warmup = _env_int("DIFFULEX_VLLM_ATTN_PERF_WARMUP", 5)
    iters = _env_int("DIFFULEX_VLLM_ATTN_PERF_ITERS", 30)
    page_size = _env_int("DIFFULEX_VLLM_ATTN_PERF_PAGE_SIZE", 256)
    block_size = _env_int("DIFFULEX_VLLM_ATTN_PERF_BLOCK_SIZE", 256)
    diffulex_autotune = os.getenv("DIFFULEX_VLLM_ATTN_PERF_DIFFULEX_AUTOTUNE", "0") == "1"
    case_filter = os.getenv("DIFFULEX_VLLM_ATTN_PERF_CASE_FILTER")
    cases = _default_cases()
    if case_filter:
        cases = [case for case in cases if case_filter.lower() in case.name.lower()]
        if not cases:
            raise ValueError(f"No attention perf cases matched DIFFULEX_VLLM_ATTN_PERF_CASE_FILTER={case_filter!r}")

    results = [
        _bench_case(
            case,
            page_size=page_size,
            block_size=block_size,
            device=device,
            dtype=dtype,
            warmup=warmup,
            iters=iters,
            diffulex_autotune=diffulex_autotune,
        )
        for case in cases
    ]
    _print_results(results, device=device, dtype=dtype, iters=iters)
