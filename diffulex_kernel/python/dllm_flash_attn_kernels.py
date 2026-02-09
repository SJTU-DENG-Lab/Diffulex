"""
Diffulex Flash-Attn kernel wrappers.

Goals:
- Decode path should NOT require TileLang at import time.
- Prefill behavior remains unchanged (TileLang for block attention / flash-attn varlen otherwise),
  but TileLang is imported lazily only when prefill is called.
"""

from __future__ import annotations

import os

import torch
from flash_attn import flash_attn_varlen_func

from diffulex.attention.metadata import AttnMetaDataBase
from diffulex_kernel.python.kv_cache_kernels import load_kvcache
from diffulex_kernel.python.paged_attn_decode_triton import paged_attn_decode_unified_triton


def dllm_flash_attn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    attn_metadata: AttnMetaDataBase,
) -> torch.Tensor:
    """
    Prefill attention wrapper.

    TileLang is imported lazily so decode-only usage does not depend on TileLang.
    """
    from diffulex_kernel.python.dllm_flash_attn_prefill_tilelang import (
        dllm_flash_attn_prefill_tilelang,
    )

    return dllm_flash_attn_prefill_tilelang(q, k, v, scale, attn_metadata)


def _decode_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    attn_metadata: AttnMetaDataBase,
) -> torch.Tensor:
    """
    Varlen decode path:
    - gather/dequant KV cache with Triton `load_kvcache`
    - run `flash_attn_varlen_func`
    """
    do_profile = os.getenv("DIFFULEX_PROFILE_KVCACHE", "0") == "1"
    if do_profile and q.is_cuda:
        e0, e1, e2 = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        e0.record()
        k_comb, v_comb = load_kvcache(k_cache, v_cache, attn_metadata, k, v)
        e1.record()
        out = flash_attn_varlen_func(
            q,
            k_comb,
            v_comb,
            attn_metadata.cu_seqlens_q,
            attn_metadata.cu_seqlens_k,
            attn_metadata.max_seqlen_q,
            attn_metadata.max_seqlen_k,
            softmax_scale=scale,
            block_table=None,
        )
        e2.record()
        e2.synchronize()
        print(
            f"[DIFFULEX_PROFILE_KVCACHE] decode(varlen) "
            f"load_kvcache={e0.elapsed_time(e1):.3f}ms flash_attn={e1.elapsed_time(e2):.3f}ms"
        )
        return out

    k_comb, v_comb = load_kvcache(k_cache, v_cache, attn_metadata, k, v)
    return flash_attn_varlen_func(
        q,
        k_comb,
        v_comb,
        attn_metadata.cu_seqlens_q,
        attn_metadata.cu_seqlens_k,
        attn_metadata.max_seqlen_q,
        attn_metadata.max_seqlen_k,
        softmax_scale=scale,
        block_table=None,
    )


def _decode_static_unified_triton_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    attn_metadata: AttnMetaDataBase,
) -> torch.Tensor:
    return paged_attn_decode_unified_triton(
        q,
        k,
        v,
        k_cache,
        v_cache,
        attn_metadata,
        softmax_scale=scale,
        fp8_cache=False,
    )


def _decode_static_unified_triton_fp8_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    attn_metadata: AttnMetaDataBase,
) -> torch.Tensor:
    if attn_metadata.k_scale is None or attn_metadata.v_scale is None:
        raise ValueError("FP8 KV decode requires k_scale and v_scale in metadata")

    # KV cache is stored as uint8 for FP8, but Triton expects float8 view dtype.
    from diffulex.utils.quantization.context import get_kv_cache_strategy

    strategy = get_kv_cache_strategy()
    if strategy is None or getattr(strategy, "kv_cache_format", "bf16") != "fp8":
        raise ValueError(f"Expected kv_cache_format='fp8', got strategy={type(strategy)}")

    k_cache_fp8 = strategy.view_kv_cache_for_kernels(k_cache)
    v_cache_fp8 = strategy.view_kv_cache_for_kernels(v_cache)

    return paged_attn_decode_unified_triton(
        q,
        k,
        v,
        k_cache_fp8,
        v_cache_fp8,
        attn_metadata,
        softmax_scale=scale,
        fp8_cache=True,
    )


def dllm_flash_attn_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    attn_metadata: AttnMetaDataBase,
) -> torch.Tensor:
    """
    Decode attention wrapper:
    - static: Triton paged-attention over (paged) KV cache + current-step KV
    - varlen: load_kvcache (Triton gather/dequant) + flash-attn varlen
    """
    from diffulex.utils.quantization.context import get_kv_cache_strategy

    kv_strategy = get_kv_cache_strategy()
    kv_fmt = getattr(kv_strategy, "kv_cache_format", "bf16") if kv_strategy is not None else "bf16"

    decode_mode = getattr(attn_metadata, "decode_mode", "varlen")
    if decode_mode == "static":
        # Only unified layout is supported in static paged-attention for now.
        if getattr(attn_metadata, "kv_cache_layout", "unified") != "unified":
            return _decode_varlen(q, k, v, k_cache, v_cache, scale, attn_metadata)

        if kv_fmt == "bf16":
            return _decode_static_unified_triton_bf16(q, k, v, k_cache, v_cache, scale, attn_metadata)
        if kv_fmt == "fp8":
            return _decode_static_unified_triton_fp8_cache(q, k, v, k_cache, v_cache, scale, attn_metadata)
        raise ValueError(f"Unsupported kv_cache_format={kv_fmt!r} for static decode")

    if decode_mode == "varlen":
        return _decode_varlen(q, k, v, k_cache, v_cache, scale, attn_metadata)

    raise ValueError(f"Unsupported decode mode: {decode_mode!r}")


__all__ = [
    "dllm_flash_attn_prefill",
    "dllm_flash_attn_decode",
]
