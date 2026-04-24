"""Diffulex CUDA kernel package.

Keep this module lightweight: importing `diffulex_kernel` should not eagerly
import optional heavy deps unless the corresponding kernels are actually used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffulex_kernel.python.chunked_prefill_triton import (  # noqa: F401
        chunked_prefill_attn_unified as dllm_chunked_prefill,
    )
    from diffulex_kernel.python.kv_cache_kernels import (  # noqa: F401
        load_kv_cache as load_kv_cache,
        store_kv_cache_distinct_layout as store_kv_cache_distinct_layout,
        store_kv_cache_unified_layout as store_kv_cache_unified_layout,
    )


def __getattr__(name: str):
    if name in ("dllm_chunked_prefill", "chunked_prefill_attn_unified"):
        from diffulex_kernel.python.chunked_prefill_triton import (
            chunked_prefill_attn_unified,
        )
        return chunked_prefill_attn_unified

    if name == "store_kv_cache_unified_layout":
        from diffulex_kernel.python.kv_cache_kernels import store_kv_cache_unified_layout
        return store_kv_cache_unified_layout

    if name == "store_kv_cache_distinct_layout":
        from diffulex_kernel.python.kv_cache_kernels import store_kv_cache_distinct_layout
        return store_kv_cache_distinct_layout

    if name == "load_kv_cache":
        from diffulex_kernel.python.kv_cache_kernels import load_kv_cache
        return load_kv_cache

    if name == "fused_moe":
        from diffulex_kernel.python.fused_moe_triton import fused_moe
        return fused_moe

    if name == "vllm_fused_moe":
        from diffulex_kernel.python.vllm_fuse_moe import fused_moe
        return fused_moe

    if name == "fused_expert_packed":
        from diffulex_kernel.python.fused_moe_triton import fused_expert_packed
        return fused_expert_packed
    
    if name == "fused_topk":
        from diffulex_kernel.python.fused_topk_triton import fused_topk
        return fused_topk

    if name in ("fused_group_limited_topk", "fused_grouped_topk"):
        from diffulex_kernel.python.fused_topk_triton import fused_group_limited_topk
        return fused_group_limited_topk

    raise AttributeError(name)


__all__ = [
    "dllm_chunked_prefill",
    "chunked_prefill_attn_unified",
    "store_kv_cache_unified_layout",
    "store_kv_cache_distinct_layout",
    "load_kv_cache",
    "fused_moe",
    "vllm_fused_moe",
    "fused_expert_packed",
    "fused_topk",
    "fused_group_limited_topk",
    "fused_grouped_topk",
]
