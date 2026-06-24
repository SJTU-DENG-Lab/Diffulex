from __future__ import annotations

import torch

from dataclasses import dataclass

from diffulex.attention.metadata import infer_prefill_flags
from diffulex.mixin.multi_block.attention_metadata import MultiBlockAttnMetaDataMixin


@dataclass
class FastDLLMV2AttnMetaData(MultiBlockAttnMetaDataMixin):
    fdv2_cache_only: bool = False
    fdv2_mode: int = 0


_FAST_DLLM_V2_ATTN_METADATA = FastDLLMV2AttnMetaData()


def set_fast_dllm_v2_attn_metadata(
    is_prefill: list[bool] | bool,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping: torch.Tensor | None = None,
    need_kv_cache_store: bool | None = None,
    context_lens: torch.Tensor | None = None,
    page_tables: torch.Tensor | None = None,
    page_size: int = 32,
    block_size: int = 32,
    kv_cache_layout: str = "unified",
    fdv2_cache_only: bool = False,
    fdv2_mode: int = 0,
) -> None:
    global _FAST_DLLM_V2_ATTN_METADATA
    has_prefill, all_prefill = infer_prefill_flags(is_prefill)
    _FAST_DLLM_V2_ATTN_METADATA = FastDLLMV2AttnMetaData(
        is_prefill=is_prefill if isinstance(is_prefill, list) else [bool(is_prefill)],
        enforce_eager=False,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        need_kv_cache_store_static=need_kv_cache_store,
        has_prefill_static=has_prefill,
        all_prefill_static=all_prefill,
        context_lens=context_lens,
        page_tables=page_tables,
        page_size=page_size,
        block_size=block_size,
        kv_cache_layout=kv_cache_layout,
        fdv2_cache_only=bool(fdv2_cache_only),
        fdv2_mode=int(fdv2_mode),
    )


def fetch_fast_dllm_v2_attn_metadata() -> FastDLLMV2AttnMetaData:
    return _FAST_DLLM_V2_ATTN_METADATA


def reset_fast_dllm_v2_attn_metadata() -> None:
    global _FAST_DLLM_V2_ATTN_METADATA
    _FAST_DLLM_V2_ATTN_METADATA = FastDLLMV2AttnMetaData()
