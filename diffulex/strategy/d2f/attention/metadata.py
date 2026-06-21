import torch

from dataclasses import dataclass

from diffulex.attention.metadata import infer_prefill_flags
from diffulex.mixin.multi_block.attention_metadata import MultiBlockAttnMetaDataMixin


@dataclass
class D2fAttnMetaData(MultiBlockAttnMetaDataMixin):
    def __post_init__(self):
        self.init_multi_block()


D2F_ATTN_METADATA = D2fAttnMetaData()


def fetch_d2f_attn_metadata() -> D2fAttnMetaData:
    return D2F_ATTN_METADATA


def set_d2f_attn_metadata(
    is_prefill: bool = False,
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
) -> None:
    global D2F_ATTN_METADATA
    has_prefill, all_prefill = infer_prefill_flags(is_prefill)
    D2F_ATTN_METADATA = D2fAttnMetaData(
        is_prefill=is_prefill,
        has_prefill_static=has_prefill,
        all_prefill_static=all_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        need_kv_cache_store_static=need_kv_cache_store,
        context_lens=context_lens,
        page_tables=page_tables,
        page_size=page_size,
        block_size=block_size,
        kv_cache_layout=kv_cache_layout,
    )


def reset_d2f_attn_metadata() -> None:
    global D2F_ATTN_METADATA
    D2F_ATTN_METADATA = D2fAttnMetaData()
