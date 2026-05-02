import torch

from dataclasses import dataclass

from diffulex.strategy_template.token_merge.attention.metadata import (
    TokenMergeAttnMetaDataTemplate,
)


@dataclass
class DMaxAttnMetaData(TokenMergeAttnMetaDataTemplate):
    def __post_init__(self):
        self.init_multi_block()
        self.reset_token_merge()


DMAX_ATTN_METADATA = DMaxAttnMetaData()


def fetch_dmax_attn_metadata() -> DMaxAttnMetaData:
    return DMAX_ATTN_METADATA


def set_dmax_attn_metadata(
    is_prefill: bool = False,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping: torch.Tensor | None = None,
    context_lens: torch.Tensor | None = None,
    page_tables: torch.Tensor | None = None,
    page_size: int = 32,
    block_size: int = 32,
    kv_cache_layout: str = "unified",
) -> None:
    global DMAX_ATTN_METADATA
    DMAX_ATTN_METADATA = DMaxAttnMetaData(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        page_tables=page_tables,
        page_size=page_size,
        block_size=block_size,
        kv_cache_layout=kv_cache_layout,
    )


def reset_dmax_attn_metadata() -> None:
    global DMAX_ATTN_METADATA
    DMAX_ATTN_METADATA = DMaxAttnMetaData()
