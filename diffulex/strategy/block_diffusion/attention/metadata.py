import torch

from typing import List
from dataclasses import dataclass

from diffulex.attention.metadata import AttnMetaDataBase
from diffulex.strategy.block_diffusion.engine.sequence import BDSequence


@dataclass
class BDAttnMetaData(AttnMetaDataBase):
    seqs: List[BDSequence] = None
    kv_cache_layout: str = "unified"
    need_kv_cache_store: bool = True
    
    def __post_init__(self):
        if self.context_lens is not None and sum(self.context_lens) > 0:
            self.total_lens = self.diffusion_block_size + self.context_lens


BD_ATTN_METADATA = BDAttnMetaData()

def fetch_bd_attn_metadata() -> BDAttnMetaData:
    return BD_ATTN_METADATA

def set_bd_attn_metadata(
    is_prefill: bool = False,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping: torch.Tensor | None = None,
    context_lens: torch.Tensor | None = None,
    block_tables: torch.Tensor | None = None,
    page_block_size: int = 32,
    diffusion_block_size: int = 32,
    decode_mode: str = "static",
    attn_type: str = "full_attention",
    kv_cache_layout: str = "unified",
    need_kv_cache_store: bool = True,
) -> None:
    global BD_ATTN_METADATA
    BD_ATTN_METADATA = BDAttnMetaData(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        page_block_size=page_block_size,
        diffusion_block_size=diffusion_block_size,
        kv_cache_layout=kv_cache_layout,
        need_kv_cache_store=need_kv_cache_store,
        decode_mode=decode_mode,
        attn_type=attn_type,
    )

def reset_bd_attn_metadata() -> None:
    global BD_ATTN_METADATA
    BD_ATTN_METADATA = BDAttnMetaData()