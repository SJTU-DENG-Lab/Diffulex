import torch

from typing import List
from dataclasses import dataclass

from diffulex.attention.metadata import AttnMetaDataBase
from diffulex.strategy.block_diffusion.engine.sequence import BDSequence


@dataclass
class BDAttnMetaData(AttnMetaDataBase):
    seq_lens: list[int] = None
    seq_lens_ts: torch.Tensor | None = None
    seqs: List[BDSequence] = None
    kv_cache_layout: str = "unified"
    need_kv_cache_store: bool = True
    
    def __post_init__(self):
        if self.seq_lens_ts is not None and self.context_lens is not None:
            self.total_lens = self.seq_lens_ts + self.context_lens
    
    @property
    def total_num_seqs(self) -> int:
        return len(self.seqs) if self.seqs is not None else 0


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
    seqs: List[BDSequence] | None = None,
    seq_lens: list[int] | None = None,
    seq_lens_ts: torch.Tensor | None = None,
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
        seq_lens=seq_lens,
        seq_lens_ts=seq_lens_ts,
        seqs=seqs,
        kv_cache_layout=kv_cache_layout,
        need_kv_cache_store=need_kv_cache_store,
    )

def reset_bd_attn_metadata() -> None:
    global BD_ATTN_METADATA
    BD_ATTN_METADATA = BDAttnMetaData()