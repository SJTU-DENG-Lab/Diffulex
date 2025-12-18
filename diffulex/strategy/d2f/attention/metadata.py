import torch

from typing import List
from dataclasses import dataclass

from diffulex.attention.metadata import AttnMetaDataBase
from diffulex.strategy.d2f.engine.sequence import D2FSequence


@dataclass
class D2FAttnMetaData(AttnMetaDataBase):
    seq_lens: list[int] = None
    seq_lens_ts: torch.Tensor | None = None
    seqs: List[D2FSequence] = None
    kv_cache_layout: str = "unified"
    need_kv_cache_store: bool = True
    
    def __post_init__(self):
        if self.seq_lens_ts is not None and self.context_lens is not None:
            self.total_lens = self.seq_lens_ts + self.context_lens
    
    @property
    def total_num_seqs(self) -> int:
        return len(self.seqs) if self.seqs is not None else 0


D2F_ATTN_METADATA = D2FAttnMetaData()

def fetch_d2f_attn_metadata() -> D2FAttnMetaData:
    return D2F_ATTN_METADATA

def set_d2f_attn_metadata(
    is_prefill: bool = False,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping: torch.Tensor | None = None,
    context_lens: torch.Tensor | None = None,
    block_tables: torch.Tensor | None = None,
    seqs: List[D2FSequence] | None = None,
    seq_lens: list[int] | None = None,
    seq_lens_ts: torch.Tensor | None = None,
    kv_cache_layout: str = "unified",
    need_kv_cache_store: bool = True,
    diffusion_block_size: int = 32,
    decode_mode: str = "varlen",
    attn_type: str = "full_attention",
) -> None:
    global D2F_ATTN_METADATA
    D2F_ATTN_METADATA = D2FAttnMetaData(
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
        diffusion_block_size=diffusion_block_size,
        decode_mode=decode_mode,
        attn_type=attn_type,
    )

def reset_d2f_attn_metadata() -> None:
    global D2F_ATTN_METADATA
    D2F_ATTN_METADATA = D2FAttnMetaData()