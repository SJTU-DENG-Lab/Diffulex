import torch

from typing import Callable
from dataclasses import dataclass, field


@dataclass
class AttnMetaDataBase:
    is_prefill: list[bool] = field(default_factory=lambda: [False])
    enforce_eager: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    page_tables: torch.Tensor | None = None
    status_table: torch.Tensor | None = None

    page_size: int = 32
    block_size: int = 32

    kv_cache_layout: str = "unified"
    token_merge_enabled: bool = False
    token_merge_mask: torch.Tensor | None = None
    token_merge_topk_ids: torch.Tensor | None = None
    token_merge_topk_probs: torch.Tensor | None = None
    token_merge_residual_probs: torch.Tensor | None = None
    token_merge_mask_token_id: int | None = None
    token_merge_renormalize: bool = True
    token_merge_mode: str = "dmax_topk"
    token_merge_weight: float = 1.0

    @property
    def num_reqs(self) -> int:
        return len(self.cu_seqlens_q) - 1

    @property
    def chunk_size(self) -> int:
        return self.block_size * self.buffer_size

    @property
    def need_kv_cache_store(self) -> bool:
        if is_warming_up() and self.slot_mapping.numel() > 0:
            return True

        return (self.slot_mapping >= 0).any() if self.enforce_eager else self.slot_mapping.numel() > 0


FN_TYPE_AttnMetaDataFetch = Callable[[], AttnMetaDataBase]

fetch_attn_metadata: FN_TYPE_AttnMetaDataFetch = ...


def set_fetch_fn_for_attn_metadata(fn: FN_TYPE_AttnMetaDataFetch) -> None:
    global fetch_attn_metadata
    fetch_attn_metadata = fn


WARMING_UP = False


def set_warming_up(is_warming_up: bool) -> None:
    global WARMING_UP
    WARMING_UP = is_warming_up


def is_warming_up() -> bool:
    return WARMING_UP


def reset_warming_up() -> None:
    global WARMING_UP
    WARMING_UP = False
