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
    need_kv_cache_store_static: bool | None = None
    has_prefill_static: bool | None = None
    all_prefill_static: bool | None = None
    context_lens: torch.Tensor | None = None
    page_tables: torch.Tensor | None = None
    status_table: torch.Tensor | None = None
    mask_prefix_hole: bool = False
    prefix_causal: bool = False

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
    def has_prefill(self) -> bool:
        if self.has_prefill_static is not None:
            return bool(self.has_prefill_static)
        has_prefill, _ = infer_prefill_flags(self.is_prefill)
        return bool(has_prefill)

    @property
    def all_prefill(self) -> bool:
        if self.all_prefill_static is not None:
            return bool(self.all_prefill_static)
        _, all_prefill = infer_prefill_flags(self.is_prefill)
        return bool(all_prefill)

    @property
    def need_kv_cache_store(self) -> bool:
        if self.need_kv_cache_store_static is not None:
            return bool(self.need_kv_cache_store_static)
        if self.slot_mapping is None:
            return False
        if is_warming_up() and self.slot_mapping.numel() > 0:
            return True

        return self.slot_mapping.numel() > 0


FN_TYPE_AttnMetaDataFetch = Callable[[], AttnMetaDataBase]

fetch_attn_metadata: FN_TYPE_AttnMetaDataFetch = ...


def set_fetch_fn_for_attn_metadata(fn: FN_TYPE_AttnMetaDataFetch) -> None:
    global fetch_attn_metadata
    fetch_attn_metadata = fn


def infer_prefill_flags(is_prefill) -> tuple[bool, bool]:
    if isinstance(is_prefill, bool):
        return is_prefill, is_prefill

    try:
        flags = [bool(flag) for flag in is_prefill]
    except TypeError:
        return False, False
    if not flags:
        return False, False
    return any(flags), all(flags)


WARMING_UP = False


def set_warming_up(is_warming_up: bool) -> None:
    global WARMING_UP
    WARMING_UP = is_warming_up


def is_warming_up() -> bool:
    return WARMING_UP


def reset_warming_up() -> None:
    global WARMING_UP
    WARMING_UP = False
