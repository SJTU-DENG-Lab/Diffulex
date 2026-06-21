from __future__ import annotations

import torch

from dataclasses import dataclass

from diffulex.mixin.multi_block.attention_metadata import MultiBlockAttnMetaDataMixin


@dataclass
class TokenMergeAttnMetaDataTemplate(MultiBlockAttnMetaDataMixin):
    token_merge_enabled: bool = False
    token_merge_mask: torch.Tensor | None = None
    token_merge_topk_ids: torch.Tensor | None = None
    token_merge_topk_probs: torch.Tensor | None = None
    token_merge_residual_probs: torch.Tensor | None = None
    token_merge_mask_token_id: int | None = None
    token_merge_renormalize: bool = True
    token_merge_mode: str = "dmax_topk"
    token_merge_weight: float = 1.0

    def init_token_merge(
        self,
        merge_mask: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
        topk_probs: torch.Tensor | None = None,
        residual_probs: torch.Tensor | None = None,
        mask_token_id: int | None = None,
        renormalize: bool = True,
        mode: str = "dmax_topk",
        weight: float = 1.0,
        enabled: bool | None = None,
    ) -> None:
        self.token_merge_mask = merge_mask
        self.token_merge_topk_ids = topk_ids
        self.token_merge_topk_probs = topk_probs
        self.token_merge_residual_probs = residual_probs
        self.token_merge_mask_token_id = mask_token_id
        self.token_merge_renormalize = renormalize
        self.token_merge_mode = mode
        self.token_merge_weight = weight
        metadata_complete = (
            merge_mask is not None
            and topk_ids is not None
            and topk_probs is not None
            and residual_probs is not None
            and bool(merge_mask.numel())
        )
        if enabled is not None:
            self.token_merge_enabled = bool(enabled) and metadata_complete
        else:
            self.token_merge_enabled = metadata_complete

    def reset_token_merge(self) -> None:
        self.init_token_merge()
