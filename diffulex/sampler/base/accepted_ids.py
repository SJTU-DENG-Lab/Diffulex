from __future__ import annotations

import torch


class SemiCompleteAcceptedIdsMixin:
    accepted_ids_default_threshold: float = 0.95
    accepted_ids_use_block_threshold: bool = True

    def _accepted_ids_threshold(self, block, **kwargs) -> float:
        if self.accepted_ids_use_block_threshold and hasattr(block, "thresholds"):
            return float(block.thresholds.accept_threshold)
        return float(kwargs.get("threshold", self.accepted_ids_default_threshold))

    @staticmethod
    def _should_force_top1_transfer(block) -> bool:
        prev_block = getattr(block, "prev_block", None)
        if prev_block is None:
            return True
        should_force_decode_topk = getattr(block, "should_force_decode_topk", None)
        if should_force_decode_topk is not None:
            return bool(should_force_decode_topk)
        return bool(getattr(prev_block, "is_semi_complete", False))

    def _compute_accepted_ids(
        self,
        block,
        confidence: torch.Tensor,
        initial_confidence: torch.Tensor,
        sampled_tokens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del sampled_tokens
        accept_threshold = self._accepted_ids_threshold(block, **kwargs)
        high_conf_indices = torch.where(initial_confidence > accept_threshold)[0]
        if self._should_force_top1_transfer(block):
            top1_idx = (
                torch.topk(confidence, 1)[1]
                if len(high_conf_indices) == 0
                else torch.tensor([], device=confidence.device, dtype=torch.long)
            )
            return torch.unique(torch.cat([top1_idx, high_conf_indices]))
        return high_conf_indices
