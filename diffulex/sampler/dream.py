import torch

from diffulex.sampler.auto_sampler import AutoSampler
from diffulex.sampler.base import DllmSamplerShiftBase


@AutoSampler.register("dream")
class DreamSampler(DllmSamplerShiftBase):
    def _compute_accepted_ids(
        self,
        block,
        confidence: torch.Tensor,
        initial_confidence: torch.Tensor,
        sampled_tokens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        accept_threshold = block.thresholds.accept_threshold
        pre_block_complete = block.prev_block.is_semi_complete if block.prev_block else True
        high_conf_indices = torch.where(initial_confidence > accept_threshold)[0]
        # Keep Dream's shifting behavior: only force a top-1 transfer token
        # once the previous block is semi-complete (or for the initial block).
        if pre_block_complete:
            topk_idx = (
                torch.topk(confidence, 1)[1]
                if len(high_conf_indices) == 0
                else torch.tensor([], device=confidence.device, dtype=torch.long)
            )
            return torch.unique(torch.cat([topk_idx, high_conf_indices]))
        return high_conf_indices
