import torch

from diffulex.sampler.auto_sampler import AutoSampler
from diffulex.sampler.base import DllmSamplerNoShiftBase


@AutoSampler.register("llada")
class LLaDASampler(DllmSamplerNoShiftBase):
    def _compute_accepted_ids(
        self,
        block,
        confidence: torch.Tensor,
        initial_confidence: torch.Tensor,
        sampled_tokens: torch.Tensor,
        *,
        threshold: float = 0.95,
        **kwargs,
    ) -> torch.Tensor:
        accept_threshold = block.thresholds.accept_threshold
        high_conf_indices = torch.where(initial_confidence > accept_threshold)[0]
        is_initial_block = block.block_id == 0 and block.prev_block is None
        if block.should_force_decode_topk or is_initial_block:
            topk_idx = (
                torch.topk(confidence, 1)[1]
                if len(high_conf_indices) == 0
                else torch.tensor([], device=confidence.device, dtype=torch.long)
            )
            return torch.unique(torch.cat([topk_idx, high_conf_indices]))
        return high_conf_indices
