import torch

from diffulex.sampler.auto_sampler import AutoSampler
from diffulex.sampler.base import DllmSamplerNoShiftBase


@AutoSampler.register("sdar")
class SDARSampler(DllmSamplerNoShiftBase):
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
        high_conf_indices = torch.where(initial_confidence > threshold)[0]
        if block.should_force_decode_topk:
            topk_idx = (
                torch.topk(confidence, 1)[1]
                if len(high_conf_indices) == 0
                else torch.tensor([], device=confidence.device, dtype=torch.long)
            )
            return torch.unique(torch.cat([topk_idx, high_conf_indices]))
        return high_conf_indices
