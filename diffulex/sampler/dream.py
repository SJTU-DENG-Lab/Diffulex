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
        if pre_block_complete:
            if len(high_conf_indices) == 0:
                number_transfer_tokens = 1
                _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                return transfer_index
            transfer_index = torch.tensor([], device=sampled_tokens.device, dtype=torch.long)
            return torch.unique(torch.cat([transfer_index, high_conf_indices]))
        return high_conf_indices
