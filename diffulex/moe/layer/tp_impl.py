import re

import torch
import torch.nn as nn
import torch.distributed as dist

from diffulex.layer.linear import ReplicatedLinear, divide
from diffulex.moe.layer.base import FusedMoE
from diffulex.utils.checkpoint import LoadContext, ResolvedWeight
from diffulex.utils.parallelism import get_tp_rank, get_tp_world_size


class TPFusedMoE(FusedMoE):
    """
    if tp is on and ep is off, will shard expert weight
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        *,
        hidden_act: str = "silu",
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__(
            hidden_size,
            intermediate_size,
            num_experts,
            top_k,
            hidden_act=hidden_act,
            norm_topk_prob=norm_topk_prob
        )
        
        self.tp_rank = get_tp_rank()
        self.tp_size = get_tp_world_size()
        self.local_intermediate_size = divide(self.intermediate_size, self.tp_size)

        self.gate = ReplicatedLinear(hidden_size, self.num_experts, bias=False)
        self.w13 = nn.Parameter(
            torch.empty(self.num_experts, self.local_intermediate_size * 2, hidden_size)
        )
        self.w2 = nn.Parameter(
            torch.empty(self.num_experts, hidden_size, self.local_intermediate_size)
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape
        flat_hidden_states = hidden_states.reshape(-1, original_shape[-1])
        
        router_logits = self.gate(flat_hidden_states)
        topk_output = self.router(router_logits)
        topk_weights = topk_output.weights
        topk_ids = topk_output.ids
        final_hidden_states = self.expert_gemm(
            impl="triton",
            hidden_states=flat_hidden_states,
            w13=self.w13,
            w2=self.w2,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            local_expert_start=0,
            hidden_act=self.hidden_act,
        )
        dist.all_reduce(final_hidden_states)

        return final_hidden_states.reshape(original_shape), router_logits

    def _get_tp_shard_range(self):
        start = self.tp_rank * self.local_intermediate_size
        end = start + self.local_intermediate_size
        return start, end
    
    def load_w1(self, loaded_weight: torch.Tensor, expert_idx: int) -> None:
        start, end = self._get_tp_shard_range()
        # loaded_weight: [intermediate_size, hidden_size]
        shard = loaded_weight[start:end, :]
        self.w13.data[expert_idx, 0:self.local_intermediate_size].copy_(shard)
    
    def load_w3(self, loaded_weight: torch.Tensor, expert_idx: int) -> None:
        start, end = self._get_tp_shard_range()
        # loaded_weight: [intermediate_size, hidden_size]
        shard = loaded_weight[start:end, :]
        self.w13.data[expert_idx, self.local_intermediate_size:2*self.local_intermediate_size].copy_(shard)
    
    def load_w2(self, loaded_weight: torch.Tensor, expert_idx: int) -> None:
        start, end = self._get_tp_shard_range()
        target = self.w2.data[expert_idx]  # [hidden_size, local_intermediate_size]
        if loaded_weight.shape == (self.hidden_size, self.intermediate_size):
            shard = loaded_weight[:, start:end]
        elif loaded_weight.shape == (self.intermediate_size, self.hidden_size):
            shard = loaded_weight[start:end, :].transpose(0, 1).contiguous()
        else:
            raise ValueError(
                f"Unexpected down_proj weight shape: {loaded_weight.shape}, "
                f"target shape: {target.shape}"
            )
        target.copy_(shard)

    def resolve_checkpoint_weight(self, suffix: str, ctx: LoadContext) -> ResolvedWeight | None:
        match = re.fullmatch(r"experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", suffix)
        if match is None:
            return None

        expert_idx = int(match.group(1))
        proj_name = match.group(2)

        if proj_name == "gate_proj":
            return ResolvedWeight(
                loader=lambda loaded_weight, expert_idx=expert_idx: self.load_w1(
                    loaded_weight,
                    expert_idx,
                )
            )

        if proj_name == "up_proj":
            return ResolvedWeight(
                loader=lambda loaded_weight, expert_idx=expert_idx: self.load_w3(
                    loaded_weight,
                    expert_idx,
                )
            )
        
        if proj_name == "down_proj":
            return ResolvedWeight(
                loader=lambda loaded_weight, expert_idx=expert_idx: self.load_w2(
                    loaded_weight,
                    expert_idx,
                )
            )
        
        return None

__all__ = ["TPFusedMoE"]
