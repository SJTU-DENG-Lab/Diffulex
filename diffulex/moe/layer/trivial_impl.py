import re

import torch
import torch.nn as nn

from diffulex.layer.linear import ReplicatedLinear
from diffulex.moe.layer.base import FusedMoE
from diffulex.utils.checkpoint import LoadContext, ResolvedWeight


class TrivialFusedMoE(FusedMoE):
    """
    single device moe layer
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

        self.gate = ReplicatedLinear(hidden_size, self.num_experts, bias=False)
        self.w13 = nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_size * 2, hidden_size)
        )
        self.w2 = nn.Parameter(
            torch.empty(self.num_experts, hidden_size, self.intermediate_size)
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

        return final_hidden_states.reshape(original_shape), router_logits

    def load_w1(self, loaded_weight: torch.Tensor, expert_idx: int) -> None:
        self.w13.data[expert_idx, 0 : self.intermediate_size].copy_(loaded_weight)

    def load_w3(self, loaded_weight: torch.Tensor, expert_idx: int) -> None:
        self.w13.data[expert_idx, self.intermediate_size : 2 * self.intermediate_size].copy_(loaded_weight)

    def load_w2(self, loaded_weight: torch.Tensor, expert_idx: int) -> None:
        self.w2.data[expert_idx].copy_(loaded_weight)

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


__all__ = ["TrivialFusedMoE"]
