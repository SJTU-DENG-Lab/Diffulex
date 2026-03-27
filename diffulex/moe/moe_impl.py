from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffulex.layer.activation import SiluAndMul
from diffulex.layer.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from diffulex.moe.config import (
    get_moe_intermediate_size,
    get_num_experts,
    get_num_experts_per_tok,
    get_norm_topk_prob,
)
from diffulex.moe.topk import TopKRouter


class ExpertMLP(nn.Module):
    """Dense expert used inside sparse MoE blocks."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(f"Only silu experts are supported right now, got {hidden_act!r}.")

        self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden_states = self.act_fn(torch.cat([gate, up], dim=-1))
        return self.down_proj(hidden_states)


class SparseMoEBlock(nn.Module):
    """Naive sparse MoE block matching SDAR-MoE checkpoint structure."""

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
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                ExpertMLP(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    hidden_act=hidden_act,
                )
                for _ in range(num_experts)
            ]
        )
        self.router = TopKRouter(top_k=top_k, renormalize=norm_topk_prob)

    @classmethod
    def from_config(cls, config) -> "SparseMoEBlock":
        return cls(
            hidden_size=config.hidden_size,
            intermediate_size=get_moe_intermediate_size(config),
            num_experts=get_num_experts(config),
            top_k=get_num_experts_per_tok(config),
            hidden_act=getattr(config, "hidden_act", "silu"),
            norm_topk_prob=get_norm_topk_prob(config),
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape
        hidden_size = original_shape[-1]
        flat_hidden_states = hidden_states.reshape(-1, hidden_size)

        router_logits = self.gate(flat_hidden_states)
        topk_output = self.router(router_logits)

        final_hidden_states = torch.zeros_like(flat_hidden_states)
        expert_mask = F.one_hot(topk_output.ids, num_classes=self.num_experts).permute(2, 1, 0)
        routing_weights = topk_output.weights.to(flat_hidden_states.dtype)

        for expert_idx in range(self.num_experts):
            topk_slot_idx, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue

            expert_input = flat_hidden_states[token_idx]
            expert_output = self.experts[expert_idx](expert_input)
            expert_output = expert_output * routing_weights[token_idx, topk_slot_idx].unsqueeze(-1)
            final_hidden_states.index_add_(0, token_idx, expert_output)

        return final_hidden_states.reshape(original_shape), router_logits


__all__ = ["ExpertMLP", "SparseMoEBlock"]
