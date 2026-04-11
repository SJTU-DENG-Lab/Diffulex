from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffulex.moe.config import (
    get_moe_intermediate_size,
    get_num_experts,
    get_num_experts_per_tok,
    get_norm_topk_prob,
)
from diffulex.moe.topk import build_topk_router
from diffulex_kernel import fused_moe

class FusedMoE(nn.Module, ABC):
    
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

        if hidden_act != "silu":
            raise NotImplementedError("only silu is supported currently")
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_act = hidden_act
        self.norm_topk_prob = norm_topk_prob

        self.router = build_topk_router(
            "triton",
            top_k=top_k,
            renormalize=norm_topk_prob,
            scoring_func="softmax",
        )

    @classmethod
    def from_config(cls, config) -> "FusedMoE":
        return cls(
            hidden_size=config.hidden_size,
            intermediate_size=get_moe_intermediate_size(config),
            num_experts=get_num_experts(config),
            top_k=get_num_experts_per_tok(config),
            hidden_act=getattr(config, "hidden_act", "silu"),
            norm_topk_prob=get_norm_topk_prob(config),
        )
    
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # return final_hidden_states, router_logits
        raise NotImplementedError
    
    def expert_gemm(
        self,
        impl: str,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        local_expert_start: int = 0,
        hidden_act: str = "silu"
    ) -> torch.Tensor:
        if impl == "triton":
            return fused_moe(
                hidden_states=hidden_states,
                w13=w13,
                w2=w2,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                local_expert_start=local_expert_start,
                hidden_act=hidden_act,
            )
        if impl == "trivial":
            num_tokens, hidden_size = hidden_states.shape
            num_local_experts = w13.shape[0]
            intermediate_size = w13.shape[1] // 2
            final_hidden_states = hidden_states.new_zeros((num_tokens, hidden_size))
            local_topk_ids = topk_ids.to(torch.int64) - int(local_expert_start)
            
            for token_idx in range(num_tokens):
                token_hidden = hidden_states[token_idx]
                token_out = torch.zeros(hidden_size, device=hidden_states.device, dtype=torch.float32)

                for slot_idx in range(topk_ids.shape[1]):
                    local_expert_idx = int(local_topk_ids[token_idx, slot_idx].item())

                    if local_expert_idx < 0 or local_expert_idx >= num_local_experts:
                        continue

                    weight = topk_weights[token_idx, slot_idx]
                    if weight.item() == 0:
                        continue

                    expert_w13 = w13[local_expert_idx]
                    gate_proj = expert_w13[:intermediate_size]
                    up_proj = expert_w13[intermediate_size:]
                    
                    gate = torch.matmul(token_hidden, gate_proj.transpose(0, 1))
                    up = torch.matmul(token_hidden, up_proj.transpose(0, 1))
                    activated = F.silu(gate) * up
                    
                    expert_out = torch.matmul(activated, w2[local_expert_idx].transpose(0, 1))
                    token_out += expert_out.float() * weight.float()
                
                final_hidden_states[token_idx] = token_out.to(hidden_states.dtype)

            return final_hidden_states


__all__ = ["FusedMoE"]
