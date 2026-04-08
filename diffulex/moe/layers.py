from __future__ import annotations

import re

import torch
import torch.nn as nn

from diffulex.layer.linear import ReplicatedLinear, divide
from diffulex.moe.config import (
    get_moe_intermediate_size,
    get_num_experts,
    get_num_experts_per_tok,
    get_norm_topk_prob,
)
from diffulex.moe.dispatcher import build_dispatcher
from diffulex.moe.runner import build_runner
from diffulex.moe.topk import build_topk_router
from diffulex.utils.checkpoint import LoadContext, ResolvedWeight
from diffulex.utils.parallelism import get_model_parallelism_metadata

"""
Note:
Currently supported layouts:
1. TP only: tp_size > 1, ep_size = 1
2. pure EP: ep_size > 1, tp_size = 1, world_size == ep_size
3. full-expert TP == EP: ep_size > 1, tp_size == ep_size, world_size == ep_size == tp_size
"""


class FusedMoE(nn.Module):
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
            raise ValueError(f"Only silu experts are supported right now, got {hidden_act!r}.")

        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        layout = get_model_parallelism_metadata()
        self.world_size = layout.world_size
        self.global_rank = layout.global_rank
        self.tp_size = layout.tp_size
        self.ep_size = layout.ep_size
        self.tp_enabled = self.tp_size > 1
        self.ep_enabled = self.ep_size > 1

        if self.tp_enabled and not self.ep_enabled:
            self.expert_tp_size = self.tp_size
            self.tp_rank = layout.tp_rank
            self.ep_rank = layout.ep_rank
            self.local_intermediate_size = divide(self.intermediate_size, self.tp_size)
            self.num_local_experts = self.num_experts
            dispatcher_impl = "trivial"
        elif self.ep_enabled:
            self.expert_tp_size = 1
            self.tp_rank = layout.tp_rank
            self.ep_rank = layout.ep_rank
            self.local_intermediate_size = self.intermediate_size
            self.num_local_experts = divide(self.num_experts, self.ep_size)
            dispatcher_impl = "trivial"
        else:
            self.expert_tp_size = 1
            self.tp_rank = layout.tp_rank
            self.ep_rank = layout.ep_rank
            self.local_intermediate_size = self.intermediate_size
            self.num_local_experts = self.num_experts
            dispatcher_impl = "trivial"

        self.local_expert_start = self.ep_rank * self.num_local_experts
        self.local_expert_end = self.local_expert_start + self.num_local_experts
        self.active_expert_ids = list(range(self.local_expert_start, self.local_expert_end))

        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)
        self.w13 = nn.Parameter(
            torch.empty(self.num_local_experts, self.local_intermediate_size * 2, hidden_size)
        )
        self.w13.weight_loader = self.w13_weight_loader
        self.w2 = nn.Parameter(
            torch.empty(self.num_local_experts, hidden_size, self.local_intermediate_size)
        )
        self.w2.weight_loader = self.w2_weight_loader

        self.router = build_topk_router(
            "trivial",
            top_k=top_k,
            renormalize=norm_topk_prob,
            scoring_func="softmax",
        )
        self.runner = build_runner(
            "trivial",
            num_experts=num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            top_k=top_k,
            hidden_act=hidden_act,
            w13=self.w13,
            w2=self.w2,
        )
        self.dispatcher = build_dispatcher(
            dispatcher_impl,
            num_experts=self.num_experts,
            top_k=self.top_k,
            num_local_experts=self.num_local_experts,
            ep_size=self.ep_size,
            local_expert_start=self.local_expert_start,
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

    def owns_global_expert(self, global_expert_id: int) -> bool:
        return self.local_expert_start <= global_expert_id < self.local_expert_end

    def global_to_local_expert_id(self, global_expert_id: int) -> int:
        if not self.owns_global_expert(global_expert_id):
            raise IndexError(f"Global expert id {global_expert_id} is not owned by this rank.")
        return global_expert_id - self.local_expert_start

    def _select_local_expert_weights(self, loaded_weight: torch.Tensor) -> torch.Tensor:
        if loaded_weight.size(0) == self.num_experts:
            return loaded_weight[self.local_expert_start : self.local_expert_end]
        if loaded_weight.size(0) == self.num_local_experts:
            return loaded_weight
        raise ValueError(
            f"Unexpected expert weight shape: {loaded_weight.shape}. "
            f"Expected 0-dim to be {self.num_experts} or {self.num_local_experts}."
        )

    def _slice_expert_input_shard(self, loaded_weight: torch.Tensor, *, dim: int) -> torch.Tensor:
        if self.expert_tp_size == 1:
            return loaded_weight

        shard_size = divide(loaded_weight.size(dim), self.expert_tp_size)
        start_idx = self.tp_rank * shard_size
        if dim == 0:
            return loaded_weight[start_idx : start_idx + shard_size]
        if dim == 1:
            return loaded_weight[:, start_idx : start_idx + shard_size]
        if dim == 2:
            return loaded_weight[:, :, start_idx : start_idx + shard_size]
        raise ValueError(f"Unsupported dim for sharding: {dim}")

    def w13_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if loaded_weight.shape == param.data.shape:
            param.data.copy_(loaded_weight)
            return

        local_expert_weights = self._select_local_expert_weights(loaded_weight)
        if local_expert_weights.size(1) == param.data.size(1):
            param.data.copy_(local_expert_weights)
            return

        shard_size = self.local_intermediate_size
        gate_weight = self._slice_expert_input_shard(
            local_expert_weights[:, : self.intermediate_size],
            dim=1,
        )
        up_weight = self._slice_expert_input_shard(
            local_expert_weights[:, self.intermediate_size :],
            dim=1,
        )
        param.data[:, :shard_size].copy_(gate_weight)
        param.data[:, shard_size:].copy_(up_weight)

    def w2_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if loaded_weight.shape == param.data.shape:
            param.data.copy_(loaded_weight)
            return

        local_expert_weights = self._select_local_expert_weights(loaded_weight)
        if local_expert_weights.size(2) == param.data.size(2):
            param.data.copy_(local_expert_weights)
            return

        local_weight = self._slice_expert_input_shard(local_expert_weights, dim=2)
        param.data.copy_(local_weight)

    def _load_w13_expert_weight(
        self,
        loaded_weight: torch.Tensor,
        *,
        local_expert_idx: int,
        shard_id: str,
    ) -> None:
        local_weight = self._slice_expert_input_shard(loaded_weight, dim=0)
        local_offset = 0 if shard_id == "gate_proj" else self.local_intermediate_size
        self.w13.data[
            local_expert_idx,
            local_offset : local_offset + self.local_intermediate_size,
        ].copy_(local_weight)

    def _load_w2_expert_weight(
        self,
        loaded_weight: torch.Tensor,
        *,
        local_expert_idx: int,
    ) -> None:
        local_weight = self._slice_expert_input_shard(loaded_weight, dim=1)
        self.w2.data[local_expert_idx].copy_(local_weight)

    def resolve_checkpoint_weight(self, suffix: str, ctx: LoadContext) -> ResolvedWeight | None:
        if suffix == "gate.weight":
            return ResolvedWeight(param=self.gate.weight)
        if suffix == "w13":
            return ResolvedWeight(param=self.w13)
        if suffix == "w2":
            return ResolvedWeight(param=self.w2)

        match = re.fullmatch(r"experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", suffix)
        if match is None:
            return None

        expert_idx = int(match.group(1))
        if not self.owns_global_expert(expert_idx):
            return ResolvedWeight(skip=True)

        local_expert_idx = self.global_to_local_expert_id(expert_idx)
        proj_name = match.group(2)
        if proj_name in ("gate_proj", "up_proj"):
            return ResolvedWeight(
                loader=lambda loaded_weight, local_expert_idx=local_expert_idx, proj_name=proj_name: self._load_w13_expert_weight(
                    loaded_weight,
                    local_expert_idx=local_expert_idx,
                    shard_id=proj_name,
                )
            )
        return ResolvedWeight(
            loader=lambda loaded_weight, local_expert_idx=local_expert_idx: self._load_w2_expert_weight(
                loaded_weight,
                local_expert_idx=local_expert_idx,
            )
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape
        flat_hidden_states = hidden_states.reshape(-1, original_shape[-1])

        router_logits = self.gate(flat_hidden_states)
        topk_output = self.router(router_logits)
        dispatch_output = self.dispatcher.dispatch(
            flat_hidden_states,
            topk_output,
            active_expert_ids=self.active_expert_ids,
        )
        combine_input = self.runner(dispatch_output)
        final_hidden_states = self.dispatcher.combine(combine_input)

        return final_hidden_states.reshape(original_shape), router_logits


__all__ = ["FusedMoE"]
