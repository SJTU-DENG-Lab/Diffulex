from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.distributed as dist

from diffulex.moe.dispatcher.datatype import CombineInput, DispatchOutput
from diffulex.utils.parallelism import get_model_parallelism_metadata


class MoERunner(nn.Module, ABC):
    """Runs MoE MLP GEMMs efficiently."""

    def __init__(
        self,
        *,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        hidden_act: str,
        w13: torch.Tensor,
        w2: torch.Tensor,
        local_expert_start: int = 0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.hidden_act = hidden_act
        self.w13 = w13
        self.w2 = w2
        self.local_expert_start = local_expert_start
        self.layout = get_model_parallelism_metadata()

    def _all_reduce_output_if_needed(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.layout.world_size <= 1:
            return hidden_states

        # Current Diffulex only supports:
        # 1. TP-only: world == TP group
        # 2. pure EP with replicated-token semantics: world == EP group
        # 3. tp == ep: the same ranks serve as both TP and EP identities
        #
        # Under these constraints, reducing on the default process group is the
        # correct aggregation step for the standard non-A2A path.
        dist.all_reduce(hidden_states)
        return hidden_states

    @abstractmethod
    def forward(self, dispatch_output: DispatchOutput) -> CombineInput:
        raise NotImplementedError
