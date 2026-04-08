from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from diffulex.moe.dispatcher.datatype import CombineInput, DispatchOutput


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

    @abstractmethod
    def forward(self, dispatch_output: DispatchOutput) -> CombineInput:
        raise NotImplementedError
