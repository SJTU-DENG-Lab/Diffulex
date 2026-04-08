from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
import torch.nn as nn

from diffulex.moe.dispatcher.datatype import CombineInput, DispatchOutput
from diffulex.moe.topk import TopKOutput


class TokenDispatcher(nn.Module, ABC):
    """Dispatches tokens to experts / Combines tokens from experts efficiently."""

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        *,
        num_local_experts: int | None = None,
        local_expert_start: int = 0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts if num_local_experts is not None else num_experts
        self.local_expert_start = local_expert_start
        self.local_expert_end = self.local_expert_start + self.num_local_experts

    @abstractmethod
    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        *,
        active_expert_ids: Sequence[int] | None = None,
        hidden_states_scale: torch.Tensor | None = None,
    ) -> DispatchOutput:
        raise NotImplementedError

    @abstractmethod
    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        raise NotImplementedError
