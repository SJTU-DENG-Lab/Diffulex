from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from diffulex.moe.topk import TopKOutput


@dataclass(frozen=True)
class DispatchOutput:
    hidden_states: torch.Tensor
    topk_output: TopKOutput
    num_tokens: int
    expert_token_indices: tuple[torch.Tensor, ...] = ()
    expert_topk_slot_indices: tuple[torch.Tensor, ...] = ()
    hidden_states_scale: torch.Tensor | None = None
    context: Any = None

@dataclass(frozen=True)
class CombineInput:
    hidden_states: torch.Tensor
    context: Any = None
