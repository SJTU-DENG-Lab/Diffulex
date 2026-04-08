from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TopKOutput:
    weights: torch.Tensor
    ids: torch.Tensor
    router_logits: torch.Tensor
