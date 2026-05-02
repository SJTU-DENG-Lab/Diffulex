from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from diffulex.moe.topk.output import TopKOutput


class TopKRouter(nn.Module, ABC):
    """Top-k expert selection for MoE inference."""

    def __init__(
        self,
        top_k: int,
        *,
        kernel_impl: str = "triton",
        renormalize: bool = True,
        scoring_func: str = "softmax",
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.kernel_impl = kernel_impl
        self.renormalize = renormalize
        self.scoring_func = scoring_func

    @abstractmethod
    def forward(self, router_logits: torch.Tensor) -> TopKOutput:
        raise NotImplementedError
