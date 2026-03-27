from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TopKOutput:
    weights: torch.Tensor
    ids: torch.Tensor
    router_logits: torch.Tensor


class TopKRouter(nn.Module):
    """Top-k expert selection for MoE inference."""

    def __init__(
        self,
        top_k: int,
        *,
        renormalize: bool = True,
        scoring_func: str = "softmax",
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.renormalize = renormalize
        self.scoring_func = scoring_func

    def forward(self, router_logits: torch.Tensor) -> TopKOutput:
        if self.scoring_func == "softmax":
            routing_scores = F.softmax(router_logits, dim=-1, dtype=torch.float)
        elif self.scoring_func == "sigmoid":
            routing_scores = torch.sigmoid(router_logits.float())
        else:
            raise ValueError(f"Unsupported scoring function: {self.scoring_func!r}.")

        top_k = min(self.top_k, routing_scores.shape[-1])
        topk_weights, topk_ids = torch.topk(routing_scores, top_k, dim=-1)
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        return TopKOutput(
            weights=topk_weights,
            ids=topk_ids,
            router_logits=router_logits,
        )


__all__ = ["TopKOutput", "TopKRouter"]
