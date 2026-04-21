from __future__ import annotations

import torch
from einops import rearrange, repeat

from diffulex.moe.topk.base import TopKRouter
from diffulex.moe.topk.output import TopKOutput
from diffulex_kernel import fused_group_limited_topk


class GroupLimitedTopKRouter(TopKRouter):
    def __init__(
        self,
        top_k: int,
        *,
        num_experts: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float = 1.0,
        renormalize: bool = True,
        scoring_func: str = "sigmoid",
        expert_bias_getter,
    ) -> None:
        super().__init__(top_k=top_k, renormalize=renormalize, scoring_func=scoring_func)
        if scoring_func != "sigmoid":
            raise NotImplementedError("GroupLimitedTopKRouter currently supports sigmoid scoring only.")
        self.num_experts = num_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self._expert_bias_getter = expert_bias_getter

    def _group_limited_topk(self, scores: torch.Tensor) -> torch.Tensor:
        if (
            self.n_group <= 0
            or self.topk_group <= 0
            or self.n_group > self.num_experts
            or self.num_experts % self.n_group != 0
        ):
            return torch.topk(scores, k=self.top_k, dim=-1, sorted=False).indices

        experts_per_group = self.num_experts // self.n_group
        scores_by_group = rearrange(
            scores,
            "token (group expert) -> token group expert",
            group=self.n_group,
        )
        group_scores = scores_by_group.topk(
            min(2, experts_per_group),
            dim=-1,
        ).values.sum(dim=-1)
        group_idx = torch.topk(group_scores, k=min(self.topk_group, self.n_group), dim=-1, sorted=False).indices
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, group_idx, True)
        score_mask = repeat(
            group_mask,
            "token group -> token (group expert)",
            expert=experts_per_group,
        )
        masked_scores = scores.masked_fill(~score_mask, float("-inf"))
        return torch.topk(masked_scores, k=self.top_k, dim=-1, sorted=False).indices

    def _forward_naive(self, router_logits: torch.Tensor) -> TopKOutput:
        scores = torch.sigmoid(router_logits.float()).to(router_logits.dtype)
        expert_bias = self._expert_bias_getter().to(scores.device, dtype=scores.dtype)
        topk_ids = self._group_limited_topk(scores + expert_bias)
        topk_weights = torch.gather(scores, dim=-1, index=topk_ids)
        if self.renormalize and self.top_k > 1:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return TopKOutput(weights=topk_weights, ids=topk_ids, router_logits=router_logits)

    def forward(self, router_logits: torch.Tensor) -> TopKOutput:
        if not router_logits.is_cuda:
            return self._forward_naive(router_logits)

        expert_bias = self._expert_bias_getter().to(router_logits.device, dtype=router_logits.dtype)
        topk_weights, topk_ids = fused_group_limited_topk(
            router_logits=router_logits,
            expert_bias=expert_bias,
            top_k=self.top_k,
            n_group=self.n_group,
            topk_group=self.topk_group,
            routed_scaling_factor=self.routed_scaling_factor,
            renormalize=self.renormalize,
        )
        return TopKOutput(weights=topk_weights, ids=topk_ids, router_logits=router_logits)
