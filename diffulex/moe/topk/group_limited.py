from __future__ import annotations

import os
import torch
from einops import rearrange, repeat

from diffulex.moe.topk.base import TopKRouter
from diffulex.moe.topk.output import TopKOutput
from diffulex_kernel import fused_group_limited_topk


_VLLM_GROUPED_TOPK_UNAVAILABLE = False


def _vllm_grouped_topk(
    router_logits: torch.Tensor,
    expert_bias: torch.Tensor,
    *,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    global _VLLM_GROUPED_TOPK_UNAVAILABLE
    if (
        _VLLM_GROUPED_TOPK_UNAVAILABLE
        or os.getenv("DIFFULEX_DISABLE_VLLM_GROUPED_TOPK", "0") == "1"
        or not router_logits.is_cuda
        or n_group <= 0
        or topk_group <= 0
        or n_group > 32
        or top_k > 32
    ):
        return None
    try:
        from vllm import _custom_ops as ops

        return ops.grouped_topk(
            router_logits,
            n_group,
            topk_group,
            top_k,
            renormalize,
            routed_scaling_factor,
            expert_bias,
            1,
        )
    except Exception:
        _VLLM_GROUPED_TOPK_UNAVAILABLE = True
        return None


def _tile_kernels_group_limited_topk(
    scores: torch.Tensor,
    *,
    top_k: int,
    n_group: int,
    topk_group: int,
) -> torch.Tensor | None:
    try:
        import tile_kernels  # type: ignore
    except Exception:
        return None

    if (
        n_group <= 0
        or topk_group <= 0
        or n_group > int(scores.shape[-1])
        or int(scores.shape[-1]) % n_group != 0
    ):
        try:
            return tile_kernels.moe.topk_gate(scores, top_k)
        except Exception:
            return None

    experts_per_group = int(scores.shape[-1]) // n_group
    scores_by_group = rearrange(scores, "token (group expert) -> token group expert", group=n_group)
    num_group_sum_topk = min(2, experts_per_group)
    num_topk_groups = min(topk_group, n_group)
    try:
        group_idx = tile_kernels.moe.topk_sum_and_topk_group_idx(
            scores_by_group,
            num_group_sum_topk,
            num_topk_groups,
        )
        group_mask = torch.zeros(
            (scores.shape[0], n_group),
            dtype=torch.bool,
            device=scores.device,
        )
        group_mask.scatter_(1, group_idx.to(torch.int64), True)
        score_mask = repeat(group_mask, "token group -> token (group expert)", expert=experts_per_group)
        masked_scores = scores.masked_fill(~score_mask, float("-inf"))
        return tile_kernels.moe.topk_gate(masked_scores, top_k)
    except Exception:
        return None


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
        rank_scores = scores + expert_bias
        topk_ids = None
        if (
            router_logits.is_cuda
            and os.getenv("DIFFULEX_MOE_TOPK_IMPL", "").lower() in {"tile", "tilekernels", "tile_kernels"}
        ):
            topk_ids = _tile_kernels_group_limited_topk(
                rank_scores,
                top_k=self.top_k,
                n_group=self.n_group,
                topk_group=self.topk_group,
            )
        if topk_ids is None:
            topk_ids = self._group_limited_topk(rank_scores)
        topk_ids = topk_ids.to(torch.int64)
        topk_weights = torch.gather(scores, dim=-1, index=topk_ids)
        if self.renormalize and self.top_k > 1:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return TopKOutput(weights=topk_weights, ids=topk_ids, router_logits=router_logits)

    def forward(self, router_logits: torch.Tensor) -> TopKOutput:
        if not router_logits.is_cuda or os.getenv("DIFFULEX_REFERENCE_MOE_ROUTER", "0") == "1":
            return self._forward_naive(router_logits)

        expert_bias = self._expert_bias_getter().to(router_logits.device, dtype=router_logits.dtype)
        topk_output = _vllm_grouped_topk(
            router_logits,
            expert_bias,
            top_k=self.top_k,
            n_group=self.n_group,
            topk_group=self.topk_group,
            routed_scaling_factor=self.routed_scaling_factor,
            renormalize=self.renormalize,
        )
        if topk_output is None:
            topk_output = fused_group_limited_topk(
                router_logits=router_logits,
                expert_bias=expert_bias,
                top_k=self.top_k,
                n_group=self.n_group,
                topk_group=self.topk_group,
                routed_scaling_factor=self.routed_scaling_factor,
                renormalize=self.renormalize,
            )
        topk_weights, topk_ids = topk_output
        return TopKOutput(weights=topk_weights, ids=topk_ids, router_logits=router_logits)
