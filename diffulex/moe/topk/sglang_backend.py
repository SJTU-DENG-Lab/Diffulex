from __future__ import annotations

from diffulex.moe.topk.output import TopKOutput

try:
    from sglang.srt.layers.moe.topk import biased_grouped_topk_gpu as sglang_biased_grouped_topk
    from sglang.srt.layers.moe.topk import fused_topk as sglang_fused_topk
except Exception:
    sglang_biased_grouped_topk = None
    sglang_fused_topk = None


def run_sglang_fused_topk(
    *,
    router_logits,
    top_k: int,
    renormalize: bool,
    scoring_func: str,
) -> TopKOutput | None:
    if sglang_fused_topk is None:
        return None

    topk_weights, topk_ids = sglang_fused_topk(
        hidden_states=router_logits,
        gating_output=router_logits,
        topk=top_k,
        renormalize=renormalize,
        scoring_func=scoring_func,
    )
    return TopKOutput(
        weights=topk_weights,
        ids=topk_ids,
        router_logits=router_logits,
    )


def run_sglang_biased_grouped_topk(
    *,
    router_logits,
    correction_bias,
    top_k: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
) -> TopKOutput | None:
    if sglang_biased_grouped_topk is None:
        return None

    topk_weights, topk_ids = sglang_biased_grouped_topk(
        hidden_states=router_logits,
        gating_output=router_logits,
        correction_bias=correction_bias,
        topk=top_k,
        renormalize=renormalize,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )
    return TopKOutput(
        weights=topk_weights,
        ids=topk_ids,
        router_logits=router_logits,
    )


__all__ = ["run_sglang_biased_grouped_topk", "run_sglang_fused_topk"]
