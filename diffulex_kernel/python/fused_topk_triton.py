from __future__ import annotations

"""SGLang-style fused top-k draft for Diffulex.

This keeps the contract small:
1. compute one score row per token,
2. select top-k inside the same kernel,
3. require a Triton-capable CUDA path instead of silently falling back.

The kernel intentionally assumes the expert dimension is small enough to fit in
SRAM, which matches the common MoE inference case (tens to low hundreds of
experts) and keeps the draft readable.
"""

import torch
import triton
import triton.language as tl


def _validate_fused_topk_inputs(
    router_logits: torch.Tensor,
    top_k: int,
    scoring_func: str,
) -> int:
    if not router_logits.is_cuda:
        raise ValueError("fused_topk requires CUDA tensors.")
    if router_logits.dim() != 2:
        raise ValueError(
            f"fused_topk expects a 2D [num_tokens, num_experts] tensor, got {router_logits.shape}."
        )
    if router_logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            f"fused_topk only supports fp16/bf16/fp32 router logits, got {router_logits.dtype}."
        )
    if top_k <= 0:
        raise ValueError(f"fused_topk requires top_k > 0, got {top_k}.")
    if scoring_func not in {"softmax", "sigmoid"}:
        raise ValueError(f"Unsupported scoring function: {scoring_func!r}.")

    effective_topk = min(top_k, router_logits.shape[-1])
    if router_logits.shape[-1] > 4096:
        raise NotImplementedError(
            "This Triton top-k draft currently requires num_experts <= 4096."
        )
    return effective_topk


@triton.jit
def _fused_topk(
    logits_ptr,
    weights_ptr,
    ids_ptr,
    num_experts,
    stride_logits_m,
    stride_logits_n,
    stride_out_m,
    stride_out_k,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCORING_MODE: tl.constexpr,
    RENORMALIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < num_experts

    logits = tl.load(
        logits_ptr + pid * stride_logits_m + offs * stride_logits_n,
        mask=mask,
        other=-1.0e20,
    ).to(tl.float32)
    logits = tl.where(logits == logits, logits, -1.0e20)
    logits = tl.maximum(tl.minimum(logits, 1.0e20), -1.0e20)
    logits = tl.where(mask, logits, -1.0e20)

    if SCORING_MODE == 0:
        row_max = tl.max(logits, axis=0)
        exp_logits = tl.exp(logits - row_max)
        exp_logits = tl.where(mask, exp_logits, 0.0)
        row_denom = tl.sum(exp_logits, axis=0)
        scores = exp_logits / row_denom
    else:
        scores = 1.0 / (1.0 + tl.exp(-logits))
        scores = tl.where(mask, scores, 0.0)

    scores = tl.where(mask, scores, -1.0e20)
    selected_sum = tl.zeros((), dtype=tl.float32)

    for topk_idx in range(TOP_K):
        best_id = tl.argmax(scores, axis=0).to(tl.int32)
        best_score = tl.max(scores, axis=0)

        tl.store(
            weights_ptr + pid * stride_out_m + topk_idx * stride_out_k,
            best_score,
        )
        tl.store(
            ids_ptr + pid * stride_out_m + topk_idx * stride_out_k,
            best_id,
        )
        selected_sum += best_score
        scores = tl.where(offs == best_id, -1.0e20, scores)

    if RENORMALIZE:
        selected_sum = tl.maximum(selected_sum, 1e-20)
        for topk_idx in range(TOP_K):
            weight_ptr = weights_ptr + pid * stride_out_m + topk_idx * stride_out_k
            tl.store(weight_ptr, tl.load(weight_ptr) / selected_sum)


def _launch_fused_topk_kernels(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    scoring_func: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    effective_topk = _validate_fused_topk_inputs(
        router_logits=router_logits,
        top_k=top_k,
        scoring_func=scoring_func,
    )

    num_tokens, num_experts = router_logits.shape
    topk_weights = torch.empty(
        (num_tokens, effective_topk),
        dtype=torch.float32,
        device=router_logits.device,
    )
    topk_ids = torch.empty(
        (num_tokens, effective_topk),
        dtype=torch.int32,
        device=router_logits.device,
    )

    block_size = triton.next_power_of_2(num_experts)
    num_warps = 4
    if block_size >= 256:
        num_warps = 8
    if block_size >= 1024:
        num_warps = 16

    scoring_mode = {
        "softmax": 0,
        "sigmoid": 1,
    }[scoring_func]
    _fused_topk[(num_tokens,)](
        router_logits,
        topk_weights,
        topk_ids,
        num_experts,
        router_logits.stride(0),
        router_logits.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        TOP_K=effective_topk,
        BLOCK_SIZE=block_size,
        SCORING_MODE=scoring_mode,
        RENORMALIZE=renormalize,
        num_warps=num_warps,
    )
    #_validate_fused_topk_outputs(
    #    topk_weights=topk_weights,
    #    topk_ids=topk_ids,
    #    num_experts=num_experts,
    #)
    return topk_weights, topk_ids


def fused_topk(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    scoring_func: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _launch_fused_topk_kernels(
        router_logits=router_logits,
        top_k=top_k,
        renormalize=renormalize,
        scoring_func=scoring_func,
    )


def _validate_fused_group_limited_topk_inputs(
    router_logits: torch.Tensor,
    expert_bias: torch.Tensor,
    top_k: int,
    n_group: int,
    topk_group: int,
) -> int:
    if not router_logits.is_cuda or not expert_bias.is_cuda:
        raise ValueError("fused_grouped_topk requires CUDA tensors.")
    if router_logits.dim() != 2:
        raise ValueError(
            "fused_grouped_topk expects a 2D "
            f"[num_tokens, num_experts] tensor, got {router_logits.shape}."
        )
    if expert_bias.dim() != 1 or expert_bias.shape[0] != router_logits.shape[-1]:
        raise ValueError(
            "fused_grouped_topk expects expert_bias shape "
            f"[num_experts], got {expert_bias.shape} for logits {router_logits.shape}."
        )
    if router_logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            "fused_grouped_topk only supports fp16/bf16/fp32 router logits, "
            f"got {router_logits.dtype}."
        )
    if expert_bias.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            "fused_grouped_topk only supports fp16/bf16/fp32 expert bias, "
            f"got {expert_bias.dtype}."
        )
    if top_k <= 0:
        raise ValueError(f"fused_grouped_topk requires top_k > 0, got {top_k}.")
    if router_logits.shape[-1] > 4096:
        raise NotImplementedError("fused_grouped_topk currently requires num_experts <= 4096.")
    if n_group > 1024:
        raise NotImplementedError("fused_grouped_topk currently requires n_group <= 1024.")
    if n_group < 0 or topk_group < 0:
        raise ValueError("fused_grouped_topk requires non-negative group settings.")
    return min(top_k, router_logits.shape[-1])


@triton.jit
def _fused_group_limited_topk(
    logits_ptr,
    expert_bias_ptr,
    weights_ptr,
    ids_ptr,
    num_experts,
    stride_logits_m,
    stride_logits_n,
    stride_out_m,
    stride_out_k,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    N_GROUP: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    GROUP_BLOCK_SIZE: tl.constexpr,
    GROUP_LIMITED: tl.constexpr,
    ROUTED_SCALING_FACTOR: tl.constexpr,
    RENORMALIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    expert_mask = offs < num_experts

    logits = tl.load(
        logits_ptr + pid * stride_logits_m + offs * stride_logits_n,
        mask=expert_mask,
        other=-1.0e20,
    ).to(tl.float32)
    logits = tl.where(logits == logits, logits, -1.0e20)
    logits = tl.maximum(tl.minimum(logits, 1.0e20), -1.0e20)
    scores = 1.0 / (1.0 + tl.exp(-logits))
    scores = tl.where(expert_mask, scores, 0.0)

    bias = tl.load(expert_bias_ptr + offs, mask=expert_mask, other=0.0).to(tl.float32)
    routing_scores = tl.where(expert_mask, scores + bias, -1.0e20)

    if GROUP_LIMITED:
        experts_per_group = num_experts // N_GROUP
        group_ids = offs // experts_per_group
        group_offs = tl.arange(0, GROUP_BLOCK_SIZE)
        group_scores = tl.full((GROUP_BLOCK_SIZE,), -1.0e20, dtype=tl.float32)

        for group_idx in range(N_GROUP):
            in_group = expert_mask & (group_ids == group_idx)
            first_values = tl.where(in_group, routing_scores, -1.0e20)
            first_id = tl.argmax(first_values, axis=0).to(tl.int32)
            first_score = tl.max(first_values, axis=0)
            second_values = tl.where(in_group & (offs != first_id), routing_scores, -1.0e20)
            second_score = tl.max(second_values, axis=0)
            second_score = tl.where(experts_per_group > 1, second_score, 0.0)
            group_scores = tl.where(group_offs == group_idx, first_score + second_score, group_scores)

        selected_group_mask = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        for _ in range(TOPK_GROUP):
            best_group = tl.argmax(group_scores, axis=0).to(tl.int32)
            selected_group_mask = selected_group_mask | (group_ids == best_group).to(tl.int32)
            group_scores = tl.where(group_offs == best_group, -1.0e20, group_scores)
        routing_scores = tl.where(expert_mask & (selected_group_mask != 0), routing_scores, -1.0e20)

    selected_sum = tl.zeros((), dtype=tl.float32)
    for topk_idx in range(TOP_K):
        best_id = tl.argmax(routing_scores, axis=0).to(tl.int32)
        best_score = tl.max(tl.where(offs == best_id, scores, -1.0e20), axis=0)

        tl.store(weights_ptr + pid * stride_out_m + topk_idx * stride_out_k, best_score)
        tl.store(ids_ptr + pid * stride_out_m + topk_idx * stride_out_k, best_id)
        selected_sum += best_score
        routing_scores = tl.where(offs == best_id, -1.0e20, routing_scores)

    if RENORMALIZE:
        selected_sum = tl.maximum(selected_sum, 1e-20)
        for topk_idx in range(TOP_K):
            weight_ptr = weights_ptr + pid * stride_out_m + topk_idx * stride_out_k
            tl.store(weight_ptr, tl.load(weight_ptr) / selected_sum * ROUTED_SCALING_FACTOR)
    else:
        for topk_idx in range(TOP_K):
            weight_ptr = weights_ptr + pid * stride_out_m + topk_idx * stride_out_k
            tl.store(weight_ptr, tl.load(weight_ptr) * ROUTED_SCALING_FACTOR)


def fused_group_limited_topk(
    router_logits: torch.Tensor,
    expert_bias: torch.Tensor,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    effective_topk = _validate_fused_group_limited_topk_inputs(
        router_logits=router_logits,
        expert_bias=expert_bias,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
    )
    num_tokens, num_experts = router_logits.shape
    topk_weights = torch.empty(
        (num_tokens, effective_topk),
        dtype=torch.float32,
        device=router_logits.device,
    )
    topk_ids = torch.empty(
        (num_tokens, effective_topk),
        dtype=torch.int32,
        device=router_logits.device,
    )

    block_size = triton.next_power_of_2(num_experts)
    group_limited = (
        n_group > 0
        and topk_group > 0
        and n_group <= num_experts
        and num_experts % n_group == 0
    )
    group_block_size = triton.next_power_of_2(max(1, n_group)) if group_limited else 1
    num_warps = 4
    if block_size >= 256:
        num_warps = 8
    if block_size >= 1024:
        num_warps = 16

    _fused_group_limited_topk[(num_tokens,)](
        router_logits,
        expert_bias,
        topk_weights,
        topk_ids,
        num_experts,
        router_logits.stride(0),
        router_logits.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        TOP_K=effective_topk,
        BLOCK_SIZE=block_size,
        N_GROUP=max(1, n_group),
        TOPK_GROUP=min(max(1, topk_group), max(1, n_group)),
        GROUP_BLOCK_SIZE=group_block_size,
        GROUP_LIMITED=group_limited,
        ROUTED_SCALING_FACTOR=float(routed_scaling_factor),
        RENORMALIZE=renormalize and effective_topk > 1,
        num_warps=num_warps,
    )
    return topk_weights, topk_ids
