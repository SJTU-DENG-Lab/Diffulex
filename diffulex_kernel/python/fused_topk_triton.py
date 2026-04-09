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


def _validate_fused_topk_outputs(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    num_experts: int,
) -> None:
    invalid_id_mask = (topk_ids < 0) | (topk_ids >= num_experts)
    if invalid_id_mask.any():
        invalid_ids = topk_ids[invalid_id_mask][:16].detach().cpu().tolist()
        raise RuntimeError(
            "fused_topk produced out-of-range expert ids. "
            f"num_experts={num_experts}, sample_invalid_ids={invalid_ids}."
        )

    if not torch.isfinite(topk_weights).all():
        raise RuntimeError("fused_topk produced non-finite routing weights.")


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
    _validate_fused_topk_outputs(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_experts=num_experts,
    )
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
