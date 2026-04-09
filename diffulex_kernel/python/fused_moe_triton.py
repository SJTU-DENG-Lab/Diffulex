from __future__ import annotations

"""SGLang-inspired fused MoE draft for Diffulex.

Unlike SGLang's production kernel, this draft does not yet sort/pad tokens into
expert-major blocks. Instead, it keeps Diffulex's current dispatcher contract
and uses a simpler row-wise expert GEMM in Triton.
"""

import torch
import torch.nn.functional as F

import triton
import triton.language as tl


def _validate_fused_moe_inputs(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    hidden_act: str,
) -> None:
    if hidden_act != "silu":
        raise ValueError(f"Only silu experts are supported right now, got {hidden_act!r}.")
    if not hidden_states.is_cuda:
        raise ValueError("fused_moe requires CUDA tensors.")
    if hidden_states.dim() != 2 or topk_ids.dim() != 2 or topk_weights.dim() != 2:
        raise ValueError(
            "fused_moe expects hidden_states/topk_ids/topk_weights to be 2D tensors."
        )
    if hidden_states.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            f"fused_moe only supports fp16/bf16/fp32 activations, got {hidden_states.dtype}."
        )
    if w13.dtype != hidden_states.dtype or w2.dtype != hidden_states.dtype:
        raise TypeError(
            "fused_moe expects hidden_states, w13, and w2 to share the same dtype."
        )
    if topk_ids.shape != topk_weights.shape:
        raise ValueError(
            f"topk_ids and topk_weights must have the same shape, got {topk_ids.shape} and {topk_weights.shape}."
        )
    if w13.shape[0] != w2.shape[0] or w13.shape[2] != hidden_states.shape[1]:
        raise ValueError(
            "Weight shapes do not match hidden_states or local expert count."
        )
    if w13.shape[1] % 2 != 0:
        raise ValueError("fused_moe expects w13.shape[1] to be 2 * intermediate_size.")


@triton.jit
def _expert_gemm(
    a_ptr,
    expert_ids_ptr,
    w_ptr,
    out_ptr,
    num_rows,
    num_cols,
    k_dim,
    num_experts,
    stride_am,
    stride_ak,
    stride_e,
    stride_we,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = offs_m < num_rows

    expert_ids = tl.load(
        expert_ids_ptr + offs_m * stride_e,
        mask=row_mask,
        other=-1,
    ).to(tl.int32)
    valid_expert = (expert_ids >= 0) & (expert_ids < num_experts)
    expert_offsets = tl.where(valid_expert, expert_ids, 0).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)

    for k_start in range(0, k_dim, BLOCK_K):
        current_k = k_start + k_offsets
        k_mask = current_k < k_dim
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + current_k[None, :] * stride_ak,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            w_ptr
            + expert_offsets[:, None, None] * stride_we
            + offs_n[None, :, None] * stride_wn
            + current_k[None, None, :] * stride_wk,
            mask=row_mask[:, None, None]
            & valid_expert[:, None, None]
            & (offs_n[None, :, None] < num_cols)
            & k_mask[None, None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(a[:, None, :] * b, axis=2)

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=row_mask[:, None] & (offs_n[None, :] < num_cols),
    )


def _launch_fused_moe_kernels(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    local_expert_start: int,
    hidden_act: str,
) -> torch.Tensor:
    if hidden_states.shape[0] == 0 or topk_ids.shape[1] == 0:
        return hidden_states.new_zeros((hidden_states.shape[0], w2.shape[1]))

    _validate_fused_moe_inputs(
        hidden_states=hidden_states,
        w13=w13,
        w2=w2,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        hidden_act=hidden_act,
    )

    num_tokens, hidden_size = hidden_states.shape
    top_k = topk_ids.shape[1]
    num_slots = num_tokens * top_k
    num_local_experts = w13.shape[0]
    intermediate_twice = w13.shape[1]
    intermediate_size = intermediate_twice // 2

    local_ids = topk_ids.to(torch.int32) - int(local_expert_start)
    local_mask = (local_ids >= 0) & (local_ids < num_local_experts)
    local_ids = torch.where(local_mask, local_ids, torch.full_like(local_ids, -1))

    expanded_hidden_states = hidden_states.repeat_interleave(top_k, dim=0).contiguous()
    flat_local_ids = local_ids.reshape(-1).contiguous()

    gate_up = torch.empty(
        (num_slots, intermediate_twice),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    block_m = 8
    block_n = 64 if intermediate_twice >= 64 else triton.next_power_of_2(intermediate_twice)
    block_k = 32 if hidden_size >= 32 else triton.next_power_of_2(hidden_size)
    _expert_gemm[
        (triton.cdiv(num_slots, block_m), triton.cdiv(intermediate_twice, block_n))
    ](
        expanded_hidden_states,
        flat_local_ids,
        w13,
        gate_up,
        num_slots,
        intermediate_twice,
        hidden_size,
        num_local_experts,
        expanded_hidden_states.stride(0),
        expanded_hidden_states.stride(1),
        flat_local_ids.stride(0),
        w13.stride(0),
        w13.stride(1),
        w13.stride(2),
        gate_up.stride(0),
        gate_up.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )

    gate, up = gate_up.chunk(2, dim=-1)
    activated = (F.silu(gate) * up).contiguous()

    slot_outputs = torch.empty(
        (num_slots, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    block_n = 64 if w2.shape[1] >= 64 else triton.next_power_of_2(w2.shape[1])
    block_k = 32 if intermediate_size >= 32 else triton.next_power_of_2(intermediate_size)
    _expert_gemm[
        (triton.cdiv(num_slots, block_m), triton.cdiv(w2.shape[1], block_n))
    ](
        activated,
        flat_local_ids,
        w2,
        slot_outputs,
        num_slots,
        w2.shape[1],
        intermediate_size,
        num_local_experts,
        activated.stride(0),
        activated.stride(1),
        flat_local_ids.stride(0),
        w2.stride(0),
        w2.stride(1),
        w2.stride(2),
        slot_outputs.stride(0),
        slot_outputs.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )

    slot_outputs = slot_outputs.view(num_tokens, top_k, w2.shape[1])
    combined_output = (
        slot_outputs.float() * topk_weights.unsqueeze(-1).float()
    ).sum(dim=1)
    return combined_output.to(hidden_states.dtype)


def fused_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    local_expert_start: int = 0,
    hidden_act: str = "silu",
) -> torch.Tensor:
    return _launch_fused_moe_kernels(
        hidden_states=hidden_states,
        w13=w13,
        w2=w2,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        local_expert_start=local_expert_start,
        hidden_act=hidden_act,
    )
