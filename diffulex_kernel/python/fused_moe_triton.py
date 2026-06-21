from __future__ import annotations

"""SGLang-inspired fused MoE draft for Diffulex.

This version keeps Diffulex's current high-level dispatcher contract
(`hidden_states`, `topk_ids`, `topk_weights`) but performs a local
expert-major pack/sort step before launching Triton kernels.
"""

from math import prod
import os

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from diffulex.moe.metadata import ExpertExecutionMetadata


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


PACKED_BLOCK_M = _env_int("DIFFULEX_MOE_GEMM_BLOCK_M", 16)
PACKED_BLOCK_N = _env_int("DIFFULEX_MOE_GEMM_BLOCK_N", 64)
PACKED_BLOCK_K = _env_int("DIFFULEX_MOE_GEMM_BLOCK_K", 32)
PACKED_BLOCK_K_W13 = _env_int("DIFFULEX_MOE_GEMM_BLOCK_K_W13", PACKED_BLOCK_K)
PACKED_BLOCK_K_W2 = _env_int("DIFFULEX_MOE_GEMM_BLOCK_K_W2", PACKED_BLOCK_K)
PACKED_NUM_WARPS = _env_int("DIFFULEX_MOE_GEMM_NUM_WARPS", 4)

ALIGN_COUNT_BLOCK_M = _env_int("DIFFULEX_MOE_ALIGN_COUNT_BLOCK_M", 256)
ALIGN_COUNT_NUM_WARPS = _env_int("DIFFULEX_MOE_ALIGN_COUNT_NUM_WARPS", 8)
ALIGN_BLOCK_E = _env_int("DIFFULEX_MOE_ALIGN_BLOCK_E", 16)
ALIGN_FILL_NUM_WARPS = _env_int("DIFFULEX_MOE_ALIGN_FILL_NUM_WARPS", 1)

WORKSPACE_CACHE: dict[tuple[str, int, torch.dtype], torch.Tensor] = {}


def _workspace_device_index(device: torch.device) -> int:
    return device.index if device.index is not None else -1


def _get_workspace_tensor(
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    zero: bool = False,
    fill_value: int | float | None = None,
) -> torch.Tensor:
    numel = prod(shape)
    key = (name, _workspace_device_index(device), dtype)
    buffer = WORKSPACE_CACHE.get(key)
    if buffer is None or buffer.numel() < numel:
        buffer = torch.empty(numel, device=device, dtype=dtype)
        WORKSPACE_CACHE[key] = buffer
    out = buffer[:numel].view(*shape)
    if zero:
        out.zero_()
    elif fill_value is not None:
        out.fill_(fill_value)
    return out


def _validate_fused_moe_inputs(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    hidden_act: str,
) -> None:
    if hidden_act not in {"silu", "gelu", "gelu_tanh", "gelu_pytorch_tanh"}:
        raise ValueError(f"Unsupported fused_moe expert activation: {hidden_act!r}.")
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
def _grouped_expert_gemm_gathered(
    a_ptr,
    token_ids_ptr,
    expert_ids_ptr,
    w_ptr,
    out_ptr,
    num_rows,
    num_cols,
    k_dim,
    stride_am,
    stride_ak,
    stride_t,
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

    token_ids = tl.load(
        token_ids_ptr + offs_m * stride_t,
        mask=row_mask,
        other=-1,
    ).to(tl.int32)
    expert_ids = tl.load(
        expert_ids_ptr + offs_m * stride_e,
        mask=row_mask,
        other=0,
    ).to(tl.int32)
    valid_rows = row_mask & (token_ids >= 0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)

    for k_start in range(0, k_dim, BLOCK_K):
        current_k = k_start + k_offsets
        k_mask = current_k < k_dim
        a = tl.load(
            a_ptr + token_ids[:, None] * stride_am + current_k[None, :] * stride_ak,
            mask=valid_rows[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            w_ptr
            + expert_ids[:, None, None] * stride_we
            + offs_n[None, :, None] * stride_wn
            + current_k[None, None, :] * stride_wk,
            mask=valid_rows[:, None, None] & (offs_n[None, :, None] < num_cols) & k_mask[None, None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(a[:, None, :] * b, axis=2)

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=row_mask[:, None] & (offs_n[None, :] < num_cols),
    )


@triton.jit
def _grouped_expert_gemm_packed(
    a_ptr,
    expert_ids_ptr,
    w_ptr,
    out_ptr,
    num_rows,
    num_cols,
    k_dim,
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
        other=0,
    ).to(tl.int32)

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
            + expert_ids[:, None, None] * stride_we
            + offs_n[None, :, None] * stride_wn
            + current_k[None, None, :] * stride_wk,
            mask=row_mask[:, None, None] & (offs_n[None, :, None] < num_cols) & k_mask[None, None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(a[:, None, :] * b, axis=2)

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=row_mask[:, None] & (offs_n[None, :] < num_cols),
    )


@triton.jit
def _grouped_expert_gemm_packed_aligned(
    a_ptr,
    sorted_slot_ids_ptr,
    expert_block_ids_ptr,
    w_ptr,
    out_ptr,
    num_slots,
    num_padded_slots,
    num_cols,
    k_dim,
    num_local_experts,
    stride_am,
    stride_ak,
    stride_s,
    stride_b,
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
    row_mask = offs_m < num_padded_slots

    slot_ids = tl.load(
        sorted_slot_ids_ptr + offs_m * stride_s,
        mask=row_mask,
        other=num_slots,
    ).to(tl.int32)
    expert_id = tl.load(
        expert_block_ids_ptr + pid_m * stride_b,
        mask=pid_m >= 0,
        other=-1,
    ).to(tl.int32)
    valid_rows = row_mask & (slot_ids >= 0) & (slot_ids < num_slots) & (expert_id >= 0) & (expert_id < num_local_experts)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)

    for k_start in range(0, k_dim, BLOCK_K):
        current_k = k_start + k_offsets
        k_mask = current_k < k_dim
        a = tl.load(
            a_ptr + slot_ids[:, None] * stride_am + current_k[None, :] * stride_ak,
            mask=valid_rows[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            w_ptr
            + expert_id * stride_we
            + offs_n[:, None] * stride_wn
            + current_k[None, :] * stride_wk,
            mask=(expert_id >= 0) & (expert_id < num_local_experts) & (offs_n[:, None] < num_cols) & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(a, tl.trans(b))

    tl.store(
        out_ptr + slot_ids[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=valid_rows[:, None] & (offs_n[None, :] < num_cols),
    )


@triton.jit
def _weighted_scatter_add(
    slot_outputs_ptr,
    token_ids_ptr,
    weights_ptr,
    out_ptr,
    num_rows,
    num_cols,
    stride_sm,
    stride_sn,
    stride_t,
    stride_w,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = offs_m < num_rows
    col_mask = offs_n < num_cols

    token_ids = tl.load(
        token_ids_ptr + offs_m * stride_t,
        mask=row_mask,
        other=-1,
    ).to(tl.int64)
    valid_rows = row_mask & (token_ids >= 0)
    weights = tl.load(
        weights_ptr + offs_m * stride_w,
        mask=valid_rows,
        other=0.0,
    ).to(tl.float32)
    values = tl.load(
        slot_outputs_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn,
        mask=valid_rows[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    weighted = values * weights[:, None]

    out_ptrs = out_ptr + token_ids[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.atomic_add(
        out_ptrs,
        weighted,
        mask=valid_rows[:, None] & col_mask[None, :],
    )


@triton.jit
def _count_slots_per_expert(
    local_expert_ids_ptr,
    counts_ptr,
    num_slots,
    num_local_experts: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offsets < num_slots
    expert_ids = tl.load(local_expert_ids_ptr + offsets, mask=mask, other=-1).to(tl.int32)
    valid = mask & (expert_ids >= 0) & (expert_ids < num_local_experts)
    tl.atomic_add(counts_ptr + expert_ids, 1, sem="relaxed", mask=valid)


@triton.jit
def _scatter_slots_by_expert(
    local_expert_ids_ptr,
    offsets_ptr,
    cursors_ptr,
    sorted_slot_ids_ptr,
    num_slots,
    num_local_experts: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    slot_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = slot_offsets < num_slots
    expert_ids = tl.load(local_expert_ids_ptr + slot_offsets, mask=mask, other=-1).to(tl.int32)
    valid = mask & (expert_ids >= 0) & (expert_ids < num_local_experts)
    expert_offsets = tl.load(offsets_ptr + expert_ids, mask=valid, other=0).to(tl.int32)
    local_offsets = tl.atomic_add(cursors_ptr + expert_ids, 1, sem="relaxed", mask=valid).to(tl.int32)
    dst_offsets = expert_offsets + local_offsets
    tl.store(sorted_slot_ids_ptr + dst_offsets, slot_offsets, mask=valid)


@triton.jit
def _build_expert_offsets(
    counts_ptr,
    offsets_ptr,
    padded_counts_ptr,
    num_local_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    running_offset = tl.full((), 0, dtype=tl.int32)
    for expert_id in range(0, num_local_experts):
        count = tl.load(counts_ptr + expert_id).to(tl.int32)
        padded_count = ((count + block_size - 1) // block_size) * block_size
        tl.store(offsets_ptr + expert_id, running_offset)
        tl.store(padded_counts_ptr + expert_id, padded_count)
        running_offset += padded_count


@triton.jit
def _fill_expert_block_ids(
    offsets_ptr,
    padded_counts_ptr,
    expert_block_ids_ptr,
    num_local_experts: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    expert_offsets = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    expert_mask = expert_offsets < num_local_experts
    start_offsets = tl.load(offsets_ptr + expert_offsets, mask=expert_mask, other=0).to(tl.int32)
    padded_counts = tl.load(padded_counts_ptr + expert_offsets, mask=expert_mask, other=0).to(tl.int32)
    start_blocks = start_offsets // block_size
    num_blocks = padded_counts // block_size

    for block_offset in range(0, max_num_blocks):
        valid = expert_mask & (block_offset < num_blocks)
        tl.store(expert_block_ids_ptr + start_blocks + block_offset, expert_offsets, mask=valid)


def _pack_fused_moe_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_local_experts: int,
    local_expert_start: int,
    block_m: int,
) -> ExpertExecutionMetadata:
    num_tokens, hidden_size = hidden_states.shape
    del hidden_size
    top_k = topk_ids.shape[1]
    num_slots = num_tokens * top_k

    if num_slots == 0:
        empty_i32 = torch.empty((0,), device=hidden_states.device, dtype=torch.int32)
        empty_w = torch.empty((0,), device=hidden_states.device, dtype=topk_weights.dtype)
        return ExpertExecutionMetadata(
            packed_token_ids=empty_i32,
            packed_local_expert_ids=empty_i32,
            packed_weights=empty_w,
            num_slots=0,
        )

    local_ids = topk_ids.to(torch.int32) - int(local_expert_start)
    local_mask = (local_ids >= 0) & (local_ids < num_local_experts)
    flat_valid_mask = local_mask.reshape(-1)

    flat_local_ids = local_ids.reshape(-1)
    flat_token_ids = (
        torch.arange(num_tokens, device=hidden_states.device, dtype=torch.int32)
        .unsqueeze(1)
        .expand(num_tokens, top_k)
        .reshape(-1)
    )
    flat_weights = topk_weights.reshape(-1)

    packed_token_ids = _get_workspace_tensor(
        "packed_token_ids",
        (num_slots,),
        dtype=torch.int32,
        device=hidden_states.device,
    )
    packed_local_expert_ids = _get_workspace_tensor(
        "packed_local_expert_ids",
        (num_slots,),
        dtype=torch.int32,
        device=hidden_states.device,
    )
    packed_weights = _get_workspace_tensor(
        "packed_weights",
        (num_slots,),
        dtype=topk_weights.dtype,
        device=hidden_states.device,
    )
    packed_token_ids.copy_(torch.where(flat_valid_mask, flat_token_ids, torch.full_like(flat_token_ids, -1)))
    packed_local_expert_ids.copy_(
        torch.where(flat_valid_mask, flat_local_ids, torch.zeros_like(flat_local_ids))
    )
    packed_weights.copy_(
        torch.where(flat_valid_mask, flat_weights, torch.zeros_like(flat_weights))
    )

    return ExpertExecutionMetadata(
        packed_token_ids=packed_token_ids,
        packed_local_expert_ids=packed_local_expert_ids,
        packed_weights=packed_weights,
        num_slots=num_slots,
    )


def _build_aligned_block_metadata(
    packed_local_expert_ids: torch.Tensor,
    *,
    block_size: int,
    num_local_experts: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int | None]:
    if packed_local_expert_ids.numel() == 0:
        return None, None, None
    return _build_aligned_block_metadata_triton(
        packed_local_expert_ids,
        block_size=block_size,
        num_local_experts=num_local_experts,
    )


def _build_aligned_block_metadata_triton(
    packed_local_expert_ids: torch.Tensor,
    *,
    block_size: int,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    num_slots = int(packed_local_expert_ids.numel())
    max_num_tokens_padded = num_slots + (num_local_experts + 1) * (block_size - 1)
    max_num_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    max_blocks_per_expert = triton.cdiv(num_slots, block_size)

    counts = _get_workspace_tensor(
        "aligned_counts",
        (num_local_experts,),
        dtype=torch.int32,
        device=packed_local_expert_ids.device,
        zero=True,
    )
    _count_slots_per_expert[
        (triton.cdiv(num_slots, ALIGN_COUNT_BLOCK_M),)
    ](
        packed_local_expert_ids,
        counts,
        num_slots,
        num_local_experts,
        BLOCK_M=ALIGN_COUNT_BLOCK_M,
        num_warps=ALIGN_COUNT_NUM_WARPS,
    )

    offsets = _get_workspace_tensor(
        "aligned_offsets",
        (num_local_experts,),
        dtype=torch.int32,
        device=packed_local_expert_ids.device,
    )
    padded_counts = _get_workspace_tensor(
        "aligned_padded_counts",
        (num_local_experts,),
        dtype=torch.int32,
        device=packed_local_expert_ids.device,
    )
    _build_expert_offsets[(1,)](
        counts,
        offsets,
        padded_counts,
        num_local_experts,
        block_size,
        num_warps=1,
    )
    cursors = _get_workspace_tensor(
        "aligned_cursors",
        (num_local_experts,),
        dtype=torch.int32,
        device=packed_local_expert_ids.device,
        zero=True,
    )
    sorted_slot_ids = _get_workspace_tensor(
        "aligned_sorted_slot_ids",
        (max_num_tokens_padded,),
        dtype=torch.int32,
        device=packed_local_expert_ids.device,
        fill_value=-1,
    )
    expert_block_ids = _get_workspace_tensor(
        "aligned_expert_block_ids",
        (max_num_blocks,),
        dtype=torch.int32,
        device=packed_local_expert_ids.device,
        fill_value=-1,
    )
    _scatter_slots_by_expert[
        (triton.cdiv(num_slots, ALIGN_COUNT_BLOCK_M),)
    ](
        packed_local_expert_ids,
        offsets,
        cursors,
        sorted_slot_ids,
        num_slots,
        num_local_experts,
        BLOCK_M=ALIGN_COUNT_BLOCK_M,
        num_warps=ALIGN_COUNT_NUM_WARPS,
    )
    _fill_expert_block_ids[
        (triton.cdiv(num_local_experts, ALIGN_BLOCK_E),)
    ](
        offsets,
        padded_counts,
        expert_block_ids,
        num_local_experts,
        block_size,
        max_blocks_per_expert,
        BLOCK_E=ALIGN_BLOCK_E,
        num_warps=ALIGN_FILL_NUM_WARPS,
    )
    return sorted_slot_ids, expert_block_ids, max_num_tokens_padded


def _launch_grouped_expert_gemm_gathered(
    hidden_states: torch.Tensor,
    w: torch.Tensor,
    packed_inputs: ExpertExecutionMetadata,
    *,
    num_cols: int,
    k_dim: int,
    block_k_override: int | None = None,
) -> torch.Tensor:
    packed_out = _get_workspace_tensor(
        "gate_up",
        (packed_inputs.num_slots, num_cols),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    if packed_inputs.num_slots == 0:
        return packed_out

    block_m = PACKED_BLOCK_M
    block_n = min(PACKED_BLOCK_N, triton.next_power_of_2(num_cols))
    base_block_k = PACKED_BLOCK_K if block_k_override is None else int(block_k_override)
    block_k = min(base_block_k, triton.next_power_of_2(k_dim))
    _grouped_expert_gemm_gathered[
        (triton.cdiv(packed_inputs.num_slots, block_m), triton.cdiv(num_cols, block_n))
    ](
        hidden_states,
        packed_inputs.packed_token_ids,
        packed_inputs.packed_local_expert_ids,
        w,
        packed_out,
        packed_inputs.num_slots,
        num_cols,
        k_dim,
        hidden_states.stride(0),
        hidden_states.stride(1),
        packed_inputs.packed_token_ids.stride(0),
        packed_inputs.packed_local_expert_ids.stride(0),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        packed_out.stride(0),
        packed_out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=PACKED_NUM_WARPS,
    )
    return packed_out


def _launch_grouped_expert_gemm_packed(
    packed_hidden_states: torch.Tensor,
    w: torch.Tensor,
    packed_inputs: ExpertExecutionMetadata,
    *,
    num_cols: int,
    k_dim: int,
    workspace_name: str = "packed_slot_outputs",
    block_k_override: int | None = None,
) -> torch.Tensor:
    packed_out = _get_workspace_tensor(
        workspace_name,
        (packed_inputs.num_slots, num_cols),
        dtype=packed_hidden_states.dtype,
        device=packed_hidden_states.device,
    )
    if packed_inputs.num_slots == 0:
        return packed_out

    block_m = PACKED_BLOCK_M
    block_n = min(PACKED_BLOCK_N, triton.next_power_of_2(num_cols))
    base_block_k = PACKED_BLOCK_K if block_k_override is None else int(block_k_override)
    block_k = min(base_block_k, triton.next_power_of_2(k_dim))
    if (
        packed_inputs.sorted_slot_ids is not None
        and packed_inputs.expert_block_ids is not None
        and packed_inputs.num_tokens_post_padded is not None
    ):
        _grouped_expert_gemm_packed_aligned[
            (triton.cdiv(packed_inputs.num_tokens_post_padded, block_m), triton.cdiv(num_cols, block_n))
        ](
            packed_hidden_states,
            packed_inputs.sorted_slot_ids,
            packed_inputs.expert_block_ids,
            w,
            packed_out,
            packed_inputs.num_slots,
            packed_inputs.num_tokens_post_padded,
            num_cols,
            k_dim,
            w.shape[0],
            packed_hidden_states.stride(0),
            packed_hidden_states.stride(1),
            packed_inputs.sorted_slot_ids.stride(0),
            packed_inputs.expert_block_ids.stride(0),
            w.stride(0),
            w.stride(1),
            w.stride(2),
            packed_out.stride(0),
            packed_out.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=PACKED_NUM_WARPS,
        )
        return packed_out

    _grouped_expert_gemm_packed[
        (triton.cdiv(packed_inputs.num_slots, block_m), triton.cdiv(num_cols, block_n))
    ](
        packed_hidden_states,
        packed_inputs.packed_local_expert_ids,
        w,
        packed_out,
        packed_inputs.num_slots,
        num_cols,
        k_dim,
        packed_hidden_states.stride(0),
        packed_hidden_states.stride(1),
        packed_inputs.packed_local_expert_ids.stride(0),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        packed_out.stride(0),
        packed_out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=PACKED_NUM_WARPS,
    )
    return packed_out


def _combine_packed_moe_outputs(
    packed_slot_outputs: torch.Tensor,
    packed_inputs: ExpertExecutionMetadata,
    *,
    num_tokens: int,
) -> torch.Tensor:
    combined_output = _get_workspace_tensor(
        "combined_output",
        (num_tokens, packed_slot_outputs.shape[1]),
        dtype=torch.float32,
        device=packed_slot_outputs.device,
        zero=True,
    )
    if packed_inputs.num_slots == 0:
        return combined_output.to(packed_slot_outputs.dtype)

    block_m = PACKED_BLOCK_M
    block_n = 64 if packed_slot_outputs.shape[1] >= 64 else triton.next_power_of_2(packed_slot_outputs.shape[1])
    _weighted_scatter_add[
        (triton.cdiv(packed_inputs.num_slots, block_m), triton.cdiv(packed_slot_outputs.shape[1], block_n))
    ](
        packed_slot_outputs,
        packed_inputs.packed_token_ids,
        packed_inputs.packed_weights,
        combined_output,
        packed_inputs.num_slots,
        packed_slot_outputs.shape[1],
        packed_slot_outputs.stride(0),
        packed_slot_outputs.stride(1),
        packed_inputs.packed_token_ids.stride(0),
        packed_inputs.packed_weights.stride(0),
        combined_output.stride(0),
        combined_output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return combined_output.to(packed_slot_outputs.dtype)


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
    hidden_states = hidden_states.contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.contiguous()

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
    num_local_experts = w13.shape[0]
    intermediate_twice = w13.shape[1]
    intermediate_size = intermediate_twice // 2

    packed_inputs = _pack_fused_moe_inputs(
        hidden_states,
        topk_ids,
        topk_weights,
        num_local_experts=num_local_experts,
        local_expert_start=local_expert_start,
        block_m=PACKED_BLOCK_M,
    )

    gate_up = _launch_grouped_expert_gemm_gathered(
        hidden_states,
        w13,
        packed_inputs,
        num_cols=intermediate_twice,
        k_dim=hidden_size,
        block_k_override=PACKED_BLOCK_K_W13,
    )

    gate = gate_up[:, :intermediate_size]
    up = gate_up[:, intermediate_size:]
    if hidden_act == "silu":
        F.silu(gate, inplace=True)
        gate.mul_(up)
    else:
        gate = F.gelu(gate, approximate="tanh") * up
    packed_slot_outputs = _launch_grouped_expert_gemm_packed(
        gate,
        w2,
        packed_inputs,
        num_cols=w2.shape[1],
        k_dim=intermediate_size,
        block_k_override=PACKED_BLOCK_K_W2,
    )
    return _combine_packed_moe_outputs(
        packed_slot_outputs,
        packed_inputs,
        num_tokens=num_tokens,
    ).to(hidden_states.dtype)


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


def fused_expert_packed(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    execution_metadata: ExpertExecutionMetadata,
    *,
    hidden_act: str = "silu",
) -> torch.Tensor:
    hidden_states = hidden_states.contiguous()
    if hidden_states.shape[0] == 0:
        return hidden_states.new_empty((0, w2.shape[1]))
    _validate_fused_moe_inputs(
        hidden_states=hidden_states,
        w13=w13,
        w2=w2,
        topk_ids=execution_metadata.packed_local_expert_ids[:, None],
        topk_weights=execution_metadata.packed_weights[:, None],
        hidden_act=hidden_act,
    )
    num_local_experts = w13.shape[0]
    if execution_metadata.disable_aligned_metadata:
        sorted_slot_ids = None
        expert_block_ids = None
        num_tokens_post_padded = None
    else:
        sorted_slot_ids, expert_block_ids, num_tokens_post_padded = _build_aligned_block_metadata(
            execution_metadata.packed_local_expert_ids,
            block_size=PACKED_BLOCK_M,
            num_local_experts=num_local_experts,
        )
    packed_inputs = ExpertExecutionMetadata(
        packed_token_ids=execution_metadata.packed_token_ids,
        packed_local_expert_ids=execution_metadata.packed_local_expert_ids,
        packed_weights=execution_metadata.packed_weights,
        num_slots=execution_metadata.num_slots,
        seg_indptr=execution_metadata.seg_indptr,
        num_recv_tokens_per_expert=execution_metadata.num_recv_tokens_per_expert,
        sorted_slot_ids=sorted_slot_ids,
        expert_block_ids=expert_block_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        disable_aligned_metadata=execution_metadata.disable_aligned_metadata,
    )
    hidden_size = hidden_states.shape[1]
    intermediate_twice = w13.shape[1]
    intermediate_size = intermediate_twice // 2
    gate_up = _launch_grouped_expert_gemm_packed(
        hidden_states,
        w13,
        packed_inputs,
        num_cols=intermediate_twice,
        k_dim=hidden_size,
        workspace_name="packed_gate_up",
        block_k_override=PACKED_BLOCK_K_W13,
    )
    gate = gate_up[:, :intermediate_size]
    up = gate_up[:, intermediate_size:]
    if hidden_act == "silu":
        F.silu(gate, inplace=True)
        gate.mul_(up)
    else:
        gate = F.gelu(gate, approximate="tanh") * up
    return _launch_grouped_expert_gemm_packed(
        gate,
        w2,
        packed_inputs,
        num_cols=w2.shape[1],
        k_dim=intermediate_size,
        workspace_name="packed_slot_outputs",
        block_k_override=PACKED_BLOCK_K_W2,
    ).to(hidden_states.dtype)
