from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _greedy_confidence_stage1(
    logits_ptr,
    local_ids_ptr,
    local_max_ptr,
    local_sum_ptr,
    logits_stride_m,
    vocab_size: tl.constexpr,
    vocab_limit: tl.constexpr,
    forbidden_token_id: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < vocab_size
    if vocab_limit >= 0:
        mask = mask & (offsets < vocab_limit)
    if forbidden_token_id >= 0:
        mask = mask & (offsets != forbidden_token_id)

    logits = tl.load(
        logits_ptr + row * logits_stride_m + offsets,
        mask=offsets < vocab_size,
        other=-float("inf"),
    ).to(tl.float32)
    logits = tl.where(mask, logits, -float("inf"))

    block_max = tl.max(logits, axis=0)
    tie_ids = tl.where((logits == block_max) & mask, offsets, -1)
    block_token = tl.max(tie_ids, axis=0)

    exp_values = tl.exp(logits - block_max)
    exp_values = tl.where(mask, exp_values, 0.0)
    block_sum = tl.sum(exp_values, axis=0)

    out_offset = row * tl.num_programs(1) + block_idx
    tl.store(local_ids_ptr + out_offset, block_token)
    tl.store(local_max_ptr + out_offset, block_max)
    tl.store(local_sum_ptr + out_offset, block_sum)


@triton.jit
def _greedy_confidence_stage2(
    local_ids_ptr,
    local_max_ptr,
    local_sum_ptr,
    confidence_ptr,
    sampled_tokens_ptr,
    num_blocks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks
    base = row * num_blocks + offsets

    local_max = tl.load(local_max_ptr + base, mask=mask, other=-float("inf"))
    local_ids = tl.load(local_ids_ptr + base, mask=mask, other=-1)
    local_sum = tl.load(local_sum_ptr + base, mask=mask, other=0.0)

    global_max = tl.max(local_max, axis=0)
    candidate_ids = tl.where((local_max == global_max) & mask, local_ids, -1)
    sampled_token = tl.max(candidate_ids, axis=0)
    denom = tl.sum(local_sum * tl.exp(local_max - global_max), axis=0)
    confidence = 1.0 / denom

    tl.store(sampled_tokens_ptr + row, sampled_token)
    tl.store(confidence_ptr + row, confidence)


def greedy_confidence(
    logits: torch.Tensor,
    *,
    vocab_limit: int | None = None,
    forbidden_token_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedy top-1 plus exact softmax confidence without materializing probs."""
    if not logits.is_cuda:
        raise ValueError("greedy_confidence requires CUDA logits.")
    if logits.dim() != 2:
        raise ValueError(f"greedy_confidence expects [num_rows, vocab], got {tuple(logits.shape)}.")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Unsupported logits dtype for greedy_confidence: {logits.dtype}.")

    num_rows, vocab_size = logits.shape
    if num_rows == 0:
        empty_i64 = torch.empty((0,), dtype=torch.int64, device=logits.device)
        empty_f32 = torch.empty((0,), dtype=torch.float32, device=logits.device)
        return empty_f32, empty_i64, empty_f32

    effective_vocab_limit = vocab_size if vocab_limit is None else int(vocab_limit)
    if effective_vocab_limit < 0 or effective_vocab_limit > vocab_size:
        effective_vocab_limit = vocab_size
    effective_forbidden_token_id = -1 if forbidden_token_id is None else int(forbidden_token_id)
    if effective_forbidden_token_id < 0 or effective_forbidden_token_id >= vocab_size:
        effective_forbidden_token_id = -1

    block_size = 1024
    num_blocks = triton.cdiv(vocab_size, block_size)
    local_ids = torch.empty((num_rows, num_blocks), dtype=torch.int64, device=logits.device)
    local_max = torch.empty((num_rows, num_blocks), dtype=torch.float32, device=logits.device)
    local_sum = torch.empty((num_rows, num_blocks), dtype=torch.float32, device=logits.device)
    confidence = torch.empty((num_rows,), dtype=logits.dtype, device=logits.device)
    sampled_tokens = torch.empty((num_rows,), dtype=torch.int64, device=logits.device)

    _greedy_confidence_stage1[(num_rows, num_blocks)](
        logits,
        local_ids,
        local_max,
        local_sum,
        logits.stride(0),
        vocab_size,
        effective_vocab_limit,
        effective_forbidden_token_id,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )

    reduce_block_size = triton.next_power_of_2(num_blocks)
    _greedy_confidence_stage2[(num_rows,)](
        local_ids,
        local_max,
        local_sum,
        confidence,
        sampled_tokens,
        num_blocks,
        BLOCK_SIZE=reduce_block_size,
        num_warps=1,
    )
    initial_confidence = confidence.clone()
    return confidence, sampled_tokens, initial_confidence
