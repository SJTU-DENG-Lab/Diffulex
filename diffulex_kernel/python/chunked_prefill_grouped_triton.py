"""Experimental grouped paged attention kernel for chunked prefill.

This file is intentionally separate from chunked_prefill_triton.py while the
optimized launch decomposition is being developed.

Goal:
- Preserve the current Diffulex metadata semantics.
- Change the CTA ownership from `(request, q_head, q_token_block)` to
  `(request, kv_head, q_token_block)`.
- Flatten `(query token, q_head_within_kv_group)` into the row dimension, so a
  single K/V tile load serves all Q heads mapped to the same KV head.

The first implementation is a conservative correctness-oriented kernel:
- unified KV cache layout only: [num_pages, page_size, kv_heads, head_dim]
- bf16/fp16 style dense K/V cache, no quantization
- same two-stage model as the current Diffulex kernel:
  1. attend to existing paged cache
  2. attend to new K/V from the current input tensors
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from diffulex.attention.metadata import AttnMetaDataBase


@triton.jit
def _chunked_prefill_grouped_attn_unified_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_tables_ptr,
    status_table_ptr,
    context_lens_ptr,
    cu_seqlens_q_ptr,
    valid_slices_ptr,
    prefix_lens_ptr,
    padded_prefix_lens_ptr,
    softmax_scale,
    q_stride_s,
    q_stride_h,
    q_stride_d,
    kv_stride_s,
    kv_stride_h,
    kv_stride_d,
    o_stride_s,
    o_stride_h,
    o_stride_d,
    k_cache_stride_npages,
    k_cache_stride_psz,
    k_cache_stride_h,
    k_cache_stride_d,
    v_cache_stride_npages,
    v_cache_stride_psz,
    v_cache_stride_h,
    v_cache_stride_d,
    page_tables_stride_nreqs,
    page_tables_stride_pages,
    NUM_GROUPS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DLLM_BLOCK_SIZE: tl.constexpr,
    IS_BLOCK_CAUSAL: tl.constexpr,
    IS_PREFIX_FULL: tl.constexpr,
    MASK_PREFIX_HOLE: tl.constexpr,
    PREFIX_CAUSAL: tl.constexpr,
):
    req_id = tl.program_id(0)
    kv_head_id = tl.program_id(1)
    q_block_id = tl.program_id(2)

    status = tl.load(status_table_ptr + req_id).to(tl.int32)
    context_len = tl.load(context_lens_ptr + req_id).to(tl.int32)
    q_start = tl.load(cu_seqlens_q_ptr + req_id).to(tl.int32)
    q_end = tl.load(cu_seqlens_q_ptr + req_id + 1).to(tl.int32)
    valid_slice = tl.load(valid_slices_ptr + req_id).to(tl.int32)
    prefix_len = tl.load(prefix_lens_ptr + req_id).to(tl.int32)
    padded_prefix_len = tl.load(padded_prefix_lens_ptr + req_id).to(tl.int32)

    q_len = q_end - q_start
    valid_q_len = valid_slice - q_start
    valid_kv_len = valid_q_len
    new_len = q_len

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM_PADDED)
    q_rel = q_block_id * BLOCK_Q + offs_m // NUM_GROUPS
    q_group = offs_m % NUM_GROUPS
    q_head_id = kv_head_id * NUM_GROUPS + q_group

    mask_q = (q_rel < valid_q_len) & (q_head_id < NUM_Q_HEADS)
    mask_d = offs_d < HEAD_DIM
    abs_q = context_len + q_rel

    q_offs = (
        (q_start + q_rel[:, None]) * q_stride_s
        + q_head_id[:, None] * q_stride_h
        + offs_d[None, :] * q_stride_d
    )
    q = tl.load(q_ptr + q_offs, mask=mask_q[:, None] & mask_d[None, :], other=0.0).to(tl.bfloat16)

    m = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM_PADDED], dtype=tl.float32)

    full_cache_range = tl.cdiv(context_len, BLOCK_N)
    for cache_block_id in range(0, full_cache_range):
        offs_kv_cache = cache_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_cache_valid = offs_kv_cache < context_len
        page_rel_ids = offs_kv_cache // PAGE_SIZE
        page_abs_ids = tl.load(
            page_tables_ptr + req_id * page_tables_stride_nreqs + page_rel_ids * page_tables_stride_pages,
            mask=kv_cache_valid,
            other=-1,
        ).to(tl.int32)
        page_offs = offs_kv_cache % PAGE_SIZE
        page_token_valid = kv_cache_valid & (page_abs_ids >= 0)
        if MASK_PREFIX_HOLE:
            prefix_hole = (offs_kv_cache >= prefix_len) & (offs_kv_cache < padded_prefix_len)
            page_token_valid = page_token_valid & ~prefix_hole

        k_offs = (
            page_abs_ids[:, None] * k_cache_stride_npages
            + page_offs[:, None] * k_cache_stride_psz
            + kv_head_id * k_cache_stride_h
            + offs_d[None, :] * k_cache_stride_d
        )
        k = tl.load(k_cache_ptr + k_offs, mask=page_token_valid[:, None] & mask_d[None, :], other=0.0).to(
            tl.bfloat16
        )

        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * softmax_scale
        scores = tl.where(mask_q[:, None] & page_token_valid[None, :], scores, float("-inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        l_new = l * tl.exp(m - m_new) + tl.sum(p, axis=1)
        acc *= tl.exp(m - m_new)[:, None]

        v_offs = (
            page_abs_ids[:, None] * v_cache_stride_npages
            + page_offs[:, None] * v_cache_stride_psz
            + kv_head_id * v_cache_stride_h
            + offs_d[None, :] * v_cache_stride_d
        )
        v = tl.load(v_cache_ptr + v_offs, mask=page_token_valid[:, None] & mask_d[None, :], other=0.0).to(
            tl.bfloat16
        )

        acc += tl.dot(p.to(tl.bfloat16), v).to(tl.float32)
        m = m_new
        l = l_new

    kv_start = q_start
    full_range = tl.cdiv(valid_kv_len, BLOCK_N)
    max_q_idx_in_chunk = tl.minimum(valid_q_len - 1, (q_block_id + 1) * BLOCK_Q - 1)
    max_abs_q_idx = context_len + max_q_idx_in_chunk
    max_abs_kv_idx = ((max_abs_q_idx // DLLM_BLOCK_SIZE) + 1) * DLLM_BLOCK_SIZE
    max_rel_kv_len = max_abs_kv_idx - context_len
    block_causal_range = tl.minimum(tl.maximum(0, tl.cdiv(max_rel_kv_len, BLOCK_N)), full_range)

    if PREFIX_CAUSAL and status == 0:
        loop_range = full_range
    elif IS_BLOCK_CAUSAL and not IS_PREFIX_FULL:
        loop_range = block_causal_range
    elif IS_BLOCK_CAUSAL and IS_PREFIX_FULL:
        is_prefilling = status == 0
        if is_prefilling:
            loop_range = full_range
        else:
            loop_range = block_causal_range
    else:
        loop_range = full_range

    for kv_block_id in range(0, loop_range):
        kv_block_start = kv_block_id * BLOCK_N
        offs_kv = kv_block_start + tl.arange(0, BLOCK_N)
        abs_kv = context_len + offs_kv
        kv_token_valid = (offs_kv < new_len) & (offs_kv < valid_q_len)

        k_offs = (kv_start + offs_kv[None, :]) * kv_stride_s + kv_head_id * kv_stride_h + offs_d[:, None] * kv_stride_d
        k = tl.load(k_ptr + k_offs, mask=kv_token_valid[None, :] & mask_d[:, None], other=0.0).to(tl.bfloat16)

        scores = tl.dot(q, k).to(tl.float32) * softmax_scale
        score_valid = mask_q[:, None] & kv_token_valid[None, :]
        if PREFIX_CAUSAL and status == 0:
            score_mask = score_valid & (abs_kv[None, :] <= abs_q[:, None])
        elif IS_BLOCK_CAUSAL and not IS_PREFIX_FULL:
            block_mask = ((abs_q // DLLM_BLOCK_SIZE + 1) * DLLM_BLOCK_SIZE)[:, None] > abs_kv[None, :]
            score_mask = score_valid & block_mask
        elif IS_BLOCK_CAUSAL and IS_PREFIX_FULL:
            if is_prefilling:
                pure_prefix = (q_rel < prefix_len)[:, None] & (offs_kv < prefix_len)[None, :]
                padded_causal = ((q_rel >= prefix_len) & (q_rel < padded_prefix_len))[:, None] & (
                    offs_kv < padded_prefix_len
                )[None, :]
                block_mask = ((abs_q // DLLM_BLOCK_SIZE + 1) * DLLM_BLOCK_SIZE)[:, None] > abs_kv[None, :]
                block_extend = block_mask & (q_rel >= padded_prefix_len)[:, None]
                score_mask = score_valid & (pure_prefix | padded_causal | block_extend)
            else:
                block_mask = ((abs_q // DLLM_BLOCK_SIZE + 1) * DLLM_BLOCK_SIZE)[:, None] > abs_kv[None, :]
                score_mask = score_valid & block_mask
        else:
            score_mask = score_valid
        scores = tl.where(score_mask, scores, float("-inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        l_new = l * tl.exp(m - m_new) + tl.sum(p, axis=1)
        acc *= tl.exp(m - m_new)[:, None]

        v_offs = (
            (kv_start + offs_kv[:, None]) * kv_stride_s
            + kv_head_id * kv_stride_h
            + offs_d[None, :] * kv_stride_d
        )
        v = tl.load(v_ptr + v_offs, mask=kv_token_valid[:, None] & mask_d[None, :], other=0.0).to(tl.bfloat16)

        acc += tl.dot(p.to(tl.bfloat16), v).to(tl.float32)
        m = m_new
        l = l_new

    out = acc / l[:, None]
    o_offs = (
        (q_start + q_rel[:, None]) * o_stride_s
        + q_head_id[:, None] * o_stride_h
        + offs_d[None, :] * o_stride_d
    )
    tl.store(o_ptr + o_offs, out.to(tl.bfloat16), mask=mask_q[:, None] & mask_d[None, :])


def chunked_prefill_attn_grouped_unified(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    attn_metadata: AttnMetaDataBase,
    softmax_scale: float | None = None,
    *,
    block_q: int | None = None,
    block_n: int = 16,
) -> torch.Tensor:
    """Run the experimental grouped chunked prefill kernel.

    The output should match chunked_prefill_attn_unified for supported cases.
    Keep this entrypoint explicit while the kernel is experimental.
    """
    out = torch.empty_like(q)
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    num_groups = num_heads // num_kv_heads
    head_dim = q.shape[-1]
    head_dim_padded = 1 << (head_dim - 1).bit_length()
    softmax_scale = float(softmax_scale if softmax_scale is not None else 1.0 / head_dim**0.5)
    num_reqs = attn_metadata.cu_seqlens_q.shape[0] - 1
    # On D=512 shapes, BLOCK_M=16 is conservative but underutilizes each CTA.
    # BLOCK_M=32 fits on RTX 3090 with BLOCK_N=16 and is much closer to the
    # useful work per launch we want for grouped GQA.
    block_q = int(block_q if block_q is not None else max(1, 32 // num_groups))
    block_m = block_q * num_groups

    grid = (num_reqs, num_kv_heads, triton.cdiv(int(attn_metadata.max_seqlen_q), block_q))
    _chunked_prefill_grouped_attn_unified_kernel[grid](
        q,
        k,
        v,
        out,
        k_cache,
        v_cache,
        attn_metadata.page_tables,
        attn_metadata.status_table,
        attn_metadata.context_lens,
        attn_metadata.cu_seqlens_q,
        attn_metadata.valid_slices,
        attn_metadata.prefix_lens,
        attn_metadata.padded_prefix_lens,
        softmax_scale,
        *q.stride(),
        *k.stride(),
        *out.stride(),
        *k_cache.stride(),
        *v_cache.stride(),
        *attn_metadata.page_tables.stride(),
        NUM_GROUPS=num_groups,
        NUM_Q_HEADS=num_heads,
        HEAD_DIM=head_dim,
        HEAD_DIM_PADDED=head_dim_padded,
        PAGE_SIZE=k_cache.shape[1],
        BLOCK_Q=block_q,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        DLLM_BLOCK_SIZE=attn_metadata.block_size,
        IS_BLOCK_CAUSAL=attn_metadata.is_block_causal,
        IS_PREFIX_FULL=attn_metadata.is_prefix_full,
        MASK_PREFIX_HOLE=bool(getattr(attn_metadata, "mask_prefix_hole", False)),
        PREFIX_CAUSAL=bool(getattr(attn_metadata, "prefix_causal", False)),
    )
    return out
