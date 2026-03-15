"""
FP8 KV Cache Unified Attention Triton Kernel

Extended version of chunked_prefill_attn_unified that supports FP8 quantized KV cache.
- Stage 1: Attention against cached FP8 KV (dequantized to BF16 on-the-fly)
- Stage 2: Attention against new BF16 KV (unchanged from original)

This kernel maintains the same interface as the original unified kernel,
only adding k_scale and v_scale parameters for FP8 dequantization.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _chunked_prefill_attn_unified_fp8_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    k_cache_ptr,          # fp8 cache
    v_cache_ptr,          # fp8 cache
    k_scale_ptr,          # fp32 scalar ptr
    v_scale_ptr,          # fp32 scalar ptr
    page_tables_ptr,
    status_table_ptr,
    context_lens_ptr,
    cu_seqlens_q_ptr,
    valid_slices_ptr,
    prefix_lens_ptr,
    padded_prefix_lens_ptr,
    softmax_scale,  # fp32 scalar
    # q/k/v/o strides
    q_stride_s,
    q_stride_h,
    q_stride_d,
    kv_stride_s,
    kv_stride_h,
    kv_stride_d,
    o_stride_s,
    o_stride_h,
    o_stride_d,
    # cache strides: [npages, psz, kvh, d]
    k_cache_stride_npages,
    k_cache_stride_psz,
    k_cache_stride_h,
    k_cache_stride_d,
    v_cache_stride_npages,
    v_cache_stride_psz,
    v_cache_stride_h,
    v_cache_stride_d,
    # page_tables strides
    page_tables_stride_nreqs,
    page_tables_stride_pages,
    # misc
    NUM_GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DLLM_BLOCK_SIZE: tl.constexpr,
    IS_BLOCK_CAUSAL: tl.constexpr,
    IS_PREFIX_FULL: tl.constexpr,
):
    """
    Unified attention kernel with FP8 KV cache support.
    
    Stage 1: Load FP8 K/V from cache, dequantize to BF16, compute attention
    Stage 2: Load BF16 K/V from current step, compute attention
    """
    req_id = tl.program_id(0)
    head_id = tl.program_id(1)
    q_block_id = tl.program_id(2)

    kv_head_id = head_id // NUM_GROUPS

    # Load metadata
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

    # Setup Q loading
    offs_q_block = q_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM_PADDED)
    mask_q_block = offs_q_block < valid_q_len
    mask_d = offs_d < HEAD_DIM

    offs_q = (
        (q_start + offs_q_block[:, None]) * q_stride_s
        + head_id * q_stride_h
        + offs_d[None, :] * q_stride_d
    )
    q = tl.load(
        q_ptr + offs_q,
        mask=mask_q_block[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.bfloat16)

    # Load FP8 dequantization scales (global per-tensor)
    k_scale = tl.load(k_scale_ptr).to(tl.float32)
    v_scale = tl.load(v_scale_ptr).to(tl.float32)

    # Flash attention accumulators
    m = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM_PADDED], dtype=tl.float32)

    # ---------------------------------------------------------
    # Stage 1: Attention against FP8 KV Cache (dequant to BF16)
    # ---------------------------------------------------------
    offs_kv_cache_block = tl.arange(0, BLOCK_N)
    mask_kv_cache_block = offs_kv_cache_block < PAGE_SIZE
    num_pages = (context_len + PAGE_SIZE - 1) // PAGE_SIZE

    for page_rel_id in range(0, num_pages):
        page_abs_id = tl.load(
            page_tables_ptr + req_id * page_tables_stride_nreqs + page_rel_id * page_tables_stride_pages
        ).to(tl.int32)

        page_token_ids = offs_kv_cache_block + page_rel_id * PAGE_SIZE
        page_token_valid_map = (page_abs_id >= 0) & (page_token_ids < context_len) & mask_kv_cache_block

        # Load K from FP8 cache
        k_offs = (
            page_abs_id * k_cache_stride_npages
            + offs_kv_cache_block[:, None] * k_cache_stride_psz
            + kv_head_id * k_cache_stride_h
            + offs_d[None, :] * k_cache_stride_d
        )

        # Load FP8, dequantize to BF16: BF16 = FP8 * scale
        # NOTE: Use FP32 intermediate to avoid cvt.bf16.f16 (requires sm_90+)
        k_fp8 = tl.load(
            k_cache_ptr + k_offs,
            mask=page_token_valid_map[:, None] & mask_d[None, :],
            other=0.0,
        )
        k = (k_fp8.to(tl.float32) * k_scale).to(tl.bfloat16)

        # Compute attention scores
        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * softmax_scale
        scores = tl.where(mask_q_block[:, None] & page_token_valid_map[None, :], scores, float("-inf"))

        # Online softmax update
        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        l_new = l * tl.exp(m - m_new) + tl.sum(p, axis=1)
        alpha = tl.exp(m - m_new)
        acc *= alpha[:, None]

        # Load V from FP8 cache
        v_offs = (
            page_abs_id * v_cache_stride_npages
            + offs_kv_cache_block[:, None] * v_cache_stride_psz
            + kv_head_id * v_cache_stride_h
            + offs_d[None, :] * v_cache_stride_d
        )

        # Load FP8, dequantize to BF16
        v_fp8 = tl.load(
            v_cache_ptr + v_offs,
            mask=page_token_valid_map[:, None] & mask_d[None, :],
            other=0.0,
        )
        v = (v_fp8.to(tl.float32) * v_scale).to(tl.bfloat16)

        # Accumulate attention output
        acc += tl.dot(p.to(tl.bfloat16), v).to(tl.float32)
        m = m_new
        l = l_new

    # ---------------------------------------------------------
    # Stage 2: Attention against new KV (BF16, unchanged)
    # ---------------------------------------------------------
    kv_start = q_start
    full_range = tl.cdiv(valid_kv_len, BLOCK_N)
    block_causal_range = tl.minimum(
        tl.cdiv(valid_q_len + (q_block_id + 1) * BLOCK_M, BLOCK_N),
        tl.cdiv(valid_kv_len, BLOCK_N),
    )

    if IS_BLOCK_CAUSAL and not IS_PREFIX_FULL:
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
        offs_kv_block = kv_block_start + tl.arange(0, BLOCK_N)
        kv_token_valid_map = (offs_kv_block < new_len) & (offs_kv_block < valid_q_len)

        # Load K (BF16, from current step)
        k_offs = (
            (kv_start + offs_kv_block[None, :]) * kv_stride_s
            + kv_head_id * kv_stride_h
            + offs_d[:, None] * kv_stride_d
        )
        k = tl.load(
            k_ptr + k_offs,
            mask=kv_token_valid_map[None, :] & mask_d[:, None],
            other=0.0,
        ).to(tl.bfloat16)

        # Compute attention scores
        scores = tl.dot(q, k).to(tl.float32) * softmax_scale
        score_valid_mask = mask_q_block[:, None] & kv_token_valid_map[None, :]

        # Apply causal mask
        if IS_BLOCK_CAUSAL and not IS_PREFIX_FULL:
            score_block_mask = ((offs_q_block // DLLM_BLOCK_SIZE + 1) * DLLM_BLOCK_SIZE)[:, None] > offs_kv_block[None, :]
            score_mask = score_valid_mask & score_block_mask
        elif IS_BLOCK_CAUSAL and IS_PREFIX_FULL:
            if is_prefilling:
                score_pure_prefix_mask = (offs_q_block < prefix_len)[:, None] & (offs_kv_block < prefix_len)[None, :]
                score_padded_causal_mask = (
                    ((offs_q_block >= prefix_len) & (offs_q_block < padded_prefix_len))[:, None]
                    & (offs_kv_block < padded_prefix_len)[None, :]
                )
                score_block_mask = ((offs_q_block // DLLM_BLOCK_SIZE + 1) * DLLM_BLOCK_SIZE)[:, None] > offs_kv_block[None, :]
                score_block_mask_extend_only = score_block_mask & (offs_q_block >= padded_prefix_len)[:, None]
                score_mask = score_pure_prefix_mask | score_padded_causal_mask | score_block_mask_extend_only
            else:
                score_block_mask = ((offs_q_block // DLLM_BLOCK_SIZE + 1) * DLLM_BLOCK_SIZE)[:, None] > offs_kv_block[None, :]
                score_mask = score_valid_mask & score_block_mask
        else:
            score_mask = score_valid_mask

        scores = tl.where(score_mask, scores, float("-inf"))

        # Online softmax update
        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        l_new = l * tl.exp(m - m_new) + tl.sum(p, axis=1)
        alpha = tl.exp(m - m_new)
        acc *= alpha[:, None]

        # Load V (BF16, from current step)
        v_offs = (
            (kv_start + offs_kv_block[:, None]) * kv_stride_s
            + kv_head_id * kv_stride_h
            + offs_d[None, :] * kv_stride_d
        )
        v = tl.load(
            v_ptr + v_offs,
            mask=kv_token_valid_map[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        acc += tl.dot(p.to(tl.bfloat16), v).to(tl.float32)
        m = m_new
        l = l_new

    # Normalize and store output
    out = acc / l[:, None]
    o_offs = (
        (q_start + offs_q_block[:, None]) * o_stride_s
        + head_id * o_stride_h
        + offs_d[None, :] * o_stride_d
    )
    tl.store(
        o_ptr + o_offs,
        out.to(tl.bfloat16),
        mask=mask_q_block[:, None] & mask_d[None, :],
    )


def chunked_prefill_attn_unified_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,   # fp8
    v_cache: torch.Tensor,   # fp8
    k_scale: torch.Tensor,   # scalar fp32 tensor
    v_scale: torch.Tensor,   # scalar fp32 tensor
    attn_metadata,
):
    """
    FP8 KV Cache Unified Attention Forward.
    
    Args:
        q: Query tensor [total_seqlen, num_heads, head_dim] (BF16)
        k: Key tensor [total_seqlen, num_kv_heads, head_dim] (BF16) - current step
        v: Value tensor [total_seqlen, num_kv_heads, head_dim] (BF16) - current step
        k_cache: Key cache [num_pages, page_size, num_kv_heads, head_dim] (FP8)
        v_cache: Value cache [num_pages, page_size, num_kv_heads, head_dim] (FP8)
        k_scale: Per-tensor K scale (scalar float32)
        v_scale: Per-tensor V scale (scalar float32)
        attn_metadata: Attention metadata object
        
    Returns:
        Output tensor [total_seqlen, num_heads, head_dim] (BF16)
    """
    o = torch.empty_like(q)
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    num_groups = num_heads // num_kv_heads

    head_dim = q.shape[-1]
    head_dim_padded = 1 << (head_dim - 1).bit_length()
    softmax_scale = 1.0 / (head_dim ** 0.5)
    page_size = k_cache.shape[1]
    num_reqs = attn_metadata.cu_seqlens_q.shape[0] - 1

    # Block sizes (tuned for RTX 4090)
    BLOCK_M = 64
    BLOCK_N = 32  # Reduced for shared memory

    grid = (num_reqs, num_heads, triton.cdiv(int(attn_metadata.max_seqlen_q), BLOCK_M))

    _chunked_prefill_attn_unified_fp8_kernel[grid](
        q,
        k,
        v,
        o,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
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
        *o.stride(),
        *k_cache.stride(),
        *v_cache.stride(),
        *attn_metadata.page_tables.stride(),
        NUM_GROUPS=num_groups,
        HEAD_DIM=head_dim,
        HEAD_DIM_PADDED=head_dim_padded,
        PAGE_SIZE=page_size,
        DLLM_BLOCK_SIZE=attn_metadata.block_size,
        IS_BLOCK_CAUSAL=attn_metadata.is_block_causal,
        IS_PREFIX_FULL=attn_metadata.is_prefix_full,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return o
