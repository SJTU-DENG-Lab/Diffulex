import torch
import triton
import triton.language as tl

from diffulex.attention.metadata import AttnMetaDataBase
from diffulex_kernel.python.auto_tuner import build_chunked_prefill_configs


@triton.autotune(
    configs=[
        triton.Config(c, num_warps=c.pop("num_warps"), num_stages=c.pop("num_stages"))
        for c in build_chunked_prefill_configs()
    ],
    key=["NUM_GROUPS", "HEAD_DIM", "IS_BLOCK_CAUSAL", "IS_PREFIX_FULL"],
)
@triton.jit
def _chunked_prefill_attn_unified_kernel(
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
    req_id = tl.program_id(0)
    head_id = tl.program_id(1)
    q_block_id = tl.program_id(2)

    kv_head_id = head_id // NUM_GROUPS

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

    offs_q_block = q_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM_PADDED)
    mask_q_block = offs_q_block < valid_q_len
    mask_d = offs_d < HEAD_DIM

    offs_q = (q_start + offs_q_block[:, None]) * q_stride_s + head_id * q_stride_h + offs_d[None, :] * q_stride_d
    q = tl.load(q_ptr + offs_q, mask=mask_q_block[:, None] & mask_d[None, :], other=0.0).to(tl.bfloat16)

    m = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM_PADDED], dtype=tl.float32)

    # Stage 1: Attention against KV Cache
    offs_kv_cache_block = tl.arange(0, BLOCK_N)
    mask_kv_cache_block = offs_kv_cache_block < PAGE_SIZE
    num_pages = (context_len + PAGE_SIZE - 1) // PAGE_SIZE
    for page_rel_id in range(0, num_pages):
        page_abs_id = tl.load(
            page_tables_ptr + req_id * page_tables_stride_nreqs + page_rel_id * page_tables_stride_pages
        ).to(tl.int32)
        page_token_ids = offs_kv_cache_block + page_rel_id * PAGE_SIZE
        page_token_valid_map = (page_abs_id >= 0) & (page_token_ids < context_len) & mask_kv_cache_block

        k_offs = (
            page_abs_id * k_cache_stride_npages
            + offs_kv_cache_block[:, None] * k_cache_stride_psz
            + kv_head_id * k_cache_stride_h
            + offs_d[None, :] * k_cache_stride_d
        )
        k = tl.load(
            k_cache_ptr + k_offs,
            mask=page_token_valid_map[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * softmax_scale
        scores = tl.where(mask_q_block[:, None] & page_token_valid_map[None, :], scores, float("-inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        l_new = l * tl.exp(m - m_new) + tl.sum(p, axis=1)
        alpha = tl.exp(m - m_new)
        acc *= alpha[:, None]

        v_offs = (
            page_abs_id * v_cache_stride_npages
            + offs_kv_cache_block[:, None] * v_cache_stride_psz
            + kv_head_id * v_cache_stride_h
            + offs_d[None, :] * v_cache_stride_d
        )
        v = tl.load(
            v_cache_ptr + v_offs,
            mask=page_token_valid_map[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        acc += tl.dot(p.to(tl.bfloat16), v).to(tl.float32)
        m = m_new
        l = l_new

    # Stage 2: Attention against new KV
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

        k_offs = (
            (kv_start + offs_kv_block[None, :]) * kv_stride_s + kv_head_id * kv_stride_h + offs_d[:, None] * kv_stride_d
        )
        k = tl.load(
            k_ptr + k_offs,
            mask=kv_token_valid_map[None, :] & mask_d[:, None],
            other=0.0,
        ).to(tl.bfloat16)

        scores = tl.dot(q, k).to(tl.float32) * softmax_scale
        score_valid_mask = mask_q_block[:, None] & kv_token_valid_map[None, :]
        if IS_BLOCK_CAUSAL and not IS_PREFIX_FULL:
            score_block_mask = ((offs_q_block // DLLM_BLOCK_SIZE + 1) * DLLM_BLOCK_SIZE)[:, None] > offs_kv_block[
                None, :
            ]
            score_mask = score_valid_mask & score_block_mask
        elif IS_BLOCK_CAUSAL and IS_PREFIX_FULL:
            if is_prefilling:
                score_pure_prefix_mask = (offs_q_block < prefix_len)[:, None] & (offs_kv_block < prefix_len)[None, :]
                score_padded_causal_mask = ((offs_q_block >= prefix_len) & (offs_q_block < padded_prefix_len))[
                    :, None
                ] & (offs_kv_block < padded_prefix_len)[None, :]
                score_block_mask = ((offs_q_block // DLLM_BLOCK_SIZE + 1) * DLLM_BLOCK_SIZE)[:, None] > offs_kv_block[
                    None, :
                ]
                score_block_mask_extend_only = score_block_mask & (offs_q_block >= padded_prefix_len)[:, None]
                score_mask = score_pure_prefix_mask | score_padded_causal_mask | score_block_mask_extend_only
            else:
                score_block_mask = ((offs_q_block // DLLM_BLOCK_SIZE + 1) * DLLM_BLOCK_SIZE)[:, None] > offs_kv_block[
                    None, :
                ]
                score_mask = score_valid_mask & score_block_mask
        else:
            score_mask = score_valid_mask
        scores = tl.where(score_mask, scores, float("-inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        l_new = l * tl.exp(m - m_new) + tl.sum(p, axis=1)
        alpha = tl.exp(m - m_new)
        acc *= alpha[:, None]

        v_offs = (
            (kv_start + offs_kv_block[:, None]) * kv_stride_s + kv_head_id * kv_stride_h + offs_d[None, :] * kv_stride_d
        )
        v = tl.load(
            v_ptr + v_offs,
            mask=kv_token_valid_map[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        acc += tl.dot(p.to(tl.bfloat16), v).to(tl.float32)
        m = m_new
        l = l_new

    out = acc / l[:, None]
    o_offs = (q_start + offs_q_block[:, None]) * o_stride_s + head_id * o_stride_h + offs_d[None, :] * o_stride_d
    tl.store(
        o_ptr + o_offs,
        out.to(tl.bfloat16),
        mask=mask_q_block[:, None] & mask_d[None, :],
    )


def chunked_prefill_attn_unified(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    attn_metadata: AttnMetaDataBase,
):
    o = torch.empty_like(q).to(q.device)
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    num_groups = num_heads // num_kv_heads

    head_dim = q.shape[-1]
    head_dim_padded = 1 << (head_dim - 1).bit_length()
    softmax_scale = 1.0 / head_dim**0.5
    page_size = k_cache.shape[1]
    num_reqs = attn_metadata.cu_seqlens_q.shape[0] - 1

    grid = lambda meta: (num_reqs, num_heads, triton.cdiv(int(attn_metadata.max_seqlen_q), meta["BLOCK_M"]))
    _chunked_prefill_attn_unified_kernel[grid](
        q,
        k,
        v,
        o,
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
    )
    return o
