# Triton JIT kernels: Pyright may falsely report "unreachable" for code inside @triton.jit
# decorated functions (the body is traced/compiled, not run as normal Python).
# pyright: reportUnreachable=false

import triton
import triton.language as tl


@triton.jit
def _chunked_prefill_attn_unified_bf16_cache_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    k_cache_ptr, v_cache_ptr,
    page_tables_ptr,
    context_lens_ptr, cu_seqlens_q_ptr, valid_slices_ptr,
    softmax_scale,  # fp32 scalar
    # q/k/v/o strides
    q_stride_s, q_stride_h, q_stride_d,
    kv_stride_s, kv_stride_h, kv_stride_d,
    o_stride_s, o_stride_h, o_stride_d,
    # cache strides: [npages, psz, kvh, d]
    k_cache_stride_npages, k_cache_stride_psz, k_cache_stride_h, k_cache_stride_d,
    v_cache_stride_npages, v_cache_stride_psz, v_cache_stride_h, v_cache_stride_d,
    # page_tables strides
    page_tables_stride_nreqs, page_tables_stride_pages,
    # misc
    NUM_GROUPS: tl.constexpr, HEAD_DIM: tl.constexpr, HEAD_DIM_PADDED: tl.constexpr, 
    PAGE_SIZE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    req_id = tl.program_id(0)
    head_id = tl.program_id(1)
    q_block_id = tl.program_id(2)

    kv_head_id = head_id // NUM_GROUPS

    q_start = tl.load(cu_seqlens_q_ptr + req_id).to(tl.int32)
    q_end = tl.load(cu_seqlens_q_ptr + req_id + 1).to(tl.int32)
    valid_slice = tl.load(valid_slices_ptr + req_id).to(tl.int32)
    q_len = q_end - q_start
    valid_q_len = valid_slice - q_start
    new_len = q_len  # decode path: current-step KV length matches query length
    context_len = tl.load(context_lens_ptr + req_id).to(tl.int32)
    
    offs_q_block = q_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM_PADDED)
    mask_q_block = offs_q_block < valid_q_len
    mask_d = offs_d < HEAD_DIM

    offs_q = (q_start + offs_q_block[:, None]) * q_stride_s + head_id * q_stride_h + offs_d[None, :] * q_stride_d
    q = tl.load(q_ptr + offs_q, mask=mask_q_block[:, None] & mask_d[None, :], other=0.0).to(tl.bfloat16)

    m = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM_PADDED], dtype=tl.float32)

    # Cache stage: iterate only needed blocks (dynamic loop, like vLLM kernels).
    offs_n_cache = tl.arange(0, BLOCK_N)
    tok_off_cache = offs_n_cache
    mask_n_cache = offs_n_cache < PAGE_SIZE

    num_cache_blocks = (context_len + PAGE_SIZE - 1) // PAGE_SIZE
    for blk in range(0, num_cache_blocks):
        page = tl.load(page_tables_ptr + req_id * page_tables_stride_nreqs + blk * page_tables_stride_pages).to(tl.int32)
        tok_base = blk * PAGE_SIZE
        tok_idx = tok_base + tok_off_cache
        valid_tok = (page >= 0) & (tok_idx < context_len) & mask_n_cache

        k_offs = (
            page * k_cache_stride_npages
            + tok_off_cache[:, None] * k_cache_stride_psz
            + kv_head_id * k_cache_stride_h
            + offs_d[None, :] * k_cache_stride_d
        )
        k_blk = tl.load(
            k_cache_ptr + k_offs,
            mask=valid_tok[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        scores = tl.dot(q, tl.trans(k_blk)).to(tl.float32) * softmax_scale
        scores = tl.where(mask_q_block[:, None] & valid_tok[None, :], scores, float("-inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        l_new = l * tl.exp(m - m_new) + tl.sum(p, axis=1)
        alpha = tl.exp(m - m_new)
        acc *= alpha[:, None]

        v_offs = (
            page * v_cache_stride_npages
            + tok_off_cache[:, None] * v_cache_stride_psz
            + kv_head_id * v_cache_stride_h
            + offs_d[None, :] * v_cache_stride_d
        )
        v_blk = tl.load(
            v_cache_ptr + v_offs,
            mask=valid_tok[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        acc += tl.dot(p.to(tl.bfloat16), v_blk).to(tl.float32)
        m = m_new
        l = l_new

    # New KV stage (dynamic tiles)
    kv_start = q_start
    for start_n in range(0, new_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid_tok = offs_n < new_len

        k_offs = (kv_start + offs_n[None, :]) * kv_stride_s + kv_head_id * kv_stride_h + offs_d[:, None] * kv_stride_d
        k_blk = tl.load(
            k_ptr + k_offs,
            mask=valid_tok[None, :] & mask_d[:, None],
            other=0.0,
        ).to(tl.bfloat16)

        scores = tl.dot(q, k_blk).to(tl.float32) * softmax_scale
        scores = tl.where(mask_q_block[:, None] & valid_tok[None, :], scores, float("-inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        l_new = l * tl.exp(m - m_new) + tl.sum(p, axis=1)
        alpha = tl.exp(m - m_new)
        acc *= alpha[:, None]

        v_offs = (kv_start + offs_n[:, None]) * kv_stride_s + kv_head_id * kv_stride_h + offs_d[None, :] * kv_stride_d
        v_blk = tl.load(
            v_ptr + v_offs,
            mask=valid_tok[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        acc += tl.dot(p.to(tl.bfloat16), v_blk).to(tl.float32)
        m = m_new
        l = l_new

    out = acc / l[:, None]
    o_offs = (q_start + offs_q_block[:, None]) * o_stride_s + head_id * o_stride_h + offs_d[None, :] * o_stride_d
    tl.store(o_ptr + o_offs, out.to(tl.bfloat16), mask=mask_q_block[:, None] & mask_d[None, :])