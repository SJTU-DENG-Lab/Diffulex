import torch
import triton
import triton.language as tl

import os

from diffulex.attention.metadata import AttnMetaDataBase


@triton.jit
def _paged_decode_attn_unified_bf16_cache_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    cu_seqlens_q_ptr,
    o_ptr,
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
    # cache strides: [nblks, page, kvh, d]
    k_cache_stride_nblks,
    k_cache_stride_page,
    k_cache_stride_h,
    k_cache_stride_d,
    v_cache_stride_nblks,
    v_cache_stride_page,
    v_cache_stride_h,
    v_cache_stride_d,
    # block_tables strides
    block_tables_stride_s,
    block_tables_stride_b,
    # misc
    NUM_GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    kv_head = pid_head // NUM_GROUPS

    q_start = tl.load(cu_seqlens_q_ptr + pid_seq).to(tl.int32)
    q_end = tl.load(cu_seqlens_q_ptr + pid_seq + 1).to(tl.int32)
    q_len = q_end - q_start
    new_len = q_len  # decode path: current-step KV length matches query length
    context_len = tl.load(context_lens_ptr + pid_seq).to(tl.int32)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM_PADDED)
    mask_m = offs_m < q_len
    mask_d = offs_d < HEAD_DIM

    q_offs = (q_start + offs_m[:, None]) * q_stride_s + pid_head * q_stride_h + offs_d[None, :] * q_stride_d
    q = tl.load(q_ptr + q_offs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.bfloat16)

    m = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM_PADDED], dtype=tl.float32)

    # Cache stage: iterate only needed blocks (dynamic loop, like vLLM kernels).
    offs_n_cache = tl.arange(0, BLOCK_N)
    tok_off_cache = offs_n_cache
    mask_n_cache = offs_n_cache < PAGE_SIZE

    num_cache_blocks = (context_len + PAGE_SIZE - 1) // PAGE_SIZE
    for blk in range(0, num_cache_blocks):
        page = tl.load(block_tables_ptr + pid_seq * block_tables_stride_s + blk * block_tables_stride_b).to(tl.int32)
        tok_base = blk * PAGE_SIZE
        tok_idx = tok_base + tok_off_cache
        valid_tok = (page >= 0) & (tok_idx < context_len) & mask_n_cache

        k_offs = (
            page * k_cache_stride_nblks
            + tok_off_cache[:, None] * k_cache_stride_page
            + kv_head * k_cache_stride_h
            + offs_d[None, :] * k_cache_stride_d
        )
        k_blk = tl.load(
            k_cache_ptr + k_offs,
            mask=valid_tok[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        scores = tl.dot(q, tl.trans(k_blk)).to(tl.float32) * softmax_scale
        scores = tl.where(mask_m[:, None] & valid_tok[None, :], scores, float("-inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        l_new = l * tl.exp(m - m_new) + tl.sum(p, axis=1)
        alpha = tl.exp(m - m_new)
        acc *= alpha[:, None]

        v_offs = (
            page * v_cache_stride_nblks
            + tok_off_cache[:, None] * v_cache_stride_page
            + kv_head * v_cache_stride_h
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

        k_offs = (kv_start + offs_n[None, :]) * kv_stride_s + kv_head * kv_stride_h + offs_d[:, None] * kv_stride_d
        k_blk = tl.load(
            k_ptr + k_offs,
            mask=valid_tok[None, :] & mask_d[:, None],
            other=0.0,
        ).to(tl.bfloat16)

        scores = tl.dot(q, k_blk).to(tl.float32) * softmax_scale
        scores = tl.where(mask_m[:, None] & valid_tok[None, :], scores, float("-inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        l_new = l * tl.exp(m - m_new) + tl.sum(p, axis=1)
        alpha = tl.exp(m - m_new)
        acc *= alpha[:, None]

        v_offs = (kv_start + offs_n[:, None]) * kv_stride_s + kv_head * kv_stride_h + offs_d[None, :] * kv_stride_d
        v_blk = tl.load(
            v_ptr + v_offs,
            mask=valid_tok[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        acc += tl.dot(p.to(tl.bfloat16), v_blk).to(tl.float32)
        m = m_new
        l = l_new

    out = acc / l[:, None]
    o_offs = (q_start + offs_m[:, None]) * o_stride_s + pid_head * o_stride_h + offs_d[None, :] * o_stride_d
    tl.store(o_ptr + o_offs, out.to(tl.bfloat16), mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def _paged_decode_attn_unified_fp8_cache_kernel_legacy(
    q_ptr,
    k_ptr,
    v_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_tables_ptr,
    context_lens_ptr,
    cu_seqlens_q_ptr,
    o_ptr,
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
    # cache strides: [nblks, page, kvh, d]
    k_cache_stride_nblks,
    k_cache_stride_page,
    k_cache_stride_h,
    k_cache_stride_d,
    v_cache_stride_nblks,
    v_cache_stride_page,
    v_cache_stride_h,
    v_cache_stride_d,
    # block_tables strides
    block_tables_stride_s,
    block_tables_stride_b,
    # misc
    NUM_GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    kv_head = pid_head // NUM_GROUPS
    k_scale = tl.load(k_scale_ptr + kv_head).to(tl.float32)
    v_scale = tl.load(v_scale_ptr + kv_head).to(tl.float32)

    q_start = tl.load(cu_seqlens_q_ptr + pid_seq).to(tl.int32)
    q_end = tl.load(cu_seqlens_q_ptr + pid_seq + 1).to(tl.int32)
    q_len = q_end - q_start
    new_len = q_len
    context_len = tl.load(context_lens_ptr + pid_seq).to(tl.int32)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM_PADDED)
    mask_m = offs_m < q_len
    mask_d = offs_d < HEAD_DIM

    q_offs = (q_start + offs_m[:, None]) * q_stride_s + pid_head * q_stride_h + offs_d[None, :] * q_stride_d
    q = tl.load(q_ptr + q_offs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.bfloat16)

    m = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM_PADDED], dtype=tl.float32)

    offs_n_cache = tl.arange(0, BLOCK_N)
    tok_off_cache = offs_n_cache
    mask_n_cache = offs_n_cache < PAGE_SIZE

    num_cache_blocks = (context_len + PAGE_SIZE - 1) // PAGE_SIZE
    for blk in range(0, num_cache_blocks):
        page = tl.load(block_tables_ptr + pid_seq * block_tables_stride_s + blk * block_tables_stride_b).to(tl.int32)
        tok_base = blk * PAGE_SIZE
        tok_idx = tok_base + tok_off_cache
        valid_tok = (page >= 0) & (tok_idx < context_len) & mask_n_cache

        k_offs = (
            page * k_cache_stride_nblks
            + tok_off_cache[:, None] * k_cache_stride_page
            + kv_head * k_cache_stride_h
            + offs_d[None, :] * k_cache_stride_d
        )
        # fp8 cache values: dot(Q, K_fp8) * k_scale == dot(Q, (K_fp8*k_scale))
        k_blk = tl.load(
            k_cache_ptr + k_offs,
            mask=valid_tok[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        scores = tl.dot(q, tl.trans(k_blk)).to(tl.float32) * (softmax_scale * k_scale)
        scores = tl.where(mask_m[:, None] & valid_tok[None, :], scores, float("-inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        l_new = l * tl.exp(m - m_new) + tl.sum(p, axis=1)
        alpha = tl.exp(m - m_new)
        acc *= alpha[:, None]

        v_offs = (
            page * v_cache_stride_nblks
            + tok_off_cache[:, None] * v_cache_stride_page
            + kv_head * v_cache_stride_h
            + offs_d[None, :] * v_cache_stride_d
        )
        v_blk = tl.load(
            v_cache_ptr + v_offs,
            mask=valid_tok[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        # Apply v_scale on weights (cheaper than scaling V elementwise).
        acc += tl.dot((p * v_scale).to(tl.bfloat16), v_blk).to(tl.float32)
        m = m_new
        l = l_new

    kv_start = q_start
    for start_n in range(0, new_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid_tok = offs_n < new_len

        k_offs = (kv_start + offs_n[None, :]) * kv_stride_s + kv_head * kv_stride_h + offs_d[:, None] * kv_stride_d
        k_blk = tl.load(
            k_ptr + k_offs,
            mask=valid_tok[None, :] & mask_d[:, None],
            other=0.0,
        ).to(tl.bfloat16)

        scores = tl.dot(q, k_blk).to(tl.float32) * softmax_scale
        scores = tl.where(mask_m[:, None] & valid_tok[None, :], scores, float("-inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        l_new = l * tl.exp(m - m_new) + tl.sum(p, axis=1)
        alpha = tl.exp(m - m_new)
        acc *= alpha[:, None]

        v_offs = (kv_start + offs_n[:, None]) * kv_stride_s + kv_head * kv_stride_h + offs_d[None, :] * kv_stride_d
        v_blk = tl.load(
            v_ptr + v_offs,
            mask=valid_tok[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        acc += tl.dot(p.to(tl.bfloat16), v_blk).to(tl.float32)
        m = m_new
        l = l_new

    out = acc / l[:, None]
    o_offs = (q_start + offs_m[:, None]) * o_stride_s + pid_head * o_stride_h + offs_d[None, :] * o_stride_d
    tl.store(o_ptr + o_offs, out.to(tl.bfloat16), mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def _paged_decode_attn_unified_fp8_cache_fused_dot_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_tables_ptr,
    context_lens_ptr,
    cu_seqlens_q_ptr,
    o_ptr,
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
    # cache strides: [nblks, page, kvh, d]
    k_cache_stride_nblks,
    k_cache_stride_page,
    k_cache_stride_h,
    k_cache_stride_d,
    v_cache_stride_nblks,
    v_cache_stride_page,
    v_cache_stride_h,
    v_cache_stride_d,
    # block_tables strides
    block_tables_stride_s,
    block_tables_stride_b,
    # misc
    KV_FORMAT: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    FP8-cache decode kernel with *fused* fp8 math:
    - Keep KV cache tiles in float8 (via fp8 view tensor)
    - Use tl.dot_scaled(..., rhs_format="e4m3/e5m2") to consume fp8 without explicit dequant tensors
    - Apply per-head scalar scales (k_scale/v_scale) without elementwise dequantization
    """
    pid_seq = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    kv_head = pid_head // NUM_GROUPS
    k_scale = tl.load(k_scale_ptr + kv_head).to(tl.float32)
    v_scale = tl.load(v_scale_ptr + kv_head).to(tl.float32)

    q_start = tl.load(cu_seqlens_q_ptr + pid_seq).to(tl.int32)
    q_end = tl.load(cu_seqlens_q_ptr + pid_seq + 1).to(tl.int32)
    q_len = q_end - q_start
    new_len = q_len
    context_len = tl.load(context_lens_ptr + pid_seq).to(tl.int32)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM_PADDED)
    mask_m = offs_m < q_len
    mask_d = offs_d < HEAD_DIM

    # Load Q (bf16). Note: triton 3.5 `tl.dot` does not support mixed bf16/fp16 x fp8.
    # We use `tl.dot_scaled` (microscaling) to accept fp8 operands.
    q_offs = (q_start + offs_m[:, None]) * q_stride_s + pid_head * q_stride_h + offs_d[None, :] * q_stride_d
    q = tl.load(q_ptr + q_offs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.bfloat16)

    m = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM_PADDED], dtype=tl.float32)

    offs_n_cache = tl.arange(0, BLOCK_N)
    tok_off_cache = offs_n_cache
    mask_n_cache = offs_n_cache < PAGE_SIZE

    num_cache_blocks = (context_len + PAGE_SIZE - 1) // PAGE_SIZE
    for blk in range(0, num_cache_blocks):
        page = tl.load(block_tables_ptr + pid_seq * block_tables_stride_s + blk * block_tables_stride_b).to(tl.int32)
        tok_base = blk * PAGE_SIZE
        tok_idx = tok_base + tok_off_cache
        valid_tok = (page >= 0) & (tok_idx < context_len) & mask_n_cache

        # K cache: keep fp8 element type; load as [K, N] to match dot_scaled rhs layout.
        k_offs = (
            page * k_cache_stride_nblks
            + tok_off_cache[None, :] * k_cache_stride_page
            + kv_head * k_cache_stride_h
            + offs_d[:, None] * k_cache_stride_d
        )
        k_blk = tl.load(
            k_cache_ptr + k_offs,
            mask=mask_d[:, None] & valid_tok[None, :],
            other=0.0,
        )

        # scores = QK^T * softmax_scale, with scalar k_scale applied after dot:
        # dot(Q, K_true) == dot(Q, K_fp8) * k_scale (per-head scalar scale).
        scores = tl.dot_scaled(
            q,
            None,
            "bf16",
            k_blk,
            None,
            KV_FORMAT,
        ) * (softmax_scale * k_scale)
        scores = tl.where(mask_m[:, None] & valid_tok[None, :], scores, float("-inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        l_new = l * tl.exp(m - m_new) + tl.sum(p, axis=1)
        alpha = tl.exp(m - m_new)
        acc *= alpha[:, None]

        # V cache: keep fp8 element type for tl.dot.
        v_offs = (
            page * v_cache_stride_nblks
            + tok_off_cache[:, None] * v_cache_stride_page
            + kv_head * v_cache_stride_h
            + offs_d[None, :] * v_cache_stride_d
        )
        v_blk = tl.load(
            v_cache_ptr + v_offs,
            mask=valid_tok[:, None] & mask_d[None, :],
            other=0.0,
        )

        # acc += P @ V_true == (P @ V_fp8) * v_scale
        acc += tl.dot_scaled(
            p.to(tl.float16),
            None,
            "fp16",
            v_blk,
            None,
            KV_FORMAT,
        ) * v_scale
        m = m_new
        l = l_new

    # New KV stage (bf16 tensors, unchanged)
    kv_start = q_start
    for start_n in range(0, new_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid_tok = offs_n < new_len

        k_offs = (kv_start + offs_n[None, :]) * kv_stride_s + kv_head * kv_stride_h + offs_d[:, None] * kv_stride_d
        k_blk = tl.load(
            k_ptr + k_offs,
            mask=valid_tok[None, :] & mask_d[:, None],
            other=0.0,
        ).to(tl.bfloat16)

        scores = tl.dot(q, k_blk, out_dtype=tl.float32) * softmax_scale
        scores = tl.where(mask_m[:, None] & valid_tok[None, :], scores, float("-inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        l_new = l * tl.exp(m - m_new) + tl.sum(p, axis=1)
        alpha = tl.exp(m - m_new)
        acc *= alpha[:, None]

        v_offs = (kv_start + offs_n[:, None]) * kv_stride_s + kv_head * kv_stride_h + offs_d[None, :] * kv_stride_d
        v_blk = tl.load(
            v_ptr + v_offs,
            mask=valid_tok[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        acc += tl.dot(p.to(tl.bfloat16), v_blk, out_dtype=tl.float32)
        m = m_new
        l = l_new

    out = acc / l[:, None]
    o_offs = (q_start + offs_m[:, None]) * o_stride_s + pid_head * o_stride_h + offs_d[None, :] * o_stride_d
    tl.store(o_ptr + o_offs, out.to(tl.bfloat16), mask=mask_m[:, None] & mask_d[None, :])


def paged_attn_decode_unified_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    attn_metadata: AttnMetaDataBase,
    *,
    softmax_scale: float,
    fp8_cache: bool,
) -> torch.Tensor:
    """
    Triton paged-attention decode for unified KV cache layout.

    q: [total_q, num_heads, head_dim] (bf16)
    k/v: [total_q, num_kv_heads, head_dim] (bf16), aligned with cu_seqlens_q
    k_cache/v_cache:
      - bf16: [num_page_blocks, page_size, num_kv_heads, head_dim]
      - fp8 : same shape but dtype must be float8 view for triton (strategy.view_kv_cache_for_kernels)
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda and k_cache.is_cuda and v_cache.is_cuda
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
    assert attn_metadata.block_tables is not None and attn_metadata.context_lens is not None and attn_metadata.cu_seqlens_q is not None
    assert attn_metadata.kv_cache_layout == "unified", f"only unified layout supported, got {attn_metadata.kv_cache_layout}"

    # Be robust to different metadata implementations (dataclass vs SimpleNamespace in tests).
    num_seqs = int(attn_metadata.cu_seqlens_q.numel() - 1)
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    num_kv_heads = k.shape[1]
    assert num_heads % num_kv_heads == 0
    num_groups = num_heads // num_kv_heads

    page_size = int(attn_metadata.page_block_size)

    # Heuristics: BLOCK_M = 64 (supports diffusion_block_size=32/64), BLOCK_N = page_size/new-tile
    BLOCK_M = 64
    BLOCK_N = 32 if page_size <= 32 else 64
    # Cache stage requires BLOCK_N == PAGE_SIZE to simplify; enforce.
    if BLOCK_N != page_size:
        BLOCK_N = page_size

    head_dim_padded = 1 << (head_dim - 1).bit_length()

    o = torch.empty_like(q)
    grid = (num_seqs, num_heads, triton.cdiv(int(attn_metadata.max_seqlen_q), BLOCK_M))

    if fp8_cache:
        if attn_metadata.k_scale is None or attn_metadata.v_scale is None:
            raise ValueError("fp8_cache=True requires attn_metadata.k_scale/v_scale")
        # Default to fused fp8-dot kernel; fallback to legacy on compile/runtime failures.
        # Set DIFFULEX_PAGED_DECODE_FP8_FUSED_DOT=0 to force legacy.
        # Set DIFFULEX_PAGED_DECODE_FP8_FUSED_DOT_STRICT=1 to raise instead of fallback.
        use_fused_dot = os.getenv("DIFFULEX_PAGED_DECODE_FP8_FUSED_DOT", "1") != "0"
        strict_fused = os.getenv("DIFFULEX_PAGED_DECODE_FP8_FUSED_DOT_STRICT", "0") == "1"
        if use_fused_dot:
            # `tl.dot_scaled` needs the fp8 format string to interpret raw bytes correctly.
            # Derive from the fp8 view dtype (torch.float8_*).
            dt = str(k_cache.dtype)
            if "e4m3" in dt:
                kv_format = "e4m3"
            elif "e5m2" in dt:
                kv_format = "e5m2"
            else:
                raise ValueError(f"Unsupported fp8 k_cache dtype for fused-dot: {k_cache.dtype}")
            try:
                _paged_decode_attn_unified_fp8_cache_fused_dot_kernel[grid](
                    q, k, v,
                    k_cache, v_cache,
                    attn_metadata.k_scale, attn_metadata.v_scale,
                    attn_metadata.block_tables,
                    attn_metadata.context_lens,
                    attn_metadata.cu_seqlens_q,
                    o,
                    softmax_scale,
                    *q.stride(), *k.stride(), *o.stride(),
                    *k_cache.stride(), *v_cache.stride(),
                    *attn_metadata.block_tables.stride(),
                    KV_FORMAT=kv_format,
                    NUM_GROUPS=num_groups,
                    HEAD_DIM=head_dim,
                    HEAD_DIM_PADDED=head_dim_padded,
                    PAGE_SIZE=page_size,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    num_warps=4,
                    num_stages=2,
                )
            except Exception:
                if strict_fused:
                    raise
                _paged_decode_attn_unified_fp8_cache_kernel_legacy[grid](
                    q, k, v,
                    k_cache, v_cache,
                    attn_metadata.k_scale, attn_metadata.v_scale,
                    attn_metadata.block_tables,
                    attn_metadata.context_lens,
                    attn_metadata.cu_seqlens_q,
                    o,
                    softmax_scale,
                    *q.stride(), *k.stride(), *o.stride(),
                    *k_cache.stride(), *v_cache.stride(),
                    *attn_metadata.block_tables.stride(),
                    NUM_GROUPS=num_groups,
                    HEAD_DIM=head_dim,
                    HEAD_DIM_PADDED=head_dim_padded,
                    PAGE_SIZE=page_size,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    num_warps=4,
                    num_stages=2,
                )
        else:
            _paged_decode_attn_unified_fp8_cache_kernel_legacy[grid](
                q, k, v,
                k_cache, v_cache,
                attn_metadata.k_scale, attn_metadata.v_scale,
                attn_metadata.block_tables,
                attn_metadata.context_lens,
                attn_metadata.cu_seqlens_q,
                o,
                softmax_scale,
                *q.stride(), *k.stride(), *o.stride(),
                *k_cache.stride(), *v_cache.stride(),
                *attn_metadata.block_tables.stride(),
                NUM_GROUPS=num_groups,
                HEAD_DIM=head_dim,
                HEAD_DIM_PADDED=head_dim_padded,
                PAGE_SIZE=page_size,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                num_warps=4,
                num_stages=2,
            )
    else:
        _paged_decode_attn_unified_bf16_cache_kernel[grid](
            q, k, v,
            k_cache, v_cache,
            attn_metadata.block_tables,
            attn_metadata.context_lens,
            attn_metadata.cu_seqlens_q,
            o,
            softmax_scale,
            *q.stride(), *k.stride(), *o.stride(),
            *k_cache.stride(), *v_cache.stride(),
            *attn_metadata.block_tables.stride(),
            NUM_GROUPS=num_groups,
            HEAD_DIM=head_dim,
            HEAD_DIM_PADDED=head_dim_padded,
            PAGE_SIZE=page_size,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=2,
        )

    return o

