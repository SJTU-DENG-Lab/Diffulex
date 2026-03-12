"""
FP8 KV Cache Triton Kernel

Custom Triton kernel for FP8 quantized KV cache attention.
Dequantizes FP8 K/V on-the-fly during attention computation.
"""

import torch
import triton
import triton.language as tl
from typing import Optional

from ..kernel_registry import KVCacheKernel
from ..kernel_registry import register_kernel as _register


@triton.jit
def _fp8_kv_attention_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    o_ptr,
    page_tables_ptr,
    context_lens_ptr,
    cu_seqlens_q_ptr,
    softmax_scale,
    # strides
    q_stride_s, q_stride_h, q_stride_d,
    k_cache_stride_npages, k_cache_stride_psz, k_cache_stride_h, k_cache_stride_d,
    v_cache_stride_npages, v_cache_stride_psz, v_cache_stride_h, v_cache_stride_d,
    page_tables_stride_nreqs, page_tables_stride_pages,
    # constants
    NUM_GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_FP8_E4M3: tl.constexpr,
):
    """
    FP8 KV Cache Attention Kernel.
    
    Loads FP8 K/V from cache, dequantizes on-the-fly, computes attention.
    """
    req_id = tl.program_id(0)
    head_id = tl.program_id(1)
    q_block_id = tl.program_id(2)
    
    kv_head_id = head_id // NUM_GROUPS
    
    context_len = tl.load(context_lens_ptr + req_id).to(tl.int32)
    q_start = tl.load(cu_seqlens_q_ptr + req_id).to(tl.int32)
    q_end = tl.load(cu_seqlens_q_ptr + req_id + 1).to(tl.int32)
    q_len = q_end - q_start
    
    # Load per-request KV scales (per-tensor scaling)
    k_scale = tl.load(k_scale_ptr + req_id).to(tl.float32)
    v_scale = tl.load(v_scale_ptr + req_id).to(tl.float32)
    
    offs_q_block = q_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM_PADDED)
    mask_q_block = offs_q_block < q_len
    mask_d = offs_d < HEAD_DIM
    
    # Load Q
    offs_q = (q_start + offs_q_block[:, None]) * q_stride_s + head_id * q_stride_h + offs_d[None, :] * q_stride_d
    q = tl.load(q_ptr + offs_q, mask=mask_q_block[:, None] & mask_d[None, :], other=0.0).to(tl.bfloat16)
    
    # Flash attention accumulators
    m = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM_PADDED], dtype=tl.float32)
    
    # Iterate over KV cache pages
    offs_kv_cache_block = tl.arange(0, BLOCK_N)
    mask_kv_cache_block = offs_kv_cache_block < PAGE_SIZE
    num_pages = (context_len + PAGE_SIZE - 1) // PAGE_SIZE
    
    for page_rel_id in range(0, num_pages):
        page_abs_id = tl.load(
            page_tables_ptr + req_id * page_tables_stride_nreqs + page_rel_id * page_tables_stride_pages
        ).to(tl.int32)
        page_token_ids = offs_kv_cache_block + page_rel_id * PAGE_SIZE
        page_token_valid_map = (page_abs_id >= 0) & (page_token_ids < context_len) & mask_kv_cache_block
        
        # Load K from FP8 cache and dequantize
        k_offs = (
            page_abs_id * k_cache_stride_npages
            + offs_kv_cache_block[:, None] * k_cache_stride_psz
            + kv_head_id * k_cache_stride_h
            + offs_d[None, :] * k_cache_stride_d
        )
        
        # Load as FP8 (stored as uint8), convert to BF16, then dequantize
        k_fp8 = tl.load(
            k_cache_ptr + k_offs,
            mask=page_token_valid_map[:, None] & mask_d[None, :],
            other=0.0,
        )
        
        # Dequantize: BF16 = FP8 * scale
        k = (k_fp8.to(tl.bfloat16) * k_scale.to(tl.bfloat16)).to(tl.bfloat16)
        
        # Compute attention scores
        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * softmax_scale
        scores = tl.where(mask_q_block[:, None] & page_token_valid_map[None, :], scores, float("-inf"))
        
        # Online softmax update
        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        l_new = l * tl.exp(m - m_new) + tl.sum(p, axis=1)
        alpha = tl.exp(m - m_new)
        acc *= alpha[:, None]
        
        # Load V from FP8 cache and dequantize
        v_offs = (
            page_abs_id * v_cache_stride_npages
            + offs_kv_cache_block[:, None] * v_cache_stride_psz
            + kv_head_id * v_cache_stride_h
            + offs_d[None, :] * v_cache_stride_d
        )
        
        v_fp8 = tl.load(
            v_cache_ptr + v_offs,
            mask=page_token_valid_map[:, None] & mask_d[None, :],
            other=0.0,
        )
        
        # Dequantize V
        v = (v_fp8.to(tl.bfloat16) * v_scale.to(tl.bfloat16)).to(tl.bfloat16)
        
        # Accumulate attention output
        acc += tl.dot(p.to(tl.bfloat16), v).to(tl.float32)
        m = m_new
        l = l_new
    
    # Normalize and store output
    out = acc / l[:, None]
    o_offs = (q_start + offs_q_block[:, None]) * q_stride_s + head_id * o_stride_h + offs_d[None, :] * o_stride_d
    tl.store(
        o_ptr + o_offs,
        out.to(tl.bfloat16),
        mask=mask_q_block[:, None] & mask_d[None, :],
    )


def fp8_kv_attention_forward(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    page_tables: torch.Tensor,
    context_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    softmax_scale: float,
    is_e4m3: bool = True,
) -> torch.Tensor:
    """
    FP8 KV Cache Attention Forward.
    
    Args:
        q: Query tensor [total_seqlen, num_heads, head_dim] (BF16)
        k_cache: Key cache [num_pages, page_size, num_kv_heads, head_dim] (FP8)
        v_cache: Value cache [num_pages, page_size, num_kv_heads, head_dim] (FP8)
        k_scale: Per-request K scales [num_reqs] (float32)
        v_scale: Per-request V scales [num_reqs] (float32)
        page_tables: Page table mapping [num_reqs, max_pages]
        context_lens: Context lengths [num_reqs]
        cu_seqlens_q: Cumulative sequence lengths [num_reqs + 1]
        softmax_scale: Softmax scaling factor
        is_e4m3: True for E4M3, False for E5M2
        
    Returns:
        Output tensor [total_seqlen, num_heads, head_dim] (BF16)
    """
    o = torch.empty_like(q)
    num_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    num_groups = num_heads // num_kv_heads
    
    head_dim = q.shape[-1]
    head_dim_padded = 1 << (head_dim - 1).bit_length()
    page_size = k_cache.shape[1]
    num_reqs = cu_seqlens_q.shape[0] - 1
    
    # Fixed block sizes for FP8 attention
    BLOCK_M = 64
    BLOCK_N = 64
    
    # Determine max sequence length
    max_seqlen = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
    
    grid = (num_reqs, num_heads, triton.cdiv(max_seqlen, BLOCK_M))
    
    _fp8_kv_attention_kernel[grid](
        q,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        o,
        page_tables,
        context_lens,
        cu_seqlens_q,
        softmax_scale,
        *q.stride(),
        *k_cache.stride(),
        *v_cache.stride(),
        *page_tables.stride(),
        NUM_GROUPS=num_groups,
        HEAD_DIM=head_dim,
        HEAD_DIM_PADDED=head_dim_padded,
        PAGE_SIZE=page_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        USE_FP8_E4M3=is_e4m3,
    )
    
    return o


@_register("triton_fp8_kv_attention")
class Fp8KVAttentionKernel(KVCacheKernel):
    """
    FP8 KV Cache Attention Triton Kernel.
    
    Performs attention with FP8 quantized KV cache, dequantizing on-the-fly.
    """
    
    name = "triton_fp8_kv_attention"
    description = "Custom Triton kernel for FP8 KV cache attention"
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if Triton and CUDA are available."""
        try:
            import triton
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        page_tables: torch.Tensor,
        context_lens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        softmax_scale: float,
        is_e4m3: bool = True,
    ) -> torch.Tensor:
        """Compute attention with FP8 KV cache."""
        return fp8_kv_attention_forward(
            q, k_cache, v_cache, k_scale, v_scale,
            page_tables, context_lens, cu_seqlens_q,
            softmax_scale, is_e4m3
        )
