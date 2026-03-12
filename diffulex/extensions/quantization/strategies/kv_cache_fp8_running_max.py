"""
FP8 KV Cache Strategy with Running Max

FP8 KV cache quantization using running max for dynamic scales.
Supports both E4M3 and E5M2 formats.

Includes custom Triton kernel support for on-the-fly dequantization.
"""

import torch
from typing import Optional, Tuple

from ..strategy import KVCacheQuantizationStrategy
from ..registry import register_kv_cache_strategy

# Try to import custom FP8 Triton kernel
try:
    from ..kernels.triton_kernels import fp8_kv_attention_forward
    _HAS_FP8_TRITON_KERNEL = True
except ImportError:
    _HAS_FP8_TRITON_KERNEL = False


# FP8 constants
FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0


def get_fp8_max(dtype: torch.dtype) -> float:
    """Get max representable value for FP8 dtype."""
    if dtype == torch.float8_e4m3fn:
        return FP8_E4M3_MAX
    elif dtype == torch.float8_e5m2:
        return FP8_E5M2_MAX
    else:
        return FP8_E4M3_MAX


def fp8_quantize(x: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to FP8.
    
    Args:
        x: Input tensor
        dtype: Target FP8 dtype
        
    Returns:
        Tuple of (quantized_tensor, scale)
    """
    fp8_max = get_fp8_max(dtype)
    
    # Compute per-tensor max
    x_max = x.abs().max().float()
    
    # Compute scale
    scale = x_max / fp8_max
    scale = torch.clamp(scale, min=1e-12)
    
    # Quantize
    x_fp8 = (x / scale).to(dtype)
    
    return x_fp8, scale


def fp8_dequantize(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 tensor."""
    return x_fp8.to(torch.bfloat16) * scale.to(torch.bfloat16)


@register_kv_cache_strategy("fp8")
@register_kv_cache_strategy("fp8_e4m3")
class FP8E4M3KVCacheStrategy(KVCacheQuantizationStrategy):
    """FP8 E4M3 KV cache with running max scales."""
    
    def __init__(self):
        self.fp8_dtype = torch.float8_e4m3fn
    
    @property
    def name(self) -> str:
        return "fp8_e4m3_kv_cache"
    
    @property
    def requires_kv_cache_scales(self) -> bool:
        return True
    
    def get_storage_dtype(self, device: torch.device) -> Tuple[torch.dtype, int]:
        return (self.fp8_dtype, 1)
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize to FP8 E4M3."""
        return fp8_quantize(x, self.fp8_dtype)
    
    def dequantize(self, q_x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize from FP8."""
        return fp8_dequantize(q_x, scale)
    
    def compute_scales(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute initial scales for K and V."""
        k_scale = k.abs().max().float() / FP8_E4M3_MAX
        v_scale = v.abs().max().float() / FP8_E4M3_MAX
        k_scale = torch.clamp(k_scale, min=1e-12)
        v_scale = torch.clamp(v_scale, min=1e-12)
        return k_scale, v_scale
    
    def update_scales(self, k: torch.Tensor, v: torch.Tensor,
                      k_scale: torch.Tensor, v_scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update scales using running max."""
        new_k_scale = k.abs().max().float() / FP8_E4M3_MAX
        new_v_scale = v.abs().max().float() / FP8_E4M3_MAX
        new_k_scale = torch.clamp(new_k_scale, min=1e-12)
        new_v_scale = torch.clamp(new_v_scale, min=1e-12)
        
        # Running max
        k_scale = torch.maximum(k_scale, new_k_scale)
        v_scale = torch.maximum(v_scale, new_v_scale)
        
        return k_scale, v_scale
    
    def init_scales(self, batch_size: int, num_heads: int,
                    device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize scales to small value."""
        k_scale = torch.tensor(1e-12, dtype=torch.float32, device=device)
        v_scale = torch.tensor(1e-12, dtype=torch.float32, device=device)
        return k_scale, v_scale
    
    def quantize_kv_for_store(self, k: torch.Tensor, v: torch.Tensor,
                              k_scale: Optional[torch.Tensor] = None,
                              v_scale: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize K and V for storage."""
        if k_scale is not None and v_scale is not None:
            # Use provided scales
            q_k = (k / k_scale.to(k.dtype)).to(self.fp8_dtype)
            q_v = (v / v_scale.to(v.dtype)).to(self.fp8_dtype)
        else:
            # Compute new scales
            q_k, _ = fp8_quantize(k, self.fp8_dtype)
            q_v, _ = fp8_quantize(v, self.fp8_dtype)
        
        return q_k, q_v
    
    def dequantize_kv_for_compute(self, q_k: torch.Tensor, q_v: torch.Tensor,
                                  k_scale: Optional[torch.Tensor] = None,
                                  v_scale: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize K and V for computation."""
        if k_scale is not None:
            k = q_k.to(torch.bfloat16) * k_scale.to(torch.bfloat16)
        else:
            k = q_k.to(torch.bfloat16)
        
        if v_scale is not None:
            v = q_v.to(torch.bfloat16) * v_scale.to(torch.bfloat16)
        else:
            v = q_v.to(torch.bfloat16)
        
        return k, v
    
    def has_triton_kernel(self) -> bool:
        """Check if custom Triton kernel is available."""
        return _HAS_FP8_TRITON_KERNEL
    
    def triton_attention(
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
    ) -> Optional[torch.Tensor]:
        """
        Compute attention using custom FP8 Triton kernel.
        
        This avoids explicit dequantization by doing it on-the-fly in the kernel.
        
        Args:
            q: Query tensor [total_seqlen, num_heads, head_dim]
            k_cache: Key cache in FP8 [num_pages, page_size, num_kv_heads, head_dim]
            v_cache: Value cache in FP8 [num_pages, page_size, num_kv_heads, head_dim]
            k_scale: Per-request K scales
            v_scale: Per-request V scales
            page_tables: Page table mapping
            context_lens: Context lengths per request
            cu_seqlens_q: Cumulative sequence lengths
            softmax_scale: Softmax scaling factor
            
        Returns:
            Attention output or None if kernel fails
        """
        if not _HAS_FP8_TRITON_KERNEL:
            return None
        
        try:
            return fp8_kv_attention_forward(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                k_scale=k_scale,
                v_scale=v_scale,
                page_tables=page_tables,
                context_lens=context_lens,
                cu_seqlens_q=cu_seqlens_q,
                softmax_scale=softmax_scale,
                is_e4m3=(self.fp8_dtype == torch.float8_e4m3fn),
            )
        except Exception:
            return None


@register_kv_cache_strategy("fp8_e5m2")
class FP8E5M2KVCacheStrategy(FP8E4M3KVCacheStrategy):
    """FP8 E5M2 KV cache with running max scales."""
    
    def __init__(self):
        self.fp8_dtype = torch.float8_e5m2
    
    @property
    def name(self) -> str:
        return "fp8_e5m2_kv_cache"
    
    def compute_scales(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute initial scales for K and V."""
        k_scale = k.abs().max().float() / FP8_E5M2_MAX
        v_scale = v.abs().max().float() / FP8_E5M2_MAX
        k_scale = torch.clamp(k_scale, min=1e-12)
        v_scale = torch.clamp(v_scale, min=1e-12)
        return k_scale, v_scale
    
    def update_scales(self, k: torch.Tensor, v: torch.Tensor,
                      k_scale: torch.Tensor, v_scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update scales using running max."""
        new_k_scale = k.abs().max().float() / FP8_E5M2_MAX
        new_v_scale = v.abs().max().float() / FP8_E5M2_MAX
        new_k_scale = torch.clamp(new_k_scale, min=1e-12)
        new_v_scale = torch.clamp(new_v_scale, min=1e-12)
        
        k_scale = torch.maximum(k_scale, new_k_scale)
        v_scale = torch.maximum(v_scale, new_v_scale)
        
        return k_scale, v_scale
