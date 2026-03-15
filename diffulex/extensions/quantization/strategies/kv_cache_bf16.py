"""
BF16 KV Cache Strategy (No Quantization)

No-op KV cache strategy that keeps tensors in BF16 format.
"""

import torch
from typing import Optional, Tuple

from ..strategy import KVCacheQuantizationStrategy
from ..registry import register_kv_cache_strategy


@register_kv_cache_strategy("bf16")
class BF16KVCacheStrategy(KVCacheQuantizationStrategy):
    """BF16 KV cache - no quantization."""
    
    @property
    def name(self) -> str:
        return "bf16_kv_cache"
    
    @property
    def requires_kv_cache_scales(self) -> bool:
        return False
    
    def get_storage_dtype(self, device: torch.device) -> Tuple[torch.dtype, int]:
        return (torch.bfloat16, 2)
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """No quantization - return as-is."""
        return x, torch.tensor(1.0, dtype=x.dtype, device=x.device)
    
    def dequantize(self, q_x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """No dequantization - return as-is."""
        return q_x
    
    def compute_scales(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """No scales needed for BF16."""
        dummy = torch.tensor(1.0, dtype=k.dtype, device=k.device)
        return dummy, dummy
    
    def update_scales(self, k: torch.Tensor, v: torch.Tensor,
                      k_scale: torch.Tensor, v_scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """No scale updates needed."""
        return k_scale, v_scale
    
    def init_scales(self, batch_size: int, num_heads: int,
                    device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize dummy scales."""
        dummy = torch.tensor(1.0, dtype=dtype, device=device)
        return dummy, dummy
    
    def quantize_kv_for_store(self, k: torch.Tensor, v: torch.Tensor,
                              k_scale: Optional[torch.Tensor] = None,
                              v_scale: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """No quantization - return as-is."""
        return k, v
    
    def dequantize_kv_for_compute(self, q_k: torch.Tensor, q_v: torch.Tensor,
                                  k_scale: Optional[torch.Tensor] = None,
                                  v_scale: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """No dequantization - return as-is."""
        return q_k, q_v
