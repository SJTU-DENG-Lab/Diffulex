"""
BF16 Linear Strategy (No Quantization)

No-op linear strategy that performs standard BF16 matmul.
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

from ..strategy import LinearQuantizationStrategy
from ..registry import register_linear_strategy


@register_linear_strategy("bf16", "bf16")
class BF16LinearStrategy(LinearQuantizationStrategy):
    """BF16 linear - standard matmul without quantization."""
    
    @property
    def name(self) -> str:
        return "bf16_linear"
    
    @property
    def linear_weight_format(self) -> str:
        return "bf16"
    
    @property
    def linear_act_format(self) -> str:
        return "bf16"
    
    def get_storage_dtype(self, device: torch.device) -> Tuple[torch.dtype, int]:
        return (torch.bfloat16, 2)
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """No quantization."""
        return x, torch.tensor(1.0, dtype=x.dtype, device=x.device)
    
    def dequantize(self, q_x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """No dequantization."""
        return q_x
    
    def quantize_weight_for_kernel(self, weight: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """No quantization."""
        return weight, {}
    
    def quantize_act_for_kernel(self, x: torch.Tensor,
                                cache_key: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """No quantization."""
        return x, {}
    
    def linear_forward(self, x: torch.Tensor, weight: torch.Tensor,
                       bias: Optional[torch.Tensor] = None,
                       *, quant_kind: str = "other", **kwargs) -> torch.Tensor:
        """Standard BF16 linear forward."""
        return F.linear(x, weight, bias)
