"""
GPTQ Marlin W4A16 Linear Strategy

GPTQ 4-bit weights repacked for Marlin kernel.
Uses Marlin's optimized GEMM implementation.
"""

import torch
from typing import Any, Dict, Optional, Tuple

from ..strategy import LinearQuantizationStrategy
from ..registry import register_linear_strategy
from ..kernels.kernel_availability import warn_kernel_unavailable, check_vllm_op_available


@register_linear_strategy("gptq_marlin_w4a16", "bf16")
class GPTQMarlinW4A16LinearStrategy(LinearQuantizationStrategy):
    """
    GPTQ + Marlin W4A16 linear quantization.
    
    Uses pre-repacked Marlin format weights for optimal performance.
    """
    
    def __init__(self, bits: int = 4, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size
        
        # Check for Marlin ops
        self.marlin_gemm = None
        self.gptq_marlin_repack = None
        try:
            import vllm._custom_ops as ops
            if hasattr(ops, 'gptq_marlin_gemm'):
                self.marlin_gemm = ops.gptq_marlin_gemm
            if hasattr(ops, 'gptq_marlin_repack'):
                self.gptq_marlin_repack = ops.gptq_marlin_repack
        except (ImportError, AttributeError):
            pass
    
    @property
    def name(self) -> str:
        return f"gptq_marlin_w4a16_g{self.group_size}"
    
    @property
    def linear_weight_format(self) -> str:
        return "marlin_int4"
    
    @property
    def linear_act_format(self) -> str:
        return "bf16"
    
    def get_storage_dtype(self, device: torch.device) -> Tuple[torch.dtype, int]:
        # Marlin uses int32 packed format
        return (torch.int32, 0)
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Marlin requires offline quantization")
    
    def dequantize(self, q_x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Marlin requires kernel-based dequantization")
    
    def quantize_weight_for_kernel(self, weight: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError("Marlin requires offline weight quantization")
    
    def quantize_act_for_kernel(self, x: torch.Tensor,
                                cache_key: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return x, {}
    
    def linear_forward(self, x: torch.Tensor, weight: torch.Tensor,
                       bias: Optional[torch.Tensor] = None,
                       *, quant_kind: str = "other", **kwargs) -> torch.Tensor:
        """
        Marlin linear forward.
        
        Expects Marlin-specific kwargs:
        - marlin_qweight: int32 repacked weights
        - marlin_scales: bfloat16/float16 scales
        - marlin_zp: int32 packed zeros (optional, for asymmetric)
        - marlin_g_idx: int32 group indices (optional)
        - marlin_workspace: int32 workspace buffer
        - num_bits: int (default 4)
        - is_k_full: bool (default True)
        """
        marlin_qweight = kwargs.get('marlin_qweight')
        marlin_scales = kwargs.get('marlin_scales')
        marlin_workspace = kwargs.get('marlin_workspace')
        num_bits = kwargs.get('num_bits', self.bits)
        is_k_full = kwargs.get('is_k_full', True)
        
        if marlin_qweight is None:
            raise ValueError("Marlin forward requires 'marlin_qweight' buffer")
        if marlin_scales is None:
            raise ValueError("Marlin forward requires 'marlin_scales' buffer")
        if marlin_workspace is None:
            raise ValueError("Marlin forward requires 'marlin_workspace' buffer")
        
        if self.marlin_gemm is None:
            raise RuntimeError("Marlin GEMM ops not available")
        
        # Reshape input to 2D
        x_2d = x.reshape(-1, x.shape[-1])
        
        # Call Marlin GEMM
        output = self.marlin_gemm(
            x_2d,
            marlin_qweight,
            marlin_scales,
            marlin_workspace,
            num_bits,
            is_k_full
        )
        
        # Reshape output
        output_shape = list(x.shape[:-1]) + [marlin_scales.shape[1]]
        output = output.reshape(output_shape)
        
        if bias is not None:
            output = output + bias
        
        return output
    
    def repack_gptq_to_marlin(self, qweight: torch.Tensor, qzeros: torch.Tensor,
                              scales: torch.Tensor, g_idx: Optional[torch.Tensor],
                              num_bits: int = 4) -> Dict[str, Any]:
        """
        Repack GPTQ weights to Marlin format.
        
        This is called during weight loading to prepare Marlin buffers.
        """
        if self.gptq_marlin_repack is None:
            raise RuntimeError("Marlin repack ops not available")
        
        # Call vLLM's repack function
        marlin_qweight, marlin_scales = self.gptq_marlin_repack(
            qweight, qzeros, scales, g_idx, num_bits
        )
        
        # Allocate workspace
        marlin_workspace = torch.zeros(
            marlin_qweight.shape[1] * 32 // num_bits,
            dtype=torch.int32,
            device=qweight.device
        )
        
        return {
            'marlin_qweight': marlin_qweight,
            'marlin_scales': marlin_scales,
            'marlin_workspace': marlin_workspace,
            'num_bits': num_bits,
            'is_k_full': True,
        }
