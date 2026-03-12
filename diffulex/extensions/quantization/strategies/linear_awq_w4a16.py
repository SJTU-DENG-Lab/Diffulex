"""
AWQ W4A16 Linear Strategy

Activation-aware weight quantization with BF16 activation.
Uses AWQ format with packed weights and group-wise scales.
"""

import torch
from typing import Any, Dict, Optional, Tuple

from ..strategy import LinearQuantizationStrategy
from ..registry import register_linear_strategy
from ..kernels.kernel_availability import warn_kernel_unavailable, check_vllm_op_available


@register_linear_strategy("awq_w4a16", "bf16")
class AWQW4A16LinearStrategy(LinearQuantizationStrategy):
    """
    AWQ W4A16 linear quantization.
    
    Expects pre-quantized weights in AWQ format:
    - qweight: int32 packed weights
    - qzeros: int32 packed zeros  
    - scales: float16/bfloat16 scales
    """
    
    def __init__(self, bits: int = 4, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size
        
        # Check for AWQ ops
        self.awq_gemm = None
        try:
            import vllm._custom_ops as ops
            if hasattr(ops, 'awq_gemm'):
                self.awq_gemm = ops.awq_gemm
        except (ImportError, AttributeError):
            pass
    
    @property
    def name(self) -> str:
        return f"awq_w4a16_g{self.group_size}"
    
    @property
    def linear_weight_format(self) -> str:
        return "awq_int4"
    
    @property
    def linear_act_format(self) -> str:
        return "bf16"
    
    def get_storage_dtype(self, device: torch.device) -> Tuple[torch.dtype, int]:
        return (torch.int32, 0)
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("AWQ requires offline quantization")
    
    def dequantize(self, q_x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("AWQ requires kernel-based dequantization")
    
    def quantize_weight_for_kernel(self, weight: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError("AWQ requires offline weight quantization")
    
    def quantize_act_for_kernel(self, x: torch.Tensor,
                                cache_key: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return x, {}
    
    def linear_forward(self, x: torch.Tensor, weight: torch.Tensor,
                       bias: Optional[torch.Tensor] = None,
                       *, quant_kind: str = "other", **kwargs) -> torch.Tensor:
        """
        AWQ linear forward.
        
        Expects AWQ-specific kwargs:
        - qweight: int32 packed weights
        - qzeros: int32 packed zeros
        - scales: float scales
        - bits: int (default 4)
        """
        qweight = kwargs.get('qweight')
        qzeros = kwargs.get('qzeros')
        scales = kwargs.get('scales')
        bits = kwargs.get('bits', self.bits)
        
        if qweight is None:
            raise ValueError("AWQ forward requires 'qweight' buffer")
        if qzeros is None:
            raise ValueError("AWQ forward requires 'qzeros' buffer")
        if scales is None:
            raise ValueError("AWQ forward requires 'scales' buffer")
        
        # Use vLLM AWQ GEMM if available
        if self.awq_gemm is not None:
            try:
                x_2d = x.reshape(-1, x.shape[-1])
                
                output = self.awq_gemm(
                    x_2d,
                    qweight,
                    scales,
                    qzeros,
                    bits
                )
                
                output_shape = list(x.shape[:-1]) + [scales.shape[1]]
                output = output.reshape(output_shape)
                
                if bias is not None:
                    output = output + bias
                
                return output
            except Exception:
                pass
        
        # Fallback: manual dequantization
        w_deq = self._manual_dequantize(qweight, qzeros, scales, bits)
        return torch.nn.functional.linear(x, w_deq.t(), bias)
    
    def _manual_dequantize(self, qweight: torch.Tensor, qzeros: torch.Tensor,
                          scales: torch.Tensor, bits: int) -> torch.Tensor:
        """Manual AWQ dequantization for fallback."""
        in_features = qweight.shape[0] * 32 // bits
        out_features = qweight.shape[1]
        num_groups = scales.shape[0]
        
        # Unpack weights
        wf = torch.arange(0, 32, bits, device=qweight.device)
        
        weight = torch.bitwise_right_shift(
            qweight.unsqueeze(-1).expand(-1, -1, 32 // bits),
            wf
        ).bitwise_and(2 ** bits - 1)
        weight = weight.reshape(in_features, out_features)
        
        # Unpack zeros
        zeros = torch.bitwise_right_shift(
            qzeros.unsqueeze(-1).expand(-1, -1, 32 // bits),
            wf
        ).bitwise_and(2 ** bits - 1)
        zeros = zeros.reshape(num_groups, out_features)
        zeros = zeros + 1
        
        # Dequantize with uniform groups
        group_size = in_features // num_groups
        weight = weight.to(torch.bfloat16)
        for i in range(num_groups):
            start = i * group_size
            end = min((i + 1) * group_size, in_features)
            weight[start:end] = (weight[start:end] - zeros[i].to(torch.bfloat16)) * scales[i].to(torch.bfloat16)
        
        return weight
    
    def prepare_awq_weight(self, qweight: torch.Tensor, qzeros: torch.Tensor,
                          scales: torch.Tensor, bits: int = 4,
                          device: torch.device = None) -> Dict[str, Any]:
        """Prepare AWQ weight buffers from model checkpoint."""
        if device is not None:
            qweight = qweight.to(device)
            qzeros = qzeros.to(device)
            scales = scales.to(device)
        
        return {
            'qweight': qweight,
            'qzeros': qzeros,
            'scales': scales,
            'bits': bits,
        }
