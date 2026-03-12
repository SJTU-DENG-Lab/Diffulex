"""
AWQ Marlin W4A16 Linear Strategy

AWQ 4-bit weights repacked for Marlin kernel.
Combines AWQ's activation-aware quantization with Marlin's fast GEMM.
"""

import torch
from typing import Any, Dict, Optional, Tuple

from ..strategy import LinearQuantizationStrategy
from ..registry import register_linear_strategy
from ..kernels.kernel_availability import warn_kernel_unavailable, check_vllm_op_available


@register_linear_strategy("awq_marlin_w4a16", "bf16")
class AWQMarlinW4A16LinearStrategy(LinearQuantizationStrategy):
    """
    AWQ + Marlin W4A16 linear quantization.
    
    Uses pre-repacked Marlin format weights from AWQ checkpoints.
    """
    
    def __init__(self, bits: int = 4, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size
        
        # Check for Marlin ops
        self.marlin_gemm = None
        self.awq_marlin_repack = None
        self._kernel_warned = False
        
        if check_vllm_op_available('gptq_marlin_gemm'):
            import vllm._custom_ops as ops
            self.marlin_gemm = ops.gptq_marlin_gemm
            if hasattr(ops, 'awq_marlin_repack'):
                self.awq_marlin_repack = ops.awq_marlin_repack
        # Note: Warning is deferred to first forward call
    
    @property
    def name(self) -> str:
        return f"awq_marlin_w4a16_g{self.group_size}"
    
    @property
    def linear_weight_format(self) -> str:
        return "marlin_int4"
    
    @property
    def linear_act_format(self) -> str:
        return "bf16"
    
    def get_storage_dtype(self, device: torch.device) -> Tuple[torch.dtype, int]:
        return (torch.int32, 0)
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("AWQ Marlin requires offline quantization")
    
    def dequantize(self, q_x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("AWQ Marlin requires kernel-based dequantization")
    
    def quantize_weight_for_kernel(self, weight: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError("AWQ Marlin requires offline weight quantization")
    
    def quantize_act_for_kernel(self, x: torch.Tensor,
                                cache_key: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return x, {}
    
    def linear_forward(self, x: torch.Tensor, weight: torch.Tensor,
                       bias: Optional[torch.Tensor] = None,
                       *, quant_kind: str = "other", **kwargs) -> torch.Tensor:
        """
        AWQ Marlin linear forward.
        
        Same interface as GPTQ Marlin.
        """
        marlin_qweight = kwargs.get('marlin_qweight')
        marlin_scales = kwargs.get('marlin_scales')
        marlin_workspace = kwargs.get('marlin_workspace')
        num_bits = kwargs.get('num_bits', self.bits)
        is_k_full = kwargs.get('is_k_full', True)
        
        if marlin_qweight is None:
            raise ValueError("AWQ Marlin forward requires 'marlin_qweight' buffer")
        if marlin_scales is None:
            raise ValueError("AWQ Marlin forward requires 'marlin_scales' buffer")
        if marlin_workspace is None:
            raise ValueError("AWQ Marlin forward requires 'marlin_workspace' buffer")
        
        if self.marlin_gemm is not None:
            try:
                x_2d = x.reshape(-1, x.shape[-1])
                
                output = self.marlin_gemm(
                    x_2d,
                    marlin_qweight,
                    marlin_scales,
                    marlin_workspace,
                    num_bits,
                    is_k_full
                )
                
                output_shape = list(x.shape[:-1]) + [marlin_scales.shape[1]]
                output = output.reshape(output_shape)
                
                if bias is not None:
                    output = output + bias
                
                return output
            except Exception as e:
                if not self._kernel_warned:
                    warn_kernel_unavailable(
                        "vllm.gptq_marlin_gemm",
                        self.name,
                        f"error: {e}"
                    )
                    self._kernel_warned = True
                raise
        else:
            if not self._kernel_warned:
                warn_kernel_unavailable(
                    "vllm.gptq_marlin_gemm",
                    self.name,
                    "no fallback available"
                )
                self._kernel_warned = True
            raise RuntimeError(
                f"[{self.name}] Marlin GEMM kernel is required but not available. "
                f"Please install vLLM with CUDA support."
            )
    
    def repack_awq_to_marlin(self, qweight: torch.Tensor, qzeros: torch.Tensor,
                             scales: torch.Tensor, num_bits: int = 4) -> Dict[str, Any]:
        """
        Repack AWQ weights to Marlin format.
        
        Called during weight loading.
        """
        if self.awq_marlin_repack is not None:
            try:
                # Call vLLM's AWQ Marlin repack
                marlin_qweight, marlin_scales = self.awq_marlin_repack(
                    qweight, qzeros, scales, num_bits
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
            except Exception as e:
                warn_kernel_unavailable(
                    "vllm.awq_marlin_repack",
                    self.name,
                    f"error: {e}"
                )
                raise
        else:
            warn_kernel_unavailable(
                "vllm.awq_marlin_repack",
                self.name,
                "no fallback available"
            )
            raise RuntimeError(
                f"[{self.name}] AWQ Marlin repack kernel is required but not available. "
                f"Please install vLLM with CUDA support."
            )
