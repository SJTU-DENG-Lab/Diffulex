"""
INT8 W8A8 Linear Strategy - Pre-quantization only

Weight is pre-quantized during model loading. No runtime weight quantization.
"""

from typing import Any, Optional, Tuple

import torch
from torch import nn

from .linear_bf16 import BF16LinearStrategy
from ..registry import register_linear_strategy

# vLLM custom ops for fast INT8 W8A8
try:
    from vllm import _custom_ops as _vllm_ops
except ImportError:
    _vllm_ops = None


@register_linear_strategy("int8", "int8")
class INT8W8A8LinearStrategy(BF16LinearStrategy):
    """
    INT8 W8A8 quantization using vLLM's CUTLASS kernels.
    
    - Weight: per-channel symmetric int8, pre-quantized during loading
    - Activation: dynamic per-token int8 quantization
    - Kernel: vLLM's cutlass_scaled_mm
    
    NOTE: This strategy requires PRE-QUANTIZED weights. It will fail if
    called with unquantized (BF16) weights.
    """
    
    name = "int8_w8a8"
    
    def __init__(self):
        super().__init__()
    
    @property
    def linear_weight_format(self) -> str:
        return "int8"
    
    @property
    def linear_act_format(self) -> str:
        return "int8"
    
    def quantize(self, weight: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Quantize weight to INT8 (per-channel symmetric).
        
        Args:
            weight: [N, K] BF16/FP16 weight tensor
            
        Returns:
            qweight: [K, N] int8 (column-major for CUTLASS)
            meta: {"scales": [1, N] float32}
        """
        if weight.dim() != 2:
            raise ValueError(f"Expected 2D weight [N,K], got shape={tuple(weight.shape)}")
        
        # Per-output-channel symmetric int8 quantization
        w = weight.to(torch.float32)
        abs_max = w.abs().amax(dim=-1, keepdim=False)  # [N]
        scales = (abs_max.clamp(min=1e-8) / 127.0).to(torch.float32)  # [N]
        
        # Quantize to [N, K] int8
        q_nk = torch.round(w / scales.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)  # [N,K]
        
        # Transpose to [K, N] for CUTLASS (stride[0]==1, column-major)
        # CRITICAL: cutlass_scaled_mm requires b.stride(0) == 1
        # q_nk.t() creates a view with stride=(1, N), which satisfies the requirement
        q_kn = q_nk.t()  # [K,N], stride(0)==1
        
        # Scale as [1, N] for broadcasting
        scale_bn = scales.unsqueeze(0).contiguous()  # [1,N]
        
        return q_kn, {"scales": scale_bn}
    
    def quantize_weight_for_kernel(
        self,
        weight: torch.Tensor,
        *,
        device: Optional[torch.device] = None,
        **_: Any,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Quantize weight for kernel consumption.
        
        Returns:
            qweight: [K, N] int8 on target device
            meta: {"scales": [1, N] float32 on target device}
        """
        q_kn, meta = self.quantize(weight)
        if device is not None:
            q_kn = q_kn.to(device=device)
            meta["scales"] = meta["scales"].to(device=device)
        return q_kn, meta
    
    def quantize_act_for_kernel(self, x: torch.Tensor,
                                cache_key: Optional[str] = None) -> Tuple[torch.Tensor, Any]:
        """
        Quantize activation for kernel consumption.
        
        Note: In W8A8, activation quantization is done inside linear_forward
        using scaled_int8_quant for better performance. This method is kept
        for interface compatibility.
        """
        if _vllm_ops is None:
            raise RuntimeError("vLLM custom ops not available")
        
        # Reshape if needed
        orig_shape = x.shape
        x2 = x.reshape(-1, x.shape[-1]) if x.dim() != 2 else x
        
        if x2.dtype not in (torch.bfloat16, torch.float16):
            x2 = x2.to(torch.bfloat16)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        
        # Use vLLM's fast quantization
        x_q, x_s, _ = _vllm_ops.scaled_int8_quant(x2, scale=None, azp=None, symmetric=True)
        
        return x_q, {"scale": x_s, "orig_shape": orig_shape}
    
    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs: Any) -> torch.Tensor:
        """Dequantize - NOT SUPPORTED to prevent slow fallback."""
        raise RuntimeError(
            "W8A8 does not provide dequantize path (avoid slow BF16 GEMM). "
            "Use cutlass_scaled_mm only."
        )
    
    def linear_forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        quant_kind: str = "other",
        quant_scales: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        INT8 W8A8 linear forward using vLLM's cutlass_scaled_mm.
        
        Args:
            x: Input tensor [..., K] (BF16/FP16)
            weight: Pre-quantized weight [K, N] int8
            bias: Optional bias [N]
            quant_scales: Weight scales [1, N] float32
            
        Returns:
            output: [..., N] (BF16/FP16)
            
        Raises:
            RuntimeError: If weight is not pre-quantized (dtype != int8)
        """
        if _vllm_ops is None:
            raise RuntimeError(
                "vLLM custom ops are required for W8A8 (scaled_int8_quant / cutlass_scaled_mm)"
            )
        
        # STRICT: Only accept pre-quantized weights
        if weight is None or weight.dtype != torch.int8 or quant_scales is None:
            raise RuntimeError(
                f"INT8 W8A8 requires pre-quantized weight (dtype=int8) and scales, "
                f"got weight.dtype={weight.dtype if weight is not None else None}, "
                f"quant_scales={quant_scales is not None}"
            )
        
        qweight = weight
        w_scales = quant_scales
        
        # Reshape input: [..., K] -> [M, K]
        orig_shape = x.shape
        x2 = x.reshape(-1, x.shape[-1]) if x.dim() != 2 else x
        
        # Ensure correct dtype and contiguity for CUTLASS
        if x2.dtype not in (torch.bfloat16, torch.float16):
            x2 = x2.to(torch.bfloat16)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        
        # Dynamic per-token int8 quantization + fused GEMM+dequant
        # scaled_int8_quant returns: (quantized_tensor, scales, azp)
        x_q, x_s, _ = _vllm_ops.scaled_int8_quant(x2, scale=None, azp=None, symmetric=True)
        
        # cutlass_scaled_mm: (M, K) @ (K, N) -> (M, N)
        output = _vllm_ops.cutlass_scaled_mm(
            x_q,           # [M, K] int8
            qweight,       # [K, N] int8, stride(0)==1
            scale_a=x_s,   # [M, 1] or scalar - per-token activation scales
            scale_b=w_scales,  # [1, N] - per-channel weight scales
            out_dtype=x2.dtype,  # bfloat16 or float16
            bias=bias.to(dtype=x2.dtype) if bias is not None else None,
        )
        
        # Reshape output back
        if x.dim() == 1:
            return output.squeeze(0)
        return output.reshape(*orig_shape[:-1], output.shape[-1])
