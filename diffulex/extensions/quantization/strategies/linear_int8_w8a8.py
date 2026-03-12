"""
INT8 W8A8 Linear Strategy - vLLM-aligned high-performance implementation.

Key optimizations (from feat/kv-cache-fp8-support):
1. Activation quantization: vllm._custom_ops.scaled_int8_quant (CUDA kernel, dynamic per-token)
2. GEMM: vllm._custom_ops.cutlass_scaled_mm (CUTLASS, no fallback)
3. Weight layout: stored as K×N (transposed), matching CUTLASS requirements

No dequantize fallback - forces CUTLASS path for performance.
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

from ..strategy import LinearQuantizationStrategy
from ..registry import register_linear_strategy


try:
    from vllm import _custom_ops as _vllm_ops
except Exception:
    _vllm_ops = None


@register_linear_strategy("int8", "int8")
class INT8W8A8LinearStrategy(LinearQuantizationStrategy):
    """
    INT8 W8A8 linear quantization using vLLM's optimized CUDA kernels.
    
    Weight layout: stored as [K, N] int8 (transposed from original [N, K])
    Scale layout: [1, N] float32 for broadcasting with per-token activation scales
    """
    
    def __init__(self):
        # Cache: id(weight) -> (qweight_int8 [K,N], w_scales_fp32 [1,N])
        self._weight_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    
    @property
    def name(self) -> str:
        return "int8_w8a8"
    
    @property
    def linear_weight_format(self) -> str:
        return "int8"
    
    @property
    def linear_act_format(self) -> str:
        return "int8"
    
    def get_storage_dtype(self, device: torch.device) -> Tuple[torch.dtype, int]:
        return (torch.int8, 1)
    
    def get_scale_shape(self, original_shape: Tuple[int, ...], **kwargs: Any) -> Tuple[int, ...]:
        """Return scale shape for weight quantization."""
        if len(original_shape) != 2:
            raise ValueError(f"Expected 2D weight [N,K], got {original_shape}")
        return (original_shape[0],)  # Per-output-channel: [N]
    
    def quantize(self, weight: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, Any]:
        """
        Quantize weight to INT8.
        
        Args:
            weight: [N, K] float tensor
            
        Returns:
            qweight: [K, N] int8 (transposed for CUTLASS)
            metadata: {"scales": [1, N] float32}
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
    ) -> Tuple[torch.Tensor, Any]:
        """
        Quantize weight for kernel consumption.
        
        Returns:
            qweight: [K, N] int8 on target device
            scales: [1, N] float32 on target device
        """
        q_kn, meta = self.quantize(weight)
        if device is not None:
            q_kn = q_kn.to(device=device)
            meta["scales"] = meta["scales"].to(device=device)
        return q_kn, meta["scales"]
    
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
            x: Input tensor [..., K]
            weight: Quantized weight [K, N] int8, or original weight
            bias: Optional bias [N]
            quant_scales: Weight scales [1, N] float32 (if weight is already quantized)
            
        Returns:
            output: [..., N]
        """
        if _vllm_ops is None:
            raise RuntimeError(
                "vLLM custom ops are required for W8A8 (scaled_int8_quant / cutlass_scaled_mm)"
            )
        
        # Get quantized weight and scales
        if weight is not None and weight.dtype == torch.int8 and quant_scales is not None:
            # Already quantized (from load-time quantization)
            qweight = weight
            w_scales = quant_scales
        else:
            # Need to quantize on-the-fly (cache by weight id)
            wid = id(weight)
            cached = self._weight_cache.get(wid)
            if cached is None or cached[0].device != x.device:
                qweight, w_scales = self.quantize_weight_for_kernel(weight, device=x.device)
                self._weight_cache[wid] = (qweight, w_scales)
            else:
                qweight, w_scales = cached
        
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
