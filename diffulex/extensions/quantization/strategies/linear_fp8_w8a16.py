"""
FP8 W8A16 Linear Strategy - vLLM-aligned high-performance implementation.

Key optimizations (from feat/kv-cache-fp8-support):
1. Weight quantization: vllm._custom_ops.scaled_fp8_quant (CUDA kernel, per-tensor)
2. GEMM: vllm._custom_ops.cutlass_scaled_mm (CUTLASS)
   - Activation stays in BF16, but is internally quantized to FP8 for GEMM
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


@register_linear_strategy("fp8_e4m3", "bf16")
class FP8E4M3W8A16LinearStrategy(LinearQuantizationStrategy):
    """
    FP8 E4M3 W8A16 linear quantization using vLLM's cutlass_scaled_mm.
    
    Weight layout: stored as [K, N] fp8 (transposed from original [N, K])
    Scale layout: [1] float32 for per-tensor scaling
    
    Activation remains in BF16, but cutlass_scaled_mm handles the mixed-type GEMM.
    """
    
    def __init__(self):
        # Cache: id(weight) -> (qweight_fp8 [K,N], w_scale_fp32 [1])
        self._weight_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    
    @property
    def name(self) -> str:
        return "fp8_e4m3_w8a16"
    
    @property
    def linear_weight_format(self) -> str:
        return "fp8_e4m3"
    
    @property
    def linear_act_format(self) -> str:
        return "bf16"
    
    def get_storage_dtype(self, device: torch.device) -> Tuple[torch.dtype, int]:
        return (torch.float8_e4m3fn, 1)
    
    def get_scale_shape(self, original_shape: Tuple[int, ...], **kwargs: Any) -> Tuple[int, ...]:
        """Return scale shape for weight quantization."""
        if len(original_shape) != 2:
            raise ValueError(f"Expected 2D weight [N,K], got {original_shape}")
        return (1,)  # Per-tensor scale
    
    def quantize(self, weight: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, Any]:
        """
        Quantize weight to FP8 using vLLM's CUDA kernel.
        
        Args:
            weight: [N, K] float tensor
            
        Returns:
            qweight: [K, N] fp8 (transposed for CUTLASS)
            metadata: {"scales": [1] float32}
        """
        if weight.dim() != 2:
            raise ValueError(f"Expected 2D weight [N,K], got shape={tuple(weight.shape)}")
        
        if _vllm_ops is None:
            raise RuntimeError("vLLM custom ops not available")
        
        # Use vLLM's fast FP8 quantization
        q_fp8, scale = _vllm_ops.scaled_fp8_quant(
            weight.to(torch.float32).contiguous(),
            scale=None
        )
        
        # Transpose to [K,N] for CUTLASS (stride(0)==1)
        q_kn_fp8 = q_fp8.t()
        
        # Scale as scalar [1]
        scale = scale.to(torch.float32).reshape(1).contiguous()
        
        return q_kn_fp8, {"scales": scale}
    
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
            qweight: [K, N] fp8 on target device
            scales: [1] float32 on target device
        """
        q_fp8, meta = self.quantize(weight)
        if device is not None:
            q_fp8 = q_fp8.to(device=device)
            meta["scales"] = meta["scales"].to(device=device)
        return q_fp8, meta["scales"]
    
    def quantize_act_for_kernel(self, x: torch.Tensor,
                                cache_key: Optional[str] = None) -> Tuple[torch.Tensor, Any]:
        """No activation quantization in W8A16 - keep BF16."""
        return x, {"scale": None}
    
    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs: Any) -> torch.Tensor:
        """Dequantize - NOT SUPPORTED to prevent slow fallback."""
        raise RuntimeError(
            "FP8 W8A16 does not provide dequantize path (avoid slow BF16 GEMM). "
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
        out_features: Optional[int] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        FP8 W8A16 linear forward using vLLM's cutlass_scaled_mm.
        
        Args:
            x: Input tensor [..., K] (BF16)
            weight: Quantized weight [K, N] fp8, or original weight
            bias: Optional bias [N]
            quant_scales: Weight scale scalar (if weight is already quantized)
            
        Returns:
            output: [..., N] (BF16)
        """
        if _vllm_ops is None:
            raise RuntimeError(
                "vLLM custom ops are required for FP8 W8A16 (scaled_fp8_quant / cutlass_scaled_mm)"
            )
        
        # Get quantized weight and scales
        if weight is not None and weight.dtype == torch.float8_e4m3fn and quant_scales is not None:
            # Already quantized (from load-time quantization)
            qweight = weight
            w_scale = quant_scales
        else:
            # Need to quantize on-the-fly (cache by weight id)
            wid = id(weight)
            cached = self._weight_cache.get(wid)
            if cached is None or cached[0].device != x.device:
                qweight, w_scale = self.quantize_weight_for_kernel(weight, device=x.device)
                self._weight_cache[wid] = (qweight, w_scale)
            else:
                qweight, w_scale = cached
        
        # Reshape input: [..., K] -> [M, K]
        orig_shape = x.shape
        x2 = x.reshape(-1, x.shape[-1]) if x.dim() != 2 else x
        
        # Ensure correct dtype and contiguity
        if x2.dtype not in (torch.bfloat16, torch.float16):
            x2 = x2.to(torch.bfloat16)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        
        # For W8A16, we need to quantize activation to FP8 for cutlass_scaled_mm
        # Then the output will be dequantized back to BF16
        x_q, x_s = _vllm_ops.scaled_fp8_quant(x2, scale=None)
        
        # cutlass_scaled_mm: (M, K) @ (K, N) -> (M, N)
        output = _vllm_ops.cutlass_scaled_mm(
            x_q,           # [M, K] fp8
            qweight,       # [K, N] fp8, stride(0)==1
            scale_a=x_s,   # [1] - per-tensor activation scale
            scale_b=w_scale,  # [1] - per-tensor weight scale
            out_dtype=x2.dtype,  # bfloat16 or float16
            bias=bias.to(dtype=x2.dtype) if bias is not None else None,
        )
        
        # Reshape output back
        if x.dim() == 1:
            return output.squeeze(0)
        return output.reshape(*orig_shape[:-1], output.shape[-1])


@register_linear_strategy("fp8_e5m2", "bf16")
class FP8E5M2W8A16LinearStrategy(FP8E4M3W8A16LinearStrategy):
    """FP8 E5M2 W8A16 linear quantization."""
    
    def __init__(self):
        super().__init__()
    
    @property
    def name(self) -> str:
        return "fp8_e5m2_w8a16"
    
    @property
    def linear_weight_format(self) -> str:
        return "fp8_e5m2"
    
    def get_storage_dtype(self, device: torch.device) -> Tuple[torch.dtype, int]:
        return (torch.float8_e5m2, 1)
    
    def quantize(self, weight: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, Any]:
        """Quantize weight to FP8 E5M2."""
        if weight.dim() != 2:
            raise ValueError(f"Expected 2D weight [N,K], got shape={tuple(weight.shape)}")
        
        if _vllm_ops is None:
            raise RuntimeError("vLLM custom ops not available")
        
        # Use vLLM's fast FP8 quantization
        q_fp8, scale = _vllm_ops.scaled_fp8_quant(
            weight.to(torch.float32).contiguous(),
            scale=None
        )
        
        # Cast to e5m2 if needed
        if q_fp8.dtype != torch.float8_e5m2:
            q_fp8 = q_fp8.to(torch.float8_e5m2)
        
        # Transpose to [K,N] for CUTLASS
        q_kn_fp8 = q_fp8.t()
        scale = scale.to(torch.float32).reshape(1).contiguous()
        
        return q_kn_fp8, {"scales": scale}
