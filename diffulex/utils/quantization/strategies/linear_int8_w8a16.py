"""
W8A16 Linear quantization strategy (int8 weight + bf16 activation).

Reference implementation using Python dequantization + torch.nn.functional.linear.
Future optimizations:
- Lazy cache quantized weights per module instance
- Replace F.linear with custom Triton/TileLang kernel for int8 GEMM
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from diffulex.utils.quantization.registry import register_linear_strategy
from diffulex.utils.quantization.strategy import LinearQuantizationStrategy


@register_linear_strategy(weight_dtype="int8", act_dtype="bf16")
def _build_linear_int8_w8a16() -> LinearQuantizationStrategy:
    return LinearInt8W8A16Strategy()


class LinearInt8W8A16Strategy(LinearQuantizationStrategy):
    """W8A16 Linear strategy: int8 weight quantization + bf16 activation.

    Current implementation: Python reference using dequantized weights + F.linear.
    Weight quantization: per-output-channel symmetric quantization to int8.
    Activation: kept as bf16 (no activation quantization).
    """

    @property
    def name(self) -> str:
        return "linear_int8_w8a16"

    @property
    def linear_weight_format(self) -> str:
        return "int8"

    @property
    def linear_act_format(self) -> str:
        return "bf16"

    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        # Weights are stored as int8 (1 byte per element)
        return torch.int8, 1

    def quantize(self, tensor: torch.Tensor, **kwargs) -> tuple[torch.Tensor, Any]:
        """Quantize tensor to int8 with per-channel (per-output) scales.
        
        Args:
            tensor: Weight tensor of shape [out_features, in_features]
            **kwargs: Additional arguments (unused for now)
        
        Returns:
            (quantized_tensor, scales): quantized_tensor is int8, scales is [out_features]
        """
        _ = kwargs
        # Per-output-channel quantization: compute scale for each output channel
        # shape: [out_features, in_features] -> scales shape: [out_features]
        abs_max = torch.abs(tensor).max(dim=-1, keepdim=True)[0]  # [out_features, 1]
        # Avoid division by zero
        scales = abs_max.clamp(min=1e-8) / 127.0  # [out_features, 1]
        
        # Quantize: round(clamp(tensor / scales, -128, 127))
        quantized = torch.round(tensor / scales).clamp(-128, 127).to(torch.int8)
        scales_1d = scales.squeeze(-1)  # [out_features]
        
        return quantized, scales_1d

    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs) -> torch.Tensor:
        """Dequantize int8 tensor back to bf16 using per-channel scales.
        
        Args:
            quantized: int8 tensor [out_features, in_features]
            scale_or_metadata: scales tensor [out_features] or dict with 'scales'
            **kwargs: Additional arguments (unused for now)
        
        Returns:
            Dequantized tensor in bf16
        """
        _ = kwargs
        if isinstance(scale_or_metadata, dict):
            scales = scale_or_metadata.get("scales")
        else:
            scales = scale_or_metadata
        
        if scales is None:
            raise ValueError("scales required for dequantization")
        
        # Ensure scales have correct shape for broadcasting
        if scales.dim() == 1:
            scales = scales.unsqueeze(-1)  # [out_features, 1]
        
        # Dequantize: quantized * scales
        dequantized = quantized.to(torch.float32) * scales
        return dequantized.to(torch.bfloat16)

    def get_scale_shape(self, original_shape: tuple[int, ...], **kwargs) -> tuple[int, ...]:
        """Return shape of scales tensor for per-channel quantization.
        
        For [out_features, in_features] weight, scales shape is [out_features].
        """
        _ = kwargs
        if len(original_shape) < 2:
            raise ValueError(f"Expected weight shape with at least 2 dims, got {original_shape}")
        # Per-output-channel: scales shape is [out_features]
        return (original_shape[0],)

    def quantize_weight_for_kernel(
        self,
        weight: torch.Tensor,
        *,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Quantize weight to int8 with per-channel scales.
        
        Returns:
            (quantized_weight, scales): quantized_weight is int8 [out, in], scales is [out]
        """
        _ = kwargs
        if device is not None:
            weight = weight.to(device=device)
        
        quantized, scales = self.quantize(weight)
        return quantized, scales

    def quantize_act_for_kernel(
        self,
        x: torch.Tensor,
        *,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        """No activation quantization for W8A16 (activation stays bf16)."""
        if device is not None:
            x = x.to(device=device)
        return x, None

    def linear_forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        quant_kind: str,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute Linear output using quantized weights (W8A16).
        
        Current implementation:
        1. Quantize weight to int8 (per-channel)
        2. Dequantize back to bf16
        3. Call F.linear with dequantized weight
        
        Future: Replace with custom int8 GEMM kernel.
        """
        _ = quant_kind, kwargs
        
        # Quantize weight
        quantized_weight, scales = self.quantize_weight_for_kernel(weight, device=x.device)
        
        # Dequantize for reference implementation
        dequantized_weight = self.dequantize(quantized_weight, scales)
        
        # Compute linear output
        return F.linear(x, dequantized_weight, bias)

