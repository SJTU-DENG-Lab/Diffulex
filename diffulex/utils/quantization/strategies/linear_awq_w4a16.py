"""
AWQ W4A16 Linear quantization strategy (vLLM standard format).

- Weight format: vLLM AWQ (packed int32 qweight/qzeros + fp16 scales)
- Activation: bf16 (no activation quantization)
- Forward: vLLM custom op `awq_gemm` (with the same heuristic as vLLM)

No TileLang dependency.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from diffulex.utils.quantization.registry import register_linear_strategy
from diffulex.utils.quantization.strategy import LinearQuantizationStrategy

try:
    from vllm import _custom_ops as ops  # type: ignore
except Exception:  # pragma: no cover
    ops = None  # type: ignore


@register_linear_strategy(weight_dtype="awq", act_dtype="bf16")
def _build_linear_awq_w4a16() -> LinearQuantizationStrategy:
    return LinearAWQW4A16Strategy()


class LinearAWQW4A16Strategy(LinearQuantizationStrategy):
    @property
    def name(self) -> str:
        return "linear_awq_w4a16"

    @property
    def linear_weight_format(self) -> str:
        return "awq"

    @property
    def linear_act_format(self) -> str:
        return "bf16"

    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        # vLLM AWQ stores packed weights in int32.
        return torch.int32, 4

    def get_scale_shape(self, original_shape: tuple[int, ...], **kwargs) -> tuple[int, ...]:
        # vLLM AWQ scales: [K/group, N], where Linear weight is (N, K).
        if len(original_shape) != 2:
            raise ValueError(f"Expected 2D weight shape, got {original_shape}")
        out_features, in_features = original_shape
        group_size = int(kwargs.get("group_size", 128))
        group_size = in_features if group_size == -1 else group_size
        if group_size <= 0 or in_features % group_size != 0:
            raise ValueError(f"Invalid group_size={group_size} for in_features={in_features}")
        num_groups = in_features // group_size
        return (num_groups, out_features)

    def quantize(self, tensor: torch.Tensor, **kwargs) -> tuple[torch.Tensor, Any]:
        # Offline AWQ is handled by `diffulex.utils.quantization.quantize_model`.
        return tensor, {}

    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs) -> torch.Tensor:
        if quantized.is_floating_point():
            return quantized
        raise NotImplementedError(
            "AWQ dequantize is not implemented in Diffulex. "
            "Use vLLM kernels via linear_forward."
        )

    def linear_forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        quant_kind: str,
        **kwargs: Any,
    ) -> torch.Tensor:
        _ = quant_kind, weight
        if ops is None:
            raise RuntimeError(
                "vLLM is required for AWQ W4A16 (missing `vllm._custom_ops`). "
                "Please install/build vLLM with CUDA ops."
            )

        qweight = kwargs.get("awq_qweight", None)
        qzeros = kwargs.get("awq_qzeros", None)
        scales = kwargs.get("awq_scales", None)

        if qweight is None or qzeros is None or scales is None:
            return F.linear(x, weight, bias)

        # Infer pack_factor from packed shapes to avoid hard-coding 4-bit.
        # AWQ: qweight [K, N/pack], scales [K/group, N]
        if scales.ndim != 2 or scales.shape[1] <= 0:
            raise RuntimeError(f"Invalid AWQ scales shape: {tuple(scales.shape)}")
        if qweight.shape[1] <= 0 or int(scales.shape[1]) % int(qweight.shape[1]) != 0:
            raise RuntimeError(
                f"Invalid AWQ packed shapes: qweight.shape={tuple(qweight.shape)}, "
                f"scales.shape={tuple(scales.shape)}"
            )
        pack_factor = int(scales.shape[1]) // int(qweight.shape[1])
        # vLLM AWQ kernels expect FP16 activations.
        x_in = x.to(dtype=torch.float16) if x.dtype != torch.float16 else x
        qweight = qweight.to(device=x.device, dtype=torch.int32)
        qzeros = qzeros.to(device=x.device, dtype=torch.int32)
        scales = scales.to(device=x.device, dtype=torch.float16)

        out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
        reshaped_x = x_in.reshape(-1, x_in.shape[-1])

        # Always use awq_gemm to avoid large temporary dequantized weight allocations.
        out = ops.awq_gemm(reshaped_x, qweight, scales, qzeros, pack_factor)

        if bias is not None:
            out.add_(bias.to(dtype=out.dtype))
        out = out.reshape(out_shape)
        return out.to(dtype=x.dtype) if out.dtype != x.dtype else out

