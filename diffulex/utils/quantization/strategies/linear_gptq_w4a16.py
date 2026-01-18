"""
GPTQ W4A16 Linear quantization strategy (vLLM standard format).

- Weight format: vLLM GPTQ (packed int32 qweight/qzeros + fp16 scales)
- Activation: bf16 (no activation quantization)
- Forward: vLLM custom op `gptq_gemm`

Design notes:
- Diffulex follows vLLM's fast path: run `gptq_shuffle` once (handled by
  `LinearBase._maybe_prepare_offline_gptq`) and then call `gptq_gemm` with
  `use_exllama=True`.
- No TileLang dependency.
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


@register_linear_strategy(weight_dtype="gptq", act_dtype="bf16")
def _build_linear_gptq_w4a16() -> LinearQuantizationStrategy:
    return LinearGPTQW4A16Strategy()


class LinearGPTQW4A16Strategy(LinearQuantizationStrategy):
    @property
    def name(self) -> str:
        return "linear_gptq_w4a16"

    @property
    def linear_weight_format(self) -> str:
        return "gptq"

    @property
    def linear_act_format(self) -> str:
        return "bf16"

    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        # vLLM GPTQ stores packed weights in int32.
        return torch.int32, 4

    def get_scale_shape(self, original_shape: tuple[int, ...], **kwargs) -> tuple[int, ...]:
        # vLLM GPTQ scales: [K/group, N], where Linear weight is (N, K).
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
        # Offline GPTQ is handled by `diffulex.utils.quantization.quantize_model`.
        return tensor, {}

    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs) -> torch.Tensor:
        if quantized.is_floating_point():
            return quantized
        raise NotImplementedError(
            "GPTQ dequantize is not implemented in Diffulex. "
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
                "vLLM is required for GPTQ W4A16 (missing `vllm._custom_ops`). "
                "Please install/build vLLM with CUDA ops."
            )

        qweight = kwargs.get("gptq_qweight", None)
        qzeros = kwargs.get("gptq_qzeros", None)
        scales = kwargs.get("gptq_scales", None)
        g_idx = kwargs.get("gptq_g_idx", None)

        if qweight is None or qzeros is None or scales is None:
            return F.linear(x, weight, bias)

        use_v2_format = bool(kwargs.get("gptq_use_v2_format", False))

        # Infer weight_bits from packed shapes to support GPTQ W2/W4/W8.
        # qzeros: [K/group, N/pack_factor] and qweight: [K/pack_factor, N]
        if qzeros.shape[1] <= 0 or qweight.shape[1] % int(qzeros.shape[1]) != 0:
            raise RuntimeError(
                f"Invalid GPTQ packed shapes: qweight.shape={tuple(qweight.shape)}, "
                f"qzeros.shape={tuple(qzeros.shape)}"
            )
        pack_factor = int(qweight.shape[1]) // int(qzeros.shape[1])
        if 32 % pack_factor != 0:
            raise RuntimeError(
                f"Unsupported GPTQ pack_factor={pack_factor} (requires 32%pack_factor==0). "
                f"qweight.shape={tuple(qweight.shape)}, qzeros.shape={tuple(qzeros.shape)}"
            )
        weight_bits = 32 // pack_factor

        # vLLM GPTQ kernels expect FP16 activations.
        x_in = x.to(dtype=torch.float16) if x.dtype != torch.float16 else x
        qweight = qweight.to(device=x.device, dtype=torch.int32)
        qzeros = qzeros.to(device=x.device, dtype=torch.int32)
        scales = scales.to(device=x.device, dtype=torch.float16)

        if g_idx is None or (isinstance(g_idx, torch.Tensor) and g_idx.numel() == 0):
            g_idx_t = torch.empty((0,), device=x.device, dtype=torch.int)
        else:
            g_idx_t = g_idx.to(device=x.device, dtype=torch.int)

        out_shape = x.shape[:-1] + (qweight.shape[-1],)
        reshaped_x = x_in.reshape(-1, x_in.shape[-1])

        output = ops.gptq_gemm(
            reshaped_x,
            qweight,
            qzeros,
            scales,
            g_idx_t,
            True,  # use_exllama (vLLM shuffles weights into exllama-friendly layout)
            use_v2_format,
            weight_bits,
        )
        if bias is not None:
            output.add_(bias.to(dtype=output.dtype))
        output = output.reshape(out_shape)
        # Keep output dtype consistent with input activations for downstream layers.
        return output.to(dtype=x.dtype) if output.dtype != x.dtype else output

