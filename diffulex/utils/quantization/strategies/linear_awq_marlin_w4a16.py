"""
AWQ Marlin (W4, A16) Linear strategy using vLLM Marlin CUDA kernels.

- Input activations: bf16 (cast to fp16 for vLLM marlin kernel)
- Weights: offline AWQ vLLM standard format (qweight/qzeros/scales)
- One-time repack/permutation is performed by Diffulex `LinearBase` and passed in via kwargs:
  - awq_marlin_qweight / awq_marlin_scales / awq_marlin_zp
  - awq_marlin_workspace
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from diffulex.utils.quantization.registry import register_linear_strategy
from diffulex.utils.quantization.strategy import LinearQuantizationStrategy

try:
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # type: ignore
        apply_awq_marlin_linear,
        marlin_make_empty_g_idx,
        marlin_permute_bias,
    )
    from vllm.scalar_type import scalar_types  # type: ignore
except Exception:  # pragma: no cover
    apply_awq_marlin_linear = None  # type: ignore
    marlin_make_empty_g_idx = None  # type: ignore
    marlin_permute_bias = None  # type: ignore
    scalar_types = None  # type: ignore


@register_linear_strategy(weight_dtype="awq_marlin", act_dtype="bf16")
def _build_linear_awq_marlin_w4a16() -> LinearQuantizationStrategy:
    return LinearAWQMarlinW4A16Strategy()


class LinearAWQMarlinW4A16Strategy(LinearQuantizationStrategy):
    @property
    def name(self) -> str:
        return "linear_awq_marlin_w4a16"

    @property
    def linear_weight_format(self) -> str:
        return "awq_marlin"

    @property
    def linear_act_format(self) -> str:
        return "bf16"

    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        return torch.int32, 4

    def get_scale_shape(self, original_shape: tuple[int, ...], **kwargs) -> tuple[int, ...]:
        # Same as AWQ: [K/group, N]
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
        return tensor, {}

    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs) -> torch.Tensor:
        if quantized.is_floating_point():
            return quantized
        raise NotImplementedError("AWQ Marlin 不提供 Python dequantize；请使用 vLLM Marlin CUDA kernel。")

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
        if apply_awq_marlin_linear is None or scalar_types is None:
            raise RuntimeError("awq_marlin 需要 vLLM (marlin_utils + scalar_types)；当前环境不可用。")

        qweight = kwargs.get("awq_marlin_qweight", None)
        scales = kwargs.get("awq_marlin_scales", None)
        zp = kwargs.get("awq_marlin_zp", None)
        workspace = kwargs.get("awq_marlin_workspace", None)
        in_features = int(kwargs.get("in_features", 0))
        out_features = int(kwargs.get("out_features", 0))

        if any(t is None for t in (qweight, scales, zp, workspace)) or in_features <= 0 or out_features <= 0:
            raise RuntimeError("awq_marlin: missing prepared marlin tensors (qweight/scales/zp/workspace).")

        # vLLM marlin kernels expect FP16 activations.
        x_in = x.to(dtype=torch.float16) if x.dtype != torch.float16 else x

        # AWQ marlin does not use g_idx.
        empty = marlin_make_empty_g_idx(x.device) if marlin_make_empty_g_idx is not None else torch.empty((0,), device=x.device, dtype=torch.int32)

        marlin_bias = None
        if bias is not None:
            marlin_bias = marlin_permute_bias(bias) if marlin_permute_bias is not None else bias

        out = apply_awq_marlin_linear(
            input=x_in,
            weight=qweight,
            weight_scale=scales,
            weight_zp=zp,
            g_idx=empty,
            g_idx_sort_indices=empty,
            workspace=workspace,
            quant_type=scalar_types.uint4,
            output_size_per_partition=out_features,
            input_size_per_partition=in_features,
            bias=marlin_bias,
            input_dtype=None,
        )
        return out.to(dtype=x.dtype) if out.dtype != x.dtype else out

