"""
FP8 W8A8 Linear quantization strategy (FP8 weight + FP8 activation), TileLang-free.

vLLM-aligned implementation:
- Weight quantization: `vllm._custom_ops.scaled_fp8_quant` (per-tensor scale).
- Activation quantization + GEMM: vLLM `Fp8LinearOp` (CUTLASS scaled_mm when available).
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from diffulex.utils.quantization.registry import register_linear_strategy
from diffulex.utils.quantization.strategy import LinearQuantizationStrategy


def _require_fp8_linear_op():
    try:
        from vllm.model_executor.layers.quantization.utils.w8a8_utils import (  # type: ignore
            Fp8LinearOp,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError("FP8 需要 vLLM（Fp8LinearOp / _custom_ops）。") from e
    return Fp8LinearOp


@register_linear_strategy(weight_dtype="fp8_e4m3", act_dtype="fp8_e4m3")
def _build_linear_fp8_e4m3_w8a8() -> LinearQuantizationStrategy:
    return LinearFP8W8A8Strategy("fp8_e4m3", "fp8_e4m3")


@register_linear_strategy(weight_dtype="fp8_e5m2", act_dtype="fp8_e5m2")
def _build_linear_fp8_e5m2_w8a8() -> LinearQuantizationStrategy:
    return LinearFP8W8A8Strategy("fp8_e5m2", "fp8_e5m2")


class LinearFP8W8A8Strategy(LinearQuantizationStrategy):
    def __init__(self, weight_dtype: str = "fp8_e4m3", act_dtype: str = "fp8_e4m3") -> None:
        super().__init__()
        self.weight_dtype_str = weight_dtype
        self.act_dtype_str = act_dtype
        # Cache: id(weight) -> (q_fp8_KN [K,N], scale_fp32 [1])
        self._weight_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        Fp8LinearOp = _require_fp8_linear_op()
        self._fp8_linear = Fp8LinearOp(act_quant_static=False)

    @property
    def name(self) -> str:
        return f"linear_fp8_{self.weight_dtype_str}_w8a8"

    @property
    def linear_weight_format(self) -> str:
        return self.weight_dtype_str

    @property
    def linear_act_format(self) -> str:
        return self.act_dtype_str

    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        return torch.uint8, 1

    def get_scale_shape(self, original_shape: tuple[int, ...], **kwargs: Any) -> tuple[int, ...]:
        _ = kwargs
        if len(original_shape) != 2:
            raise ValueError(f"Expected 2D weight [N,K], got {original_shape}")
        return (1,)

    def quantize(self, tensor: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, Any]:
        _ = kwargs
        if tensor.dim() != 2:
            raise ValueError(f"Expected 2D weight [N,K], got shape={tuple(tensor.shape)}")
        from vllm import _custom_ops as ops  # type: ignore
        from vllm.platforms import current_platform  # type: ignore

        q_fp8, scale = ops.scaled_fp8_quant(tensor.to(torch.float32).contiguous(), scale=None)
        q_kn_fp8 = q_fp8.t()  # [K,N], stride(0)==1
        scale = scale.to(torch.float32).reshape(1).contiguous()
        return q_kn_fp8, {"scales": scale, "fp8_dtype": current_platform.fp8_dtype()}

    def quantize_weight_for_kernel(
        self,
        weight: torch.Tensor,
        *,
        device: torch.device | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, Any]:
        q_fp8, meta = self.quantize(weight)
        if device is not None:
            q_fp8 = q_fp8.to(device=device)
            meta["scales"] = meta["scales"].to(device=device)
        return q_fp8, meta["scales"]

    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs: Any) -> torch.Tensor:
        _ = kwargs
        raise RuntimeError("FP8 不提供 dequantize 路径（避免走慢的反量化 + F.linear）。")

    def linear_forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        quant_kind: str,
        **kwargs: Any,
    ) -> torch.Tensor:
        _ = quant_kind
        wid = id(weight)
        cached = self._weight_cache.get(wid)
        if cached is None or cached[0].device != x.device:
            q_fp8, meta = self.quantize(weight)
            q_fp8 = q_fp8.to(device=x.device)
            w_scale = meta["scales"].to(device=x.device, dtype=torch.float32).reshape(1)
            self._weight_cache[wid] = (q_fp8, w_scale)
        else:
            q_fp8, w_scale = cached

        q_kn = q_fp8

        return self._fp8_linear.apply(
            input=x,
            weight=q_kn,
            weight_scale=w_scale,
            out_dtype=x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16,
            input_scale=None,
            bias=bias,
        )

