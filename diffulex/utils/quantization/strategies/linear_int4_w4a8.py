"""
W4A8 Linear quantization strategy (int4 weight + int8 activation), TileLang-free.

vLLM-aligned behavior:
- vLLM 的 CUTLASS W4A8 kernel 需要 sm90（Hopper）；在 sm89（如 4090）上不可用。
- 为避免静默退化到 bf16 GEMM，默认禁止 `F.linear` 慢 fallback。

如需临时允许 correctness-first 慢 fallback，可设置：
  `DIFFULEX_ALLOW_SLOW_QUANT_FALLBACK=1`
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from diffulex.utils.quantization.registry import register_linear_strategy
from diffulex.utils.quantization.strategy import LinearQuantizationStrategy

from .linear_int4_w4a16 import LinearInt4W4A16Strategy


@register_linear_strategy(weight_dtype="int4", act_dtype="int8")
def _build_linear_int4_w4a8() -> LinearQuantizationStrategy:
    return LinearInt4W4A8Strategy()


class LinearInt4W4A8Strategy(LinearQuantizationStrategy):
    def __init__(self) -> None:
        super().__init__()
        self._w4a16 = LinearInt4W4A16Strategy()

    @property
    def name(self) -> str:
        return "linear_int4_w4a8"

    @property
    def linear_weight_format(self) -> str:
        return "int4"

    @property
    def linear_act_format(self) -> str:
        return "int8"

    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        return torch.int8, 1

    def get_scale_shape(self, original_shape: tuple[int, ...], **kwargs: Any) -> tuple[int, ...]:
        return self._w4a16.get_scale_shape(original_shape, **kwargs)

    def quantize(self, tensor: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, Any]:
        return self._w4a16.quantize(tensor, **kwargs)

    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs: Any) -> torch.Tensor:
        return self._w4a16.dequantize(quantized, scale_or_metadata, **kwargs)

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
        if not bool(int(__import__("os").environ.get("DIFFULEX_ALLOW_SLOW_QUANT_FALLBACK", "0"))):
            raise RuntimeError(
                "当前平台/配置下 `int4` 在线量化没有可用的 vLLM 快 kernel（例如 4090/sm89 无 CUTLASS W4A8）。"
                "为避免静默退化到 bf16 GEMM，已禁止 `F.linear` 慢 fallback。"
                "请改用 `gptq/awq`（vLLM 标准打包格式）或设置 DIFFULEX_ALLOW_SLOW_QUANT_FALLBACK=1 临时开启。"
            )
        # Correctness-first: reuse W4A16 implementation.
        return self._w4a16.linear_forward(x, weight, bias, quant_kind="other", **kwargs)

