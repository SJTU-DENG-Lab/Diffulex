"""
W4A16 Linear quantization strategy (int4 weight + bf16 activation), TileLang-free.

vLLM-aligned behavior:
- vLLM 在 sm89（如 4090）上并没有“在线 int4 -> 快 GEMM”的通用路径；
  真正的 int4 加速通常依赖 GPTQ/AWQ 的 marlin/cutlass 以及对应的离线权重格式。
- 为避免“看起来是 int4 但实际在跑 bf16 GEMM”，默认禁止静默走 `F.linear` 慢路径。

如需临时允许 correctness-first 慢 fallback，可设置环境变量：
  `DIFFULEX_ALLOW_SLOW_QUANT_FALLBACK=1`
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from diffulex.utils.quantization.registry import register_linear_strategy
from diffulex.utils.quantization.strategy import LinearQuantizationStrategy


@register_linear_strategy(weight_dtype="int4", act_dtype="bf16")
def _build_linear_int4_w4a16() -> LinearQuantizationStrategy:
    return LinearInt4W4A16Strategy()


class LinearInt4W4A16Strategy(LinearQuantizationStrategy):
    def __init__(self) -> None:
        super().__init__()
        # Cache: id(weight) -> (packed_int8 [N, ceil(K/2)], scales_fp32 [N])
        self._weight_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def name(self) -> str:
        return "linear_int4_w4a16"

    @property
    def linear_weight_format(self) -> str:
        return "int4"

    @property
    def linear_act_format(self) -> str:
        return "bf16"

    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        return torch.int8, 1

    def get_scale_shape(self, original_shape: tuple[int, ...], **kwargs: Any) -> tuple[int, ...]:
        _ = kwargs
        if len(original_shape) != 2:
            raise ValueError(f"Expected 2D weight [N,K], got {original_shape}")
        return (original_shape[0],)

    @staticmethod
    def _pack_int4_to_int8(int4_tensor: torch.Tensor) -> torch.Tensor:
        # int4_tensor: int8 [N,K] values in [-8,7]
        n, k = int4_tensor.shape
        t = int4_tensor.clamp(-8, 7).to(torch.int16)
        u = (t + 8).to(torch.uint8)  # [0,15]
        if k % 2 != 0:
            u = torch.cat([u, torch.full((n, 1), 8, device=u.device, dtype=torch.uint8)], dim=1)
            k = k + 1
        u2 = u.view(n, k // 2, 2)
        packed = (u2[:, :, 0] | (u2[:, :, 1] << 4)).to(torch.int8)
        return packed.contiguous()

    @staticmethod
    def _unpack_int8_to_int4(packed: torch.Tensor, *, original_k: int) -> torch.Tensor:
        # packed: int8 [N, ceil(K/2)] (two nibbles per byte)
        p = packed.view(torch.uint8)
        low = (p & 0x0F).to(torch.int16) - 8
        high = ((p >> 4) & 0x0F).to(torch.int16) - 8
        n, pk = packed.shape
        out = torch.empty((n, pk * 2), device=packed.device, dtype=torch.int16)
        out[:, 0::2] = low
        out[:, 1::2] = high
        return out[:, :original_k].to(torch.int8).contiguous()

    def quantize(self, tensor: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, Any]:
        _ = kwargs
        if tensor.dim() != 2:
            raise ValueError(f"Expected 2D weight [N,K], got shape={tuple(tensor.shape)}")
        w = tensor.to(torch.bfloat16)
        abs_max = w.abs().amax(dim=-1, keepdim=True)  # [N,1]
        scales = (abs_max.clamp(min=1e-8) / 7.0).to(torch.float32).squeeze(-1)  # [N]
        q = torch.round(w.to(torch.float32) / scales.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
        packed = self._pack_int4_to_int8(q)
        return packed, {"scales": scales}

    def quantize_weight_for_kernel(
        self,
        weight: torch.Tensor,
        *,
        device: torch.device | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Quantize+pack bf16 weight for kernel consumption.

        Returns:
            (packed_int8 [N, ceil(K/2)], scales_fp32 [N])
        """
        packed, meta = self.quantize(weight)
        if device is not None:
            packed = packed.to(device=device)
            meta["scales"] = meta["scales"].to(device=device)
        return packed, meta["scales"]

    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs: Any) -> torch.Tensor:
        original_k = int(kwargs.get("original_in_features", 0))
        if original_k <= 0:
            raise ValueError("original_in_features is required to dequantize int4 weights")
        scales = scale_or_metadata.get("scales") if isinstance(scale_or_metadata, dict) else scale_or_metadata
        if scales is None:
            raise ValueError("scales required for dequantization")
        q = self._unpack_int8_to_int4(quantized, original_k=original_k).to(torch.float32)
        w = q * scales.to(torch.float32).unsqueeze(-1)
        return w.to(torch.bfloat16)

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
        original_k = int(kwargs.get("original_in_features", x.shape[-1]))
        quant_scales = kwargs.get("quant_scales", None)

        if weight is not None and weight.dtype == torch.int8 and quant_scales is not None:
            packed = weight.to(device=x.device)
            scales = quant_scales.to(device=x.device, dtype=torch.float32)
        else:
            wid = id(weight)
            cached = self._weight_cache.get(wid)
            if cached is None or cached[0].device != x.device:
                packed, meta = self.quantize(weight)
                packed = packed.to(device=x.device)
                scales = meta["scales"].to(device=x.device, dtype=torch.float32)
                self._weight_cache[wid] = (packed, scales)
            else:
                packed, scales = cached

        # Slow fallback (explicitly opted-in).
        w_deq = self.dequantize(packed, {"scales": scales}, original_in_features=original_k)
        return F.linear(x, w_deq, bias)

