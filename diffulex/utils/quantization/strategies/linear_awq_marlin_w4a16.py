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
        should_use_atomic_add_reduce,
        marlin_permute_bias,
    )
    from vllm.scalar_type import scalar_types  # type: ignore
except Exception:  # pragma: no cover
    apply_awq_marlin_linear = None  # type: ignore
    marlin_make_empty_g_idx = None  # type: ignore
    should_use_atomic_add_reduce = None  # type: ignore
    marlin_permute_bias = None  # type: ignore
    scalar_types = None  # type: ignore


@register_linear_strategy(weight_dtype="awq_marlin", act_dtype="bf16")
def _build_linear_awq_marlin_w4a16() -> LinearQuantizationStrategy:
    return LinearAWQMarlinW4A16Strategy()


class LinearAWQMarlinW4A16Strategy(LinearQuantizationStrategy):
    def __init__(self) -> None:
        super().__init__()
        self._available: bool = bool(apply_awq_marlin_linear is not None and scalar_types is not None)
        self._empty_cache: dict[int, torch.Tensor] = {}
        self._bias_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._atomic_add_cache: dict[tuple[int, int, int, int, int], bool] = {}

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
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        *,
        quant_kind: str,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zp: torch.Tensor,
        workspace: Optional[torch.Tensor] = None,
        in_features: int = 0,
        out_features: int = 0,
        group_size: int = 128,
        tp_dim: Optional[int] = None,
    ) -> torch.Tensor:
        _ = quant_kind, weight, group_size, tp_dim
        if not self._available or workspace is None:
            raise RuntimeError("awq_marlin 需要 vLLM (marlin_utils + scalar_types)；当前环境不可用。")
        if in_features <= 0 or out_features <= 0:
            raise RuntimeError("awq_marlin: missing in_features/out_features.")

        device = x.device
        dev_key = int(device.index) if device.type == "cuda" and device.index is not None else -1

        # AWQ marlin does not use g_idx/perm; pass empty tensors (cached).
        empty = self._empty_cache.get(dev_key)
        if empty is None:
            empty = marlin_make_empty_g_idx(device) if marlin_make_empty_g_idx is not None else torch.empty((0,), device=device, dtype=torch.int32)
            self._empty_cache[dev_key] = empty

        # Cache permuted bias.
        marlin_bias = None
        if bias is not None:
            bkey = (dev_key, int(bias.data_ptr()))
            marlin_bias = self._bias_cache.get(bkey)
            if marlin_bias is None:
                marlin_bias = marlin_permute_bias(bias) if marlin_permute_bias is not None else bias
                self._bias_cache[bkey] = marlin_bias

        reshaped_x = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (int(out_features),)
        m = int(reshaped_x.shape[0])
        n = int(out_features)
        k = int(reshaped_x.shape[1])
        dtype_id = 1 if reshaped_x.dtype == torch.bfloat16 else (2 if reshaped_x.dtype == torch.float16 else 0)
        use_atomic_add = False
        if should_use_atomic_add_reduce is not None:
            akey = (dev_key, dtype_id, m, n, k)
            cached = self._atomic_add_cache.get(akey)
            if cached is None:
                cached = bool(
                    should_use_atomic_add_reduce(
                        m=m, n=n, k=k, device=device, dtype=reshaped_x.dtype
                    )
                )
                self._atomic_add_cache[akey] = cached
            use_atomic_add = cached

        out = torch.ops._C.gptq_marlin_gemm(
            reshaped_x,
            None,
            qweight,
            marlin_bias,
            scales,
            None,
            None,
            zp,
            empty,
            empty,
            workspace,
            scalar_types.uint4.id,
            m,
            n,
            k,
            True,  # is_k_full
            use_atomic_add,
            True,  # use_fp32_reduce
            False,  # is_zp_float
        )
        out = out.reshape(out_shape)
        return out.to(dtype=x.dtype) if out.dtype != x.dtype else out

