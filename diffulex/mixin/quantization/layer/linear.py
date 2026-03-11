"""Quantization mixin for Linear layer.

Call init_quantization() from the layer __init__ to register quant buffers and Python meta.
Provides strategy lookup, forward plan build, and all quant-related helpers.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffulex.quantization.context import get_linear_strategy
from diffulex.quantization.strategy import LinearQuantizationStrategy
from diffulex.quantization.linear_plan_builder import build_forward_plan
from diffulex.quantization.linear_plans import ForwardPlanBase

from .offline_prep import (
    maybe_prepare_offline_gptq,
    maybe_prepare_offline_gptq_marlin,
    maybe_prepare_offline_awq_marlin,
)
from .online_quant import (
    set_quantized_weight,
    maybe_promote_weight_to_quantized_at_runtime,
    maybe_quantize_loaded_weight_param,
)


class LinearQuantizationMixin:
    """Mixin for Linear layer quantization: strategy, plan build, and weight/offline quant helpers."""

    def init_quantization(self, quant_kind: str = "other") -> None:
        """Register quant-related buffers, Python meta, and forward plan cache."""
        self.quant_kind = (quant_kind or "other").strip().lower() or "other"
        self._forward_out_features: int = int(self.output_size)
        
        # Quantized weight storage (W8A16 etc.). Empty by default.
        self.register_buffer("quant_weight_int8", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("quant_scales", torch.empty(0, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("quant_scales_1xn", torch.empty(0, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("_weight_is_quantized", torch.tensor(False, dtype=torch.bool), persistent=False)

        # GPTQ/AWQ offline (vLLM-format)
        self.register_buffer("gptq_qweight", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("gptq_qzeros", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("gptq_scales", torch.empty(0, dtype=torch.float16), persistent=False)
        self.register_buffer("gptq_g_idx", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("awq_qweight", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("awq_qzeros", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("awq_scales", torch.empty(0, dtype=torch.float16), persistent=False)
        
        self.register_buffer("_offline_quant_format", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("_offline_quant_bits", torch.tensor(0, dtype=torch.int32), persistent=False)
        self.register_buffer("_offline_quant_group_size", torch.tensor(128, dtype=torch.int32), persistent=False)
        self.register_buffer("_offline_quant_out_features", torch.tensor(0, dtype=torch.int32), persistent=False)
        self.register_buffer("_offline_quant_in_features", torch.tensor(0, dtype=torch.int32), persistent=False)
        self.register_buffer("_gptq_is_shuffled", torch.tensor(False, dtype=torch.bool), persistent=False)

        # vLLM Marlin one-time repack cache
        self.register_buffer("_gptq_marlin_is_prepared", torch.tensor(False, dtype=torch.bool), persistent=False)
        self.register_buffer("gptq_marlin_qweight", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("gptq_marlin_scales", torch.empty(0, dtype=torch.float16), persistent=False)
        self.register_buffer("gptq_marlin_zp", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("gptq_marlin_g_idx", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("gptq_marlin_g_idx_sort_indices", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("gptq_marlin_workspace", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("_awq_marlin_is_prepared", torch.tensor(False, dtype=torch.bool), persistent=False)
        self.register_buffer("awq_marlin_qweight", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("awq_marlin_scales", torch.empty(0, dtype=torch.float16), persistent=False)
        self.register_buffer("awq_marlin_zp", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("awq_marlin_workspace", torch.empty(0, dtype=torch.int32), persistent=False)

        # Python-side meta (CUDA graph friendly; avoid .item() on hot paths)
        self._weight_is_quantized_py: bool = False
        self._offline_quant_format_py: int = 0
        self._offline_quant_bits_py: int = 0
        self._offline_quant_group_size_py: int = 128
        self._offline_quant_out_features_py: int = 0
        self._offline_quant_in_features_py: int = 0
        self._gptq_is_shuffled_py: bool = False
        self._gptq_marlin_is_prepared_py: bool = False
        self._awq_marlin_is_prepared_py: bool = False

        # Forward plan cache
        self._forward_plan_enabled: bool = False
        self._forward_plan: Optional[ForwardPlanBase] = None

    def _forward_base(self, x: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        """Unified forward dispatcher for bf16 / online quant / offline GPTQ/AWQ."""
        if getattr(self, "_forward_plan_enabled", False):
            plan = getattr(self, "_forward_plan", None)
            if plan is None:
                self.build_forward_plan_for_static(x, bias)
                plan = getattr(self, "_forward_plan", None)
            if plan is not None:
                sig = plan.sig
                dev = x.device
                dev_idx = self._device_index(dev)
                if (
                    sig.device_type == dev.type
                    and sig.device_index == dev_idx
                    and sig.x_dtype == x.dtype
                    and sig.x_shape == tuple(int(v) for v in x.shape)
                    and sig.has_bias == (bias is not None)
                ):
                    return plan(x)
                self.build_forward_plan_for_static(x, bias)
                plan = getattr(self, "_forward_plan", None)
                if plan is not None:
                    sig = plan.sig
                    if (
                        sig.device_type == dev.type
                        and sig.device_index == dev_idx
                        and sig.x_dtype == x.dtype
                        and sig.x_shape == tuple(int(v) for v in x.shape)
                        and sig.has_bias == (bias is not None)
                    ):
                        return plan(x)

        strategy = self._get_linear_strategy()
        maybe_promote_weight_to_quantized_at_runtime(self, x, strategy)

        if self.has_offline_quantized_weight():
            return self._forward_offline(x, bias, strategy)

        if self.has_quantized_weight():
            return self._forward_quantized(x, bias, strategy)

        if strategy is None:
            weight = getattr(self, "weight", None)
            if weight is None:
                raise RuntimeError("No strategy is configured and bf16 weight is missing.")
            return F.linear(x, weight, bias)

        weight = getattr(self, "weight", None)
        kwargs = self._maybe_int4_original_in_features_kwargs(strategy, x)
        if kwargs:
            return strategy.linear_forward(x, weight, bias, quant_kind=self.quant_kind, **kwargs)
        return strategy.linear_forward(x, weight, bias, quant_kind=self.quant_kind)

    def _forward_offline(self, x: torch.Tensor, bias: Optional[torch.Tensor], strategy) -> torch.Tensor:
        """Forward for offline quantized weights."""
        if strategy is None:
            raise RuntimeError("Offline quantized weight is present but no linear strategy is configured.")
        weight_format = getattr(strategy, "linear_weight_format", None)
        out_features, in_features, group_size = self._offline_meta()

        if weight_format == "gptq":
            maybe_prepare_offline_gptq(self, x)
            bits = self._infer_gptq_weight_bits(in_features=in_features)
            return strategy.linear_forward(
                x, None, bias, quant_kind=self.quant_kind,
                gptq_qweight=self.gptq_qweight, gptq_qzeros=self.gptq_qzeros,
                gptq_scales=self.gptq_scales, gptq_g_idx=self.gptq_g_idx,
                weight_bits=bits, use_v2_format=False,
                out_features=out_features, in_features=in_features, group_size=group_size,
            )

        if weight_format == "awq":
            bits = int(self._offline_quant_bits_py) if int(self._offline_quant_bits_py) > 0 else 4
            pack_factor = 32 // max(1, bits)
            return strategy.linear_forward(
                x, None, bias, quant_kind=self.quant_kind,
                awq_qweight=self.awq_qweight, awq_qzeros=self.awq_qzeros,
                awq_scales=self.awq_scales, pack_factor=pack_factor,
                out_features=out_features, in_features=in_features, group_size=group_size,
            )

        if weight_format == "gptq_marlin":
            maybe_prepare_offline_gptq_marlin(self, x)
            bits = self._infer_gptq_weight_bits(in_features=in_features)
            return strategy.linear_forward(
                x, None, bias, quant_kind=self.quant_kind,
                qweight=self.gptq_marlin_qweight, scales=self.gptq_marlin_scales,
                zp=self.gptq_marlin_zp, g_idx=self.gptq_marlin_g_idx,
                g_idx_sort_indices=self.gptq_marlin_g_idx_sort_indices,
                workspace=self.gptq_marlin_workspace,
                in_features=in_features, out_features=out_features,
                group_size=group_size, weight_bits=bits, tp_dim=self.tp_dim,
            )

        if weight_format == "awq_marlin":
            maybe_prepare_offline_awq_marlin(self, x)
            return strategy.linear_forward(
                x, None, bias, quant_kind=self.quant_kind,
                qweight=self.awq_marlin_qweight, scales=self.awq_marlin_scales,
                zp=self.awq_marlin_zp, workspace=self.awq_marlin_workspace,
                in_features=in_features, out_features=out_features,
                group_size=group_size, tp_dim=self.tp_dim,
            )

        kwargs = self._build_offline_forward_kwargs(x, strategy)
        return strategy.linear_forward(x, None, bias, quant_kind=self.quant_kind, **kwargs)

    def _forward_quantized(self, x: torch.Tensor, bias: Optional[torch.Tensor], strategy) -> torch.Tensor:
        """Forward for online quantized weights."""
        if strategy is None:
            raise RuntimeError("Quantized weight is present but no linear strategy is configured.")
        extra_kwargs = self._maybe_int4_original_in_features_kwargs(strategy, x)
        
        if getattr(strategy, "name", "") in ("linear_int8_w8a16", "linear_int8_w8a8"):
            scales = self.quant_scales_1xn
            if extra_kwargs:
                return strategy.linear_forward(
                    x, self.quant_weight_int8, bias,
                    quant_kind=self.quant_kind, quant_scales=scales,
                    out_features=self._forward_out_features, **extra_kwargs,
                )
            return strategy.linear_forward(
                x, self.quant_weight_int8, bias,
                quant_kind=self.quant_kind, quant_scales=scales,
                out_features=self._forward_out_features,
            )

        if extra_kwargs:
            return strategy.linear_forward(
                x, self.quant_weight_int8, bias,
                quant_kind=self.quant_kind, quant_scales=self.quant_scales, **extra_kwargs,
            )
        return strategy.linear_forward(
            x, self.quant_weight_int8, bias,
            quant_kind=self.quant_kind, quant_scales=self.quant_scales,
        )

    def _get_linear_strategy(self) -> Optional[LinearQuantizationStrategy]:
        kind = getattr(self, "_quant_kind", None) or getattr(self, "quant_kind", "other")
        return get_linear_strategy(kind)

    def _invalidate_forward_plan(self) -> None:
        self._forward_plan = None

    @staticmethod
    def _device_index(device: torch.device) -> int:
        if device.type == "cuda" and device.index is not None:
            return int(device.index)
        return -1

    def enable_forward_plan(self, enabled: bool = True) -> None:
        """Enable/disable cached forward plan dispatch for this layer."""
        self._forward_plan_enabled = bool(enabled)
        if not self._forward_plan_enabled:
            self._invalidate_forward_plan()

    def build_forward_plan_for_static(self, example_x: torch.Tensor, bias: Optional[torch.Tensor]) -> None:
        """Build a cached forward plan for a fixed static decode-step shape."""
        strategy = self._get_linear_strategy()
        maybe_promote_weight_to_quantized_at_runtime(self, example_x, strategy)
        self._forward_plan = build_forward_plan(self, example_x, bias)

    def has_quantized_weight(self) -> bool:
        return self._weight_is_quantized_py and self.quant_weight_int8.numel() > 0 and self.quant_scales.numel() > 0

    def has_offline_quantized_weight(self) -> bool:
        if self._offline_quant_format_py == 1:
            return self.gptq_qweight.numel() > 0 and self.gptq_qzeros.numel() > 0 and self.gptq_scales.numel() > 0
        elif self._offline_quant_format_py == 2:
            return self.awq_qweight.numel() > 0 and self.awq_qzeros.numel() > 0 and self.awq_scales.numel() > 0
        return False

    def set_offline_quantized_weight(
        self, format: str, qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor,
        *, out_features: int, in_features: int, group_size: int = 128,
        g_idx: Optional[torch.Tensor] = None,
    ) -> None:
        """Set offline quantized weights (GPTQ or AWQ format)."""
        from .offline_prep import _infer_module_device
        
        module_device = _infer_module_device(self)
        format = format.strip().lower()
        if format not in ("gptq", "awq"):
            raise ValueError(f"Unsupported offline quant format: {format}. Supported: 'gptq', 'awq'")

        if format == "gptq":
            if int(qweight.shape[0]) <= 0 or in_features % int(qweight.shape[0]) != 0:
                raise ValueError(f"Cannot infer GPTQ pack_factor: in_features={in_features}, qweight.shape={tuple(qweight.shape)}")
            pack_factor = in_features // int(qweight.shape[0])
        else:
            if int(qweight.shape[1]) <= 0 or out_features % int(qweight.shape[1]) != 0:
                raise ValueError(f"Cannot infer AWQ pack_factor: out_features={out_features}, qweight.shape={tuple(qweight.shape)}")
            pack_factor = out_features // int(qweight.shape[1])
            
        if 32 % pack_factor != 0:
            raise ValueError(f"Unsupported pack_factor={pack_factor} for offline format={format}.")
        bits = 32 // pack_factor
        
        if format == "awq" and bits != 4:
            raise ValueError(f"AWQ currently only supports 4-bit, inferred bits={bits}")

        self._offline_quant_bits_py = int(bits)
        self._offline_quant_bits = torch.tensor(bits, dtype=torch.int32, device=module_device)

        if qweight.dtype != torch.int32:
            raise TypeError(f"qweight must be int32, got {qweight.dtype}")
        if qzeros.dtype != torch.int32:
            raise TypeError(f"qzeros must be int32, got {qzeros.dtype}")
        if scales.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(f"scales must be float16/bfloat16/float32, got {scales.dtype}")
        if scales.dtype != torch.float16:
            scales = scales.to(dtype=torch.float16)

        if qweight.device != module_device:
            qweight = qweight.to(device=module_device)
        if qzeros.device != module_device:
            qzeros = qzeros.to(device=module_device)
        if scales.device != module_device:
            scales = scales.to(device=module_device)
        if g_idx is not None and g_idx.device != module_device:
            g_idx = g_idx.to(device=module_device)

        qweight = qweight.contiguous()
        qzeros = qzeros.contiguous()
        scales = scales.contiguous()
        if g_idx is not None:
            g_idx = g_idx.contiguous()

        group_size_norm = in_features if group_size == -1 else group_size
        if group_size_norm <= 0 or (in_features % group_size_norm != 0):
            raise ValueError(f"Invalid group_size={group_size} for in_features={in_features}.")
        num_groups = in_features // group_size_norm

        if format == "gptq":
            expected_qweight_shape = (in_features // pack_factor, out_features)
            expected_qzeros_shape = (num_groups, out_features // pack_factor)
            expected_scales_shape = (num_groups, out_features)
        else:
            expected_qweight_shape = (in_features, out_features // pack_factor)
            expected_qzeros_shape = (num_groups, out_features // pack_factor)
            expected_scales_shape = (num_groups, out_features)

        if qweight.shape != expected_qweight_shape:
            raise ValueError(f"qweight shape mismatch: got {tuple(qweight.shape)}, expected {expected_qweight_shape}")
        if qzeros.shape != expected_qzeros_shape:
            raise ValueError(f"qzeros shape mismatch: got {tuple(qzeros.shape)}, expected {expected_qzeros_shape}")
        if scales.shape != expected_scales_shape:
            raise ValueError(f"scales shape mismatch: got {tuple(scales.shape)}, expected {expected_scales_shape}")

        if format == "gptq":
            self.gptq_qweight = qweight
            self.gptq_qzeros = qzeros
            self.gptq_scales = scales
            if g_idx is not None and getattr(g_idx, "numel", lambda: 1)() == 0:
                g_idx = None
            if g_idx is not None:
                if g_idx.shape != (in_features,):
                    raise ValueError(f"g_idx shape mismatch: got {g_idx.shape}, expected ({in_features},)")
                if g_idx.dtype != torch.int32:
                    g_idx = g_idx.to(dtype=torch.int32)
                self.gptq_g_idx = g_idx
            else:
                self.gptq_g_idx = torch.empty(0, dtype=torch.int32, device=module_device)
            self._offline_quant_format = torch.tensor(1, dtype=torch.int8, device=module_device)
            self._gptq_is_shuffled = torch.tensor(False, dtype=torch.bool, device=module_device)
            self._offline_quant_format_py = 1
            self._gptq_is_shuffled_py = False
        else:
            self.awq_qweight = qweight
            self.awq_qzeros = qzeros
            self.awq_scales = scales
            self.gptq_qweight = torch.empty(0, dtype=torch.int32, device=module_device)
            self.gptq_qzeros = torch.empty(0, dtype=torch.int32, device=module_device)
            self.gptq_scales = torch.empty(0, dtype=torch.float16, device=module_device)
            self.gptq_g_idx = torch.empty(0, dtype=torch.int32, device=module_device)
            self._offline_quant_format = torch.tensor(2, dtype=torch.int8, device=module_device)
            self._gptq_is_shuffled = torch.tensor(False, dtype=torch.bool, device=module_device)
            self._offline_quant_format_py = 2
            self._gptq_is_shuffled_py = False

        # Reset marlin-prep caches
        self._gptq_marlin_is_prepared = torch.tensor(False, dtype=torch.bool, device=module_device)
        self.gptq_marlin_qweight = torch.empty(0, dtype=torch.int32, device=module_device)
        self.gptq_marlin_scales = torch.empty(0, dtype=torch.float16, device=module_device)
        self.gptq_marlin_zp = torch.empty(0, dtype=torch.int32, device=module_device)
        self.gptq_marlin_g_idx = torch.empty(0, dtype=torch.int32, device=module_device)
        self.gptq_marlin_g_idx_sort_indices = torch.empty(0, dtype=torch.int32, device=module_device)
        self.gptq_marlin_workspace = torch.empty(0, dtype=torch.int32, device=module_device)
        self._awq_marlin_is_prepared = torch.tensor(False, dtype=torch.bool, device=module_device)
        self.awq_marlin_qweight = torch.empty(0, dtype=torch.int32, device=module_device)
        self.awq_marlin_scales = torch.empty(0, dtype=torch.float16, device=module_device)
        self.awq_marlin_zp = torch.empty(0, dtype=torch.int32, device=module_device)
        self.awq_marlin_workspace = torch.empty(0, dtype=torch.int32, device=module_device)

        self._offline_quant_group_size = torch.tensor(group_size, dtype=torch.int32, device=module_device)
        self._offline_quant_out_features = torch.tensor(out_features, dtype=torch.int32, device=module_device)
        self._offline_quant_in_features = torch.tensor(in_features, dtype=torch.int32, device=module_device)
        self._offline_quant_group_size_py = int(group_size)
        self._offline_quant_out_features_py = int(out_features)
        self._offline_quant_in_features_py = int(in_features)
        self._gptq_marlin_is_prepared_py = False
        self._awq_marlin_is_prepared_py = False

        if "weight" in self._parameters:
            self._parameters.pop("weight", None)
            setattr(self, "weight", None)
        self._invalidate_forward_plan()

    def _offline_meta(self) -> tuple[int, int, int]:
        return (
            int(self._offline_quant_out_features_py),
            int(self._offline_quant_in_features_py),
            int(self._offline_quant_group_size_py),
        )

    def _infer_gptq_weight_bits(self, *, in_features: int) -> int:
        bits = int(self._offline_quant_bits_py)
        if bits > 0:
            return bits
        if self.gptq_qweight.numel() == 0:
            raise RuntimeError("GPTQ bits inference failed: gptq_qweight is empty.")
        if self.gptq_qweight.shape[0] <= 0 or in_features % int(self.gptq_qweight.shape[0]) != 0:
            raise RuntimeError(f"GPTQ bits inference failed: in_features={in_features}, qweight.shape={tuple(self.gptq_qweight.shape)}")
        pack_factor = in_features // int(self.gptq_qweight.shape[0])
        if 32 % pack_factor != 0:
            raise RuntimeError(f"GPTQ bits inference failed: pack_factor={pack_factor} does not satisfy 32%pack_factor==0")
        return 32 // pack_factor

    def _maybe_int4_original_in_features_kwargs(self, strategy, x: torch.Tensor) -> Optional[dict]:
        if strategy is None or getattr(strategy, "linear_weight_format", None) != "int4":
            return None
        return {"original_in_features": x.shape[1]}

    def _build_offline_forward_kwargs(self, x: torch.Tensor, strategy) -> dict:
        if strategy is None:
            raise RuntimeError("Offline quantized weight is present but no linear strategy is configured.")
        format_val = int(self._offline_quant_format_py)
        weight_format = getattr(strategy, "linear_weight_format", None)
        out_features, in_features, group_size = self._offline_meta()
        meta = {"out_features": out_features, "in_features": in_features, "group_size": group_size}
        
        if format_val == 1:
            if weight_format == "gptq":
                maybe_prepare_offline_gptq(self, x)
                return {
                    **meta,
                    "gptq_qweight": self.gptq_qweight, "gptq_qzeros": self.gptq_qzeros,
                    "gptq_scales": self.gptq_scales, "gptq_group_size": group_size,
                    "gptq_g_idx": self.gptq_g_idx,
                }
            if weight_format == "gptq_marlin":
                maybe_prepare_offline_gptq_marlin(self, x)
                bits = self._infer_gptq_weight_bits(in_features=in_features)
                return {
                    **meta, "gptq_weight_bits": bits,
                    "gptq_marlin_qweight": self.gptq_marlin_qweight,
                    "gptq_marlin_scales": self.gptq_marlin_scales,
                    "gptq_marlin_zp": self.gptq_marlin_zp,
                    "gptq_marlin_g_idx": self.gptq_marlin_g_idx,
                    "gptq_marlin_g_idx_sort_indices": self.gptq_marlin_g_idx_sort_indices,
                    "gptq_marlin_workspace": self.gptq_marlin_workspace,
                }
            raise RuntimeError(f"Offline GPTQ weights present but strategy weight_format={weight_format!r} not compatible.")
            
        if format_val == 2:
            if weight_format == "awq":
                return {
                    **meta,
                    "awq_qweight": self.awq_qweight, "awq_qzeros": self.awq_qzeros,
                    "awq_scales": self.awq_scales, "awq_group_size": group_size,
                }
            if weight_format == "awq_marlin":
                maybe_prepare_offline_awq_marlin(self, x)
                return {
                    **meta,
                    "awq_marlin_qweight": self.awq_marlin_qweight,
                    "awq_marlin_scales": self.awq_marlin_scales,
                    "awq_marlin_zp": self.awq_marlin_zp,
                    "awq_marlin_workspace": self.awq_marlin_workspace,
                    "awq_weight_bits": 4,
                }
            raise RuntimeError(f"Offline AWQ weights present but strategy weight_format={weight_format!r} not compatible.")
            
        raise RuntimeError(f"Unknown offline quant format: {format_val}")

    # Backward compatibility: old method names that external code may call
    def _maybe_prepare_offline_gptq(self, x: torch.Tensor) -> None:
        maybe_prepare_offline_gptq(self, x)

    def _maybe_prepare_offline_gptq_marlin(self, x: torch.Tensor) -> None:
        maybe_prepare_offline_gptq_marlin(self, x)

    def _maybe_prepare_offline_awq_marlin(self, x: torch.Tensor) -> None:
        maybe_prepare_offline_awq_marlin(self, x)

    # Re-export for backward compatibility
    set_quantized_weight = set_quantized_weight
    maybe_promote_weight_to_quantized_at_runtime = maybe_promote_weight_to_quantized_at_runtime
    maybe_quantize_loaded_weight_param = maybe_quantize_loaded_weight_param
