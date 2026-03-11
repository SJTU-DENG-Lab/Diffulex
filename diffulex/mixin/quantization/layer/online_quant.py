"""Online quantization utilities for Linear layer.

Handles runtime weight quantization and promotion to quantized formats.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _get_linear_strategy(module: nn.Module):
    """Get linear quantization strategy for module."""
    from diffulex.quantization.context import get_linear_strategy
    kind = getattr(module, "_quant_kind", None) or getattr(module, "quant_kind", "other")
    return get_linear_strategy(kind)


def set_quantized_weight(
    module: nn.Module,
    quant_weight_int8: torch.Tensor,
    quant_scales: torch.Tensor
) -> None:
    """Set quantized weight for online quantization.
    
    Args:
        module: Linear module
        quant_weight_int8: Quantized weight tensor (int8/uint8/fp8)
        quant_scales: Scale tensor
    """
    fp8_dtypes: tuple[torch.dtype, ...] = tuple(
        d
        for d in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e4m3fnuz", None),
            getattr(torch, "float8_e5m2", None),
            getattr(torch, "float8_e5m2fnuz", None),
        )
        if d is not None
    )
    
    if quant_weight_int8.dtype not in (torch.int8, torch.uint8, *fp8_dtypes):
        raise TypeError(f"quant_weight_int8 must be int8/uint8/float8, got {quant_weight_int8.dtype}")
        
    try:
        strategy = _get_linear_strategy(module)
    except Exception:
        strategy = None
        
    scale_dtype = torch.bfloat16
    force_weight_contig = True
    
    if strategy is not None:
        weight_format = getattr(strategy, "linear_weight_format", None)
        act_format = getattr(strategy, "linear_act_format", None)
        
        if weight_format in ("fp8_e4m3", "fp8_e5m2") and act_format == "bf16":
            scale_dtype = torch.float32
            force_weight_contig = False
        elif weight_format == "int8" and act_format == "int8":
            scale_dtype = torch.float32
            force_weight_contig = False
        elif act_format in ("fp8_e4m3", "fp8_e5m2"):
            scale_dtype = torch.float32
            force_weight_contig = False
        elif act_format == "int8":
            scale_dtype = torch.float16
            
    if quant_scales.dtype != scale_dtype:
        quant_scales = quant_scales.to(dtype=scale_dtype)
    if force_weight_contig:
        quant_weight_int8 = quant_weight_int8.contiguous()
    quant_scales = quant_scales.contiguous()
    
    module.quant_weight_int8 = quant_weight_int8
    module.quant_scales = quant_scales
    module.quant_scales_1xn = quant_scales if quant_scales.dim() == 2 else quant_scales.view(1, -1)
    
    is_quantized_flag = getattr(module, "_weight_is_quantized", None)
    if is_quantized_flag is not None:
        is_quantized_flag.fill_(True)
    module._weight_is_quantized_py = True
    
    # Invalidate forward plan
    module._forward_plan = None


def maybe_promote_weight_to_quantized_at_runtime(
    module: nn.Module,
    x: torch.Tensor,
    strategy,
    *,
    expected_weight_formats: tuple[str, ...] = (
        "int8",
        "int4",
        "fp8_e4m3",
        "fp8_e5m2",
    ),
) -> None:
    """Promote bf16 weight to quantized format at runtime.
    
    Args:
        module: Linear module
        x: Input tensor
        strategy: Quantization strategy
        expected_weight_formats: Supported weight formats for promotion
    """
    if strategy is None:
        return
        
    # Check if offline or already quantized
    has_offline = getattr(module, "_offline_quant_format_py", 0) != 0
    has_quantized = getattr(module, "_weight_is_quantized_py", False)
    if has_offline or has_quantized:
        return
        
    weight_param = module._parameters.get("weight", None)
    if weight_param is None:
        return
        
    weight_format = getattr(strategy, "linear_weight_format", None)
    if weight_format not in expected_weight_formats or getattr(strategy, "name", "").startswith("linear_stub"):
        return
        
    w = getattr(module, "weight", None)
    if w is None or getattr(w, "dtype", None) not in (torch.bfloat16, torch.float16):
        return
        
    try:
        qweight, scales = strategy.quantize_weight_for_kernel(w.data, device=w.data.device)
    except Exception:
        return
        
    set_quantized_weight(module, qweight, scales)
    module._parameters.pop("weight", None)
    setattr(module, "weight", None)


def maybe_quantize_loaded_weight_param(
    module: nn.Module,
    param: nn.Parameter,
    *,
    loaded_shard_id: object = None,
    expected_shard_ids: set[object] | None = None,
) -> None:
    """Quantize loaded weight parameter if strategy supports it.
    
    Args:
        module: Linear module
        param: Weight parameter being loaded
        loaded_shard_id: Shard ID for multi-shard parameters
        expected_shard_ids: Expected set of shard IDs
    """
    current_weight = module._parameters.get("weight", None)
    if current_weight is None or current_weight is not param:
        return
        
    if expected_shard_ids is not None:
        if not hasattr(module, "_loaded_weight_shard_ids"):
            module._loaded_weight_shard_ids = set()
        module._loaded_weight_shard_ids.add(loaded_shard_id)
        if module._loaded_weight_shard_ids != expected_shard_ids:
            return
            
    try:
        strategy = _get_linear_strategy(module)
    except Exception:
        return
        
    if strategy is None or getattr(strategy, "name", "").startswith("linear_stub"):
        return
        
    weight_format = getattr(strategy, "linear_weight_format", None)
    if weight_format not in ("int8", "int4", "fp8_e4m3", "fp8_e5m2"):
        return
        
    qweight, scales = strategy.quantize_weight_for_kernel(param.data, device=param.data.device)
    set_quantized_weight(module, qweight, scales)
    module._parameters.pop("weight", None)
    setattr(module, "weight", None)
