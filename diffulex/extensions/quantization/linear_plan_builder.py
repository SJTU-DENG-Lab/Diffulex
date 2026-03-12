"""
Forward Plan Builder - vLLM-aligned factory for creating execution plans.

Builds plans that bind tensors at construction time to eliminate Python overhead.
"""

import torch
from typing import Optional, Any

from .linear_plans import (
    ForwardPlanSig, BF16Plan,
    QuantizedLinearPlan,
    OfflineGPTQPlan, OfflineAWQPlan,
    OfflineGPTQMarlinPlan, OfflineAWQMarlinPlan,
    DirectGPTQGemmPlan, DirectAWQGemmPlan, DirectMarlinGemmPlan,
)


def build_forward_plan(layer: torch.nn.Module, example_x: Optional[torch.Tensor] = None,
                       bias: Optional[torch.Tensor] = None) -> Any:
    """
    Build appropriate forward plan for a quantized linear layer.
    
    Binds tensors at build time to minimize Python overhead during forward.
    
    Args:
        layer: Linear layer with quantization buffers
        example_x: Example input tensor (for signature)
        bias: Optional bias tensor
        
    Returns:
        ForwardPlanBase instance
    """
    # Check for offline quantized weights
    if hasattr(layer, 'has_offline_quantized_weight') and layer.has_offline_quantized_weight():
        return _build_offline_plan(layer, example_x, bias)
    
    # Check for online quantized weights
    if hasattr(layer, 'has_quantized_weight') and layer.has_quantized_weight():
        return _build_online_plan(layer, example_x, bias)
    
    # Default BF16 plan
    return _build_bf16_plan(layer, example_x, bias)


def _build_signature(layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor],
                     mode: str, strategy_name: str) -> ForwardPlanSig:
    """Build plan signature for cache validation."""
    dev = x.device
    dev_idx = dev.index if dev.index is not None else 0
    
    # Get out_features from layer
    out_features = getattr(layer, '_forward_out_features', None)
    if out_features is None:
        # Try to infer from weights
        if hasattr(layer, 'quant_weight') and layer.quant_weight is not None:
            out_features = layer.quant_weight.shape[1] if len(layer.quant_weight.shape) > 1 else None
        elif hasattr(layer, 'weight') and layer.weight is not None:
            out_features = layer.weight.shape[0]
    
    return ForwardPlanSig(
        device_type=dev.type,
        device_index=dev_idx,
        x_dtype=x.dtype,
        x_shape=tuple(x.shape),
        has_bias=bias is not None,
        mode=mode,
        strategy_name=strategy_name
    )


def _build_bf16_plan(layer: torch.nn.Module, example_x: Optional[torch.Tensor],
                     bias: Optional[torch.Tensor]) -> BF16Plan:
    """Build BF16 plan with bound weight and bias."""
    weight = getattr(layer, 'weight', None)
    sig = _build_signature(layer, example_x, bias, "bf16", "bf16_linear") if example_x is not None else None
    return BF16Plan(sig, weight, bias)


def _build_online_plan(layer: torch.nn.Module, example_x: torch.Tensor,
                       bias: Optional[torch.Tensor]) -> Any:
    """Build plan for online quantized weights (INT8/FP8) with bound tensors."""
    strategy = None
    if hasattr(layer, '_quant_strategy'):
        strategy = layer._quant_strategy
    
    if strategy is None:
        return _build_bf16_plan(layer, example_x, bias)
    
    strategy_name = strategy.name
    weight_format = strategy.linear_weight_format
    act_format = strategy.linear_act_format
    quant_kind = getattr(layer, 'quant_kind', 'other')
    
    # Get bound tensors
    qweight = getattr(layer, 'quant_weight', None)
    scales = getattr(layer, 'quant_scales', None)
    
    if qweight is None or scales is None:
        return _build_bf16_plan(layer, example_x, bias)
    
    # Ensure scales are 1xN for broadcasting
    scales_1xn = scales if scales.dim() == 2 else scales.view(1, -1)
    
    # Infer out_features
    out_features = qweight.shape[1] if len(qweight.shape) > 1 else None
    if out_features is None:
        out_features = getattr(layer, '_forward_out_features', None)
    
    sig = _build_signature(layer, example_x, bias, "quant", strategy_name)
    
    # Build unified quantized linear plan
    return QuantizedLinearPlan(
        sig, strategy, quant_kind,
        qweight, scales_1xn, out_features, bias,
        weight_format=weight_format,
        act_format=act_format,
    )


def _build_offline_plan(layer: torch.nn.Module, example_x: torch.Tensor,
                        bias: Optional[torch.Tensor]) -> Any:
    """Build plan for offline quantized weights (GPTQ/AWQ/Marlin)."""
    weight_format = None
    if hasattr(layer, '_offline_quant_format'):
        weight_format = layer._offline_quant_format
    elif hasattr(layer, '_offline_quant_format_py'):
        weight_format = layer._offline_quant_format_py
    
    if weight_format is None:
        return _build_bf16_plan(layer, example_x, bias)
    
    strategy = getattr(layer, '_quant_strategy', None)
    strategy_name = strategy.name if strategy else weight_format
    quant_kind = getattr(layer, 'quant_kind', 'other')
    
    sig = _build_signature(layer, example_x, bias, "offline", strategy_name)
    
    # Check for Marlin format
    if "marlin" in weight_format.lower():
        return _build_marlin_plan(layer, sig, strategy, quant_kind, bias)
    
    # Check for GPTQ format
    if "gptq" in weight_format.lower():
        return _build_gptq_plan(layer, sig, strategy, quant_kind, bias)
    
    # Check for AWQ format
    if "awq" in weight_format.lower():
        return _build_awq_plan(layer, sig, strategy, quant_kind, bias)
    
    return _build_bf16_plan(layer, example_x, bias)


def _build_gptq_plan(layer: torch.nn.Module, sig: ForwardPlanSig,
                     strategy: Any, quant_kind: str, bias: Optional[torch.Tensor]) -> Any:
    """Build GPTQ plan with bound tensors."""
    qweight = getattr(layer, 'gptq_qweight', None)
    qzeros = getattr(layer, 'gptq_qzeros', None)
    scales = getattr(layer, 'gptq_scales', None)
    g_idx = getattr(layer, 'gptq_g_idx', None)
    
    if qweight is None or qzeros is None or scales is None:
        return _build_bf16_plan(layer, None, bias)
    
    bits = getattr(layer, '_offline_quant_bits', 4)
    is_shuffled = getattr(layer, '_gptq_is_shuffled_py', False) or bool(getattr(layer, '_gptq_is_shuffled', False))
    
    # Infer dimensions
    out_features = getattr(layer, '_forward_out_features', None)
    in_features = None
    group_size = getattr(layer, '_offline_quant_group_size', 128)
    
    if strategy is not None:
        return OfflineGPTQPlan(
            sig, strategy, quant_kind,
            qweight, qzeros, scales, g_idx,
            bits, is_shuffled,
            out_features, in_features, group_size,
            bias
        )
    
    return _build_bf16_plan(layer, None, bias)


def _build_awq_plan(layer: torch.nn.Module, sig: ForwardPlanSig,
                    strategy: Any, quant_kind: str, bias: Optional[torch.Tensor]) -> Any:
    """Build AWQ plan with bound tensors."""
    qweight = getattr(layer, 'awq_qweight', None)
    qzeros = getattr(layer, 'awq_qzeros', None)
    scales = getattr(layer, 'awq_scales', None)
    
    if qweight is None or qzeros is None or scales is None:
        return _build_bf16_plan(layer, None, bias)
    
    bits = getattr(layer, '_offline_quant_bits', 4)
    pack_factor = 32 // max(1, bits)
    
    out_features = getattr(layer, '_forward_out_features', None)
    in_features = None
    group_size = getattr(layer, '_offline_quant_group_size', 128)
    
    if strategy is not None:
        return OfflineAWQPlan(
            sig, strategy, quant_kind,
            qweight, qzeros, scales,
            bits, pack_factor,
            out_features, in_features, group_size,
            bias
        )
    
    return _build_bf16_plan(layer, None, bias)


def _build_marlin_plan(layer: torch.nn.Module, sig: ForwardPlanSig,
                       strategy: Any, quant_kind: str, bias: Optional[torch.Tensor]) -> Any:
    """Build Marlin plan with bound tensors."""
    marlin_qweight = getattr(layer, 'gptq_marlin_qweight', None) or getattr(layer, 'marlin_qweight', None)
    marlin_scales = getattr(layer, 'gptq_marlin_scales', None) or getattr(layer, 'marlin_scales', None)
    marlin_workspace = getattr(layer, 'gptq_marlin_workspace', None) or getattr(layer, 'marlin_workspace', None)
    
    if marlin_qweight is None or marlin_scales is None or marlin_workspace is None:
        return _build_bf16_plan(layer, None, bias)
    
    bits = getattr(layer, '_offline_quant_bits', 4)
    is_k_full = True
    
    out_features = getattr(layer, '_forward_out_features', None)
    in_features = None
    group_size = getattr(layer, '_offline_quant_group_size', 128)
    
    if strategy is not None:
        if "gptq" in strategy.name:
            return OfflineGPTQMarlinPlan(
                sig, strategy, quant_kind,
                marlin_qweight, marlin_scales, marlin_workspace,
                bits, is_k_full,
                in_features, out_features, group_size,
                bias
            )
        else:
            return OfflineAWQMarlinPlan(
                sig, strategy, quant_kind,
                marlin_qweight, marlin_scales, marlin_workspace,
                bits, is_k_full,
                in_features, out_features, group_size,
                bias
            )
    
    return _build_bf16_plan(layer, None, bias)


def rebuild_plan_if_needed(layer: torch.nn.Module, x: torch.Tensor,
                           bias: Optional[torch.Tensor] = None) -> bool:
    """
    Check if plan needs rebuild and rebuild if necessary.
    
    Returns:
        True if plan was rebuilt, False otherwise
    """
    if not hasattr(layer, '_forward_plan') or layer._forward_plan is None:
        layer._forward_plan = build_forward_plan(layer, x, bias)
        return True
    
    plan = layer._forward_plan
    sig = plan.get_signature()
    
    if sig is None:
        layer._forward_plan = build_forward_plan(layer, x, bias)
        return True
    
    # Check signature match
    dev = x.device
    if (sig.device_type == dev.type and 
        sig.device_index == (dev.index if dev.index is not None else 0) and
        sig.x_dtype == x.dtype and 
        sig.x_shape == tuple(x.shape)):
        return False
    
    # Signature mismatch, rebuild
    layer._forward_plan = build_forward_plan(layer, x, bias)
    return True
