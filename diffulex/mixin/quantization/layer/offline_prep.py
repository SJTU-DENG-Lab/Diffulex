"""Offline quantization preparation utilities for Linear layer.

Handles GPTQ/AWQ/Marlin offline weight preparation and repacking.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _infer_module_device(module: nn.Module) -> torch.device:
    """Infer the device of a module."""
    w = getattr(module, "weight", None)
    if isinstance(w, torch.Tensor):
        return w.device
    for p in module.parameters(recurse=False):
        return p.device
    for b in module.buffers(recurse=False):
        return b.device
    return torch.device("cpu")


def maybe_prepare_offline_gptq(module: nn.Module, x: torch.Tensor) -> None:
    """Lazy shuffle of GPTQ weights using vLLM custom ops.
    
    Args:
        module: Linear module with GPTQ offline weights
        x: Input tensor (for device inference)
    """
    format_py = getattr(module, "_offline_quant_format_py", 0)
    is_shuffled = getattr(module, "_gptq_is_shuffled_py", False)
    qweight = getattr(module, "gptq_qweight", None)
    
    if format_py != 1 or qweight is None or qweight.numel() == 0 or is_shuffled:
        return
        
    try:
        from vllm import _custom_ops as ops  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "GPTQ offline weights loaded but cannot import vLLM CUDA custom ops (vllm._custom_ops)."
        ) from e
        
    g_idx = getattr(module, "gptq_g_idx", None)
    if g_idx is None or g_idx.numel() == 0:
        g_idx_tensor = torch.empty((0,), device=x.device, dtype=torch.int)
    else:
        g_idx_tensor = g_idx.to(device=x.device, dtype=torch.int)
        
    if qweight.device != x.device:
        raise RuntimeError(
            f"GPTQ qweight device mismatch: qweight on {qweight.device}, x on {x.device}. "
            "Ensure model and input are on the same device."
        )
        
    in_features = int(getattr(module, "_offline_quant_in_features_py", 0))
    if in_features is None or in_features <= 0:
        raise RuntimeError("GPTQ offline weights loaded but cannot infer in_features to compute weight_bits.")
        
    if qweight.shape[0] <= 0 or in_features % int(qweight.shape[0]) != 0:
        raise RuntimeError(
            f"GPTQ qweight shape invalid for weight_bits inference: "
            f"in_features={in_features}, qweight.shape={tuple(qweight.shape)}"
        )
        
    pack_factor = in_features // int(qweight.shape[0])
    if 32 % pack_factor != 0:
        raise RuntimeError(
            f"GPTQ pack_factor={pack_factor} unsupported (need 32%pack_factor==0), "
            f"in_features={in_features}, qweight.shape={tuple(qweight.shape)}"
        )
        
    weight_bits = 32 // pack_factor
    ops.gptq_shuffle(qweight, g_idx_tensor, weight_bits)
    
    is_shuffled_flag = getattr(module, "_gptq_is_shuffled", None)
    if is_shuffled_flag is not None:
        is_shuffled_flag.fill_(True)
    module._gptq_is_shuffled_py = True


def maybe_prepare_offline_gptq_marlin(module: nn.Module, x: torch.Tensor) -> None:
    """Lazy repack of GPTQ weights to Marlin format using vLLM custom ops.
    
    Args:
        module: Linear module with GPTQ offline weights
        x: Input tensor (for device inference)
    """
    format_py = getattr(module, "_offline_quant_format_py", 0)
    is_prepared = getattr(module, "_gptq_marlin_is_prepared_py", False)
    qweight = getattr(module, "gptq_qweight", None)
    
    if format_py != 1 or qweight is None or qweight.numel() == 0 or is_prepared:
        return
        
    try:
        from vllm import _custom_ops as ops  # type: ignore
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # type: ignore
            marlin_make_empty_g_idx,
            marlin_make_workspace_new,
            marlin_permute_scales,
            marlin_sort_g_idx,
        )
    except Exception as e:
        raise RuntimeError(
            "GPTQ Marlin requires vLLM CUDA custom ops + marlin_utils, which are not available in the current environment."
        ) from e
        
    device = x.device
    if qweight.device != device:
        raise RuntimeError(
            f"GPTQ qweight device mismatch: qweight on {qweight.device}, x on {device}. "
            "Ensure model and input are on the same device."
        )
        
    in_features = int(getattr(module, "_offline_quant_in_features_py", 0))
    out_features = int(getattr(module, "_offline_quant_out_features_py", 0))
    group_size = int(getattr(module, "_offline_quant_group_size_py", 0))
    
    if in_features <= 0 or out_features <= 0:
        raise RuntimeError(
            f"GPTQ Marlin: invalid feature sizes: in_features={in_features}, out_features={out_features}"
        )
        
    bits = int(getattr(module, "_offline_quant_bits_py", 0))
    if bits <= 0:
        if qweight.shape[0] <= 0 or in_features % int(qweight.shape[0]) != 0:
            raise RuntimeError(
                "GPTQ Marlin: cannot infer pack_factor from qweight shape: "
                f"in_features={in_features}, qweight.shape={tuple(qweight.shape)}"
            )
        pack_factor = in_features // int(qweight.shape[0])
        if 32 % pack_factor != 0:
            raise RuntimeError(f"GPTQ Marlin: unsupported pack_factor={pack_factor} (requires 32%pack_factor==0)")
        bits = 32 // pack_factor
        
    if bits not in (4, 8):
        raise RuntimeError(f"GPTQ Marlin: only 4/8-bit are supported in this integration, got bits={bits}")
        
    # Check if already marlin-ready
    marlin_qweight = getattr(module, "gptq_marlin_qweight", None)
    marlin_scales = getattr(module, "gptq_marlin_scales", None)
    already_ready = (
        marlin_qweight is not None and marlin_qweight.numel() > 0 and
        marlin_scales is not None and marlin_scales.numel() > 0
    )
    
    if already_ready:
        if marlin_qweight.device != device or marlin_scales.device != device:
            raise RuntimeError(
                "GPTQ Marlin: prepacked marlin tensors device mismatch: "
                f"qweight on {marlin_qweight.device}, scales on {marlin_scales.device}, x on {device}."
            )
            
    g_idx = getattr(module, "gptq_g_idx", None)
    if g_idx is not None and g_idx.numel() > 0:
        g_idx_sorted, g_idx_sort_indices = marlin_sort_g_idx(g_idx.to(device=device, dtype=torch.int32))
        module.gptq_marlin_g_idx = g_idx_sorted.contiguous()
        module.gptq_marlin_g_idx_sort_indices = g_idx_sort_indices.contiguous()
    else:
        module.gptq_marlin_g_idx = marlin_make_empty_g_idx(device)
        module.gptq_marlin_g_idx_sort_indices = marlin_make_empty_g_idx(device)
        
    module.gptq_marlin_workspace = marlin_make_workspace_new(device)
    
    if not already_ready:
        module.gptq_marlin_qweight = ops.gptq_marlin_repack(
            qweight.contiguous(),
            perm=module.gptq_marlin_g_idx_sort_indices,
            size_k=in_features,
            size_n=out_features,
            num_bits=bits,
            is_a_8bit=False,
        ).contiguous()
        module.gptq_marlin_scales = marlin_permute_scales(
            getattr(module, "gptq_scales").contiguous(),
            size_k=in_features,
            size_n=out_features,
            group_size=group_size,
            is_a_8bit=False,
        ).contiguous()
        
    module.gptq_marlin_zp = marlin_make_empty_g_idx(device)
    
    is_prepared_flag = getattr(module, "_gptq_marlin_is_prepared", None)
    if is_prepared_flag is not None:
        is_prepared_flag.fill_(True)
    module._gptq_marlin_is_prepared_py = True


def maybe_prepare_offline_awq_marlin(module: nn.Module, x: torch.Tensor) -> None:
    """Lazy repack of AWQ weights to Marlin format using vLLM custom ops.
    
    Args:
        module: Linear module with AWQ offline weights
        x: Input tensor (for device inference)
    """
    format_py = getattr(module, "_offline_quant_format_py", 0)
    is_prepared = getattr(module, "_awq_marlin_is_prepared_py", False)
    awq_qweight = getattr(module, "awq_qweight", None)
    
    if format_py != 2 or awq_qweight is None or awq_qweight.numel() == 0 or is_prepared:
        return
        
    try:
        from vllm import _custom_ops as ops  # type: ignore
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # type: ignore
            awq_to_marlin_zero_points,
            marlin_make_workspace_new,
            marlin_permute_scales,
        )
    except Exception as e:
        raise RuntimeError(
            "AWQ Marlin requires vLLM CUDA custom ops + marlin_utils, which are not available in the current environment."
        ) from e
        
    device = x.device
    if awq_qweight.device != device:
        raise RuntimeError(
            f"AWQ qweight device mismatch: qweight on {awq_qweight.device}, x on {device}. "
            "Ensure model and input are on the same device."
        )
        
    in_features = int(getattr(module, "_offline_quant_in_features_py", 0))
    out_features = int(getattr(module, "_offline_quant_out_features_py", 0))
    group_size = int(getattr(module, "_offline_quant_group_size_py", 0))
    
    if in_features <= 0 or out_features <= 0:
        raise RuntimeError(
            f"AWQ Marlin: invalid feature sizes: in_features={in_features}, out_features={out_features}"
        )
        
    pack_factor = out_features // int(awq_qweight.shape[1])
    if pack_factor != 8:
        raise RuntimeError(f"AWQ Marlin: expected pack_factor=8 (W4), got pack_factor={pack_factor}")
        
    num_groups = in_features // (in_features if group_size == -1 else group_size)
    
    module.awq_marlin_workspace = marlin_make_workspace_new(device)
    module.awq_marlin_qweight = ops.awq_marlin_repack(
        awq_qweight.contiguous(),
        size_k=in_features,
        size_n=out_features,
        num_bits=4,
        is_a_8bit=False,
    ).contiguous()
    module.awq_marlin_scales = marlin_permute_scales(
        getattr(module, "awq_scales").contiguous(),
        size_k=in_features,
        size_n=out_features,
        group_size=group_size,
        is_a_8bit=False,
    ).contiguous()
    module.awq_marlin_zp = awq_to_marlin_zero_points(
        getattr(module, "awq_qzeros").contiguous(),
        size_k=num_groups,
        size_n=out_features,
        num_bits=4,
        is_a_8bit=False,
    ).contiguous()
    
    is_prepared_flag = getattr(module, "_awq_marlin_is_prepared", None)
    if is_prepared_flag is not None:
        is_prepared_flag.fill_(True)
    module._awq_marlin_is_prepared_py = True
