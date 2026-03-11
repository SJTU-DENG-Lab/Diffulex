"""Offline quantized weight loading (GPTQ/AWQ)."""

from __future__ import annotations

import os
import json
from typing import Optional

import torch
import torch.nn as nn
from safetensors import safe_open

from diffulex.config import Config
from diffulex.logger import get_logger

logger = get_logger(__name__)


def _read_quantize_config(model_dir: str) -> dict:
    """Read vLLM-style quantization metadata if present."""
    cfg_path = os.path.join(model_dir, "quantize_config.json")
    if not os.path.exists(cfg_path):
        return {}
    try:
        with open(cfg_path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _make_packed_qzeros_constant(
    *, num_groups: int, out_features: int, bits: int, device: torch.device | str
) -> torch.Tensor:
    """Create a GPTQ-style packed qzeros tensor filled with a constant."""
    if bits not in (2, 4, 8):
        raise ValueError(f"Unsupported bits={bits} for packed qzeros (expected 2/4/8)")
    pack_factor = 32 // bits
    if out_features % pack_factor != 0:
        raise ValueError(f"out_features={out_features} not divisible by pack_factor={pack_factor}")
    out_packed = out_features // pack_factor

    z = (1 << (bits - 1)) - 1
    packed_val = 0
    for i in range(pack_factor):
        packed_val |= (z & ((1 << bits) - 1)) << (bits * i)

    return torch.full(
        (int(num_groups), int(out_packed)), int(packed_val),
        dtype=torch.int32, device=device,
    )


def _set_offline_gptq_marlin_weight(
    module: nn.Module,
    *, qweight: torch.Tensor, scales: torch.Tensor,
    out_features: int, in_features: int, group_size: int, bits: int,
    g_idx: torch.Tensor | None,
) -> None:
    """Directly set GPTQ-Marlin-ready offline weights into a Linear module."""
    from diffulex.utils.loader.core import _infer_module_device
    
    module_device = _infer_module_device(module)
    if qweight.device != module_device:
        qweight = qweight.to(device=module_device)
    if scales.device != module_device:
        scales = scales.to(device=module_device)
    if g_idx is not None and g_idx.device != module_device:
        g_idx = g_idx.to(device=module_device)

    pack_factor = 32 // int(bits)
    group_size_norm = in_features if group_size == -1 else group_size
    if group_size_norm <= 0 or in_features % group_size_norm != 0:
        raise ValueError(f"Invalid group_size={group_size} for in_features={in_features}")
    num_groups = in_features // group_size_norm

    qzeros = _make_packed_qzeros_constant(
        num_groups=num_groups, out_features=out_features, bits=int(bits), device=module_device
    )

    module.gptq_qweight = qweight
    module.gptq_qzeros = qzeros
    module.gptq_scales = scales.to(dtype=torch.float16)
    if g_idx is None or getattr(g_idx, "numel", lambda: 1)() == 0:
        module.gptq_g_idx = torch.empty(0, dtype=torch.int32, device=module_device)
    else:
        module.gptq_g_idx = g_idx.to(dtype=torch.int32)

    module.gptq_marlin_qweight = qweight
    module.gptq_marlin_scales = module.gptq_scales

    module._offline_quant_format = torch.tensor(1, dtype=torch.int8, device=module_device)
    module._offline_quant_bits = torch.tensor(int(bits), dtype=torch.int32, device=module_device)
    module._offline_quant_group_size = torch.tensor(group_size, dtype=torch.int32, device=module_device)
    module._offline_quant_out_features = torch.tensor(out_features, dtype=torch.int32, device=module_device)
    module._offline_quant_in_features = torch.tensor(in_features, dtype=torch.int32, device=module_device)
    module._gptq_is_shuffled = torch.tensor(False, dtype=torch.bool, device=module_device)
    
    if hasattr(module, "_offline_quant_format_py"):
        module._offline_quant_format_py = 1
    if hasattr(module, "_offline_quant_bits_py"):
        module._offline_quant_bits_py = int(bits)
    if hasattr(module, "_offline_quant_group_size_py"):
        module._offline_quant_group_size_py = int(group_size)
    if hasattr(module, "_offline_quant_out_features_py"):
        module._offline_quant_out_features_py = int(out_features)
    if hasattr(module, "_offline_quant_in_features_py"):
        module._offline_quant_in_features_py = int(in_features)
    if hasattr(module, "_gptq_is_shuffled_py"):
        module._gptq_is_shuffled_py = False
    if hasattr(module, "_gptq_marlin_is_prepared_py"):
        module._gptq_marlin_is_prepared_py = False

    module._gptq_marlin_is_prepared = torch.tensor(False, dtype=torch.bool, device=module_device)
    module.gptq_marlin_zp = torch.empty(0, dtype=torch.int32, device=module_device)
    module.gptq_marlin_g_idx = torch.empty(0, dtype=torch.int32, device=module_device)
    module.gptq_marlin_g_idx_sort_indices = torch.empty(0, dtype=torch.int32, device=module_device)
    module.gptq_marlin_workspace = torch.empty(0, dtype=torch.int32, device=module_device)

    if hasattr(module, "_parameters") and "weight" in module._parameters:
        module._parameters.pop("weight", None)
        setattr(module, "weight", None)


def _load_tensors_for_prefix(
    key_dict: dict[str, str],
    key_to_file: dict[str, str],
    *, want_g_idx: bool
) -> tuple[Optional[torch.Tensor], ...]:
    """Load qweight/qzeros/scales/(g_idx) from the minimal set of safetensors files."""
    qweight = qzeros = scales = g_idx = None
    keys = [key_dict.get("qweight"), key_dict.get("qzeros"), key_dict.get("scales")]
    if want_g_idx:
        keys.append(key_dict.get("g_idx"))
    files_needed = {key_to_file.get(k) for k in keys if k}
    files_needed.discard(None)

    for file in files_needed:
        with safe_open(file, "pt", "cpu") as f:
            if qweight is None and (key_dict.get("qweight") in f.keys()):
                qweight = f.get_tensor(key_dict["qweight"])
            if qzeros is None and (key_dict.get("qzeros") in f.keys()):
                qzeros = f.get_tensor(key_dict["qzeros"])
            if scales is None and (key_dict.get("scales") in f.keys()):
                scales = f.get_tensor(key_dict["scales"])
            if want_g_idx and g_idx is None and ("g_idx" in key_dict) and (key_dict["g_idx"] in f.keys()):
                g_idx = f.get_tensor(key_dict["g_idx"])
    return qweight, qzeros, scales, g_idx


def load_gptq_awq_weights(model: nn.Module, config: Config):
    """Load GPTQ/AWQ offline quantized weights from checkpoint.
    
    Returns:
        Tuple of (loaded_gptq_count, loaded_awq_count, skipped_count)
    """
    loaded_gptq = 0
    loaded_awq = 0
    skipped = 0

    weight_attn_dtype = getattr(config, "linear_attn_weight_dtype", "bf16") or "bf16"
    weight_mlp_dtype = getattr(config, "linear_mlp_weight_dtype", "bf16") or "bf16"
    quantize_cfg = _read_quantize_config(getattr(config, "model", ""))
    checkpoint_format = (quantize_cfg.get("checkpoint_format") or "").strip().lower()
    ckpt_bits = int(quantize_cfg.get("bits", 0) or 0)
    ckpt_group_size = int(quantize_cfg.get("group_size", 0) or 0)

    gptq_dtypes = {"gptq", "gptq_marlin"}
    awq_dtypes = {"awq", "awq_marlin"}
    use_gptq = (weight_attn_dtype or "").lower() in gptq_dtypes or (weight_mlp_dtype or "").lower() in gptq_dtypes
    use_awq = (weight_attn_dtype or "").lower() in awq_dtypes or (weight_mlp_dtype or "").lower() in awq_dtypes
    want_gptq_marlin = (weight_attn_dtype or "").lower() == "gptq_marlin" or (weight_mlp_dtype or "").lower() == "gptq_marlin"
    want_awq_marlin = (weight_attn_dtype or "").lower() == "awq_marlin" or (weight_mlp_dtype or "").lower() == "awq_marlin"
    is_gptq_marlin_ckpt = checkpoint_format == "gptq_marlin"
    is_awq_marlin_ckpt = checkpoint_format == "awq_marlin"

    if not (use_gptq or use_awq):
        return loaded_gptq, loaded_awq, skipped

    from glob import glob
    all_files = list(glob(os.path.join(config.model, "*.safetensors")))

    # Scan keys once
    key_to_file: dict[str, str] = {}
    module_keys: dict[str, dict[str, str]] = {}
    offline_suffixes = (".qweight", ".qzeros", ".scales", ".g_idx")
    
    for file in all_files:
        with safe_open(file, "pt", "cpu") as f:
            for key in f.keys():
                if not key.endswith(offline_suffixes):
                    continue
                key_to_file[key] = file
                if key.endswith(".qweight"):
                    prefix = key[:-8]
                    module_keys.setdefault(prefix, {})["qweight"] = key
                elif key.endswith(".qzeros"):
                    prefix = key[:-7]
                    module_keys.setdefault(prefix, {})["qzeros"] = key
                elif key.endswith(".scales"):
                    prefix = key[:-7]
                    module_keys.setdefault(prefix, {})["scales"] = key
                else:
                    prefix = key[:-6]
                    module_keys.setdefault(prefix, {})["g_idx"] = key

    named_modules = dict(model.named_modules())
    offline_capable_modules: dict[str, nn.Module] = {
        name: m for name, m in named_modules.items() if hasattr(m, "set_offline_quantized_weight")
    }

    def _find_offline_capable_module(module_name: str) -> nn.Module | None:
        m = offline_capable_modules.get(module_name)
        if m is not None:
            return m
        leaf = module_name.split(".")[-1] if module_name else module_name
        for name, cand in offline_capable_modules.items():
            if name == module_name or name.endswith("." + module_name) or module_name.endswith("." + name) or (name.split(".")[-1] == leaf):
                return cand
        return None

    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    for prefix, key_dict in module_keys.items():
        if "qweight" not in key_dict or "qzeros" not in key_dict or "scales" not in key_dict:
            continue

        module_name = prefix
        for k, (v, _) in packed_modules_mapping.items():
            if k in prefix:
                module_name = prefix.replace(k, v)
                break

        try:
            module = _find_offline_capable_module(module_name)
            if module is None:
                skipped += 1
                continue

            has_g_idx = "g_idx" in key_dict
            is_gptq_keyset = has_g_idx or is_gptq_marlin_ckpt
            if is_gptq_keyset and use_gptq:
                format = "gptq"
            elif (not is_gptq_keyset) and use_awq:
                format = "awq"
            else:
                format = "gptq" if (use_gptq and is_gptq_keyset) else ("awq" if use_awq else None)

            if format is None:
                skipped += 1
                continue

            qweight, qzeros, scales, g_idx = _load_tensors_for_prefix(
                key_dict, key_to_file, want_g_idx=(format == "gptq")
            )

            if qweight is None or qzeros is None or scales is None:
                skipped += 1
                continue

            # Infer dimensions
            if format == "gptq":
                if is_gptq_marlin_ckpt:
                    out_features = int(scales.shape[1]) if scales.ndim == 2 else int(qweight.shape[1])
                    in_features = int(qweight.shape[0]) * 16
                    if ckpt_bits not in (4, 8):
                        logger.warning(f"gptq_marlin requires bits=4/8, got bits={ckpt_bits} for {module_name}. Skipping.")
                        skipped += 1
                        continue
                    pack_factor = 32 // int(ckpt_bits)
                else:
                    out_features = int(qweight.shape[1])
                    if getattr(qzeros, "numel", lambda: 1)() == 0:
                        if ckpt_bits not in (2, 4, 8):
                            logger.warning(f"qzeros empty, cannot infer bits for {module_name}. Skipping.")
                            skipped += 1
                            continue
                        pack_factor = 32 // int(ckpt_bits)
                    else:
                        if int(qzeros.shape[1]) <= 0 or out_features % int(qzeros.shape[1]) != 0:
                            logger.warning(f"Cannot infer GPTQ pack_factor for {module_name}. Skipping.")
                            skipped += 1
                            continue
                        pack_factor = out_features // int(qzeros.shape[1])
                    in_features = int(qweight.shape[0]) * pack_factor
            else:
                out_features = int(scales.shape[1]) if scales.ndim == 2 else int(qweight.shape[1])
                if int(qweight.shape[1]) <= 0 or out_features % int(qweight.shape[1]) != 0:
                    logger.warning(f"Cannot infer AWQ pack_factor for {module_name}. Skipping.")
                    skipped += 1
                    continue
                pack_factor = out_features // int(qweight.shape[1])
                in_features = int(qweight.shape[0])

            # Infer group_size
            group_size = 128
            if ckpt_group_size not in (0, None):
                group_size = int(ckpt_group_size)
            else:
                if is_gptq_marlin_ckpt and len(scales.shape) == 2 and int(scales.shape[0]) > 0:
                    num_groups = int(scales.shape[0])
                    if num_groups > 0 and in_features % num_groups == 0:
                        group_size = in_features // num_groups
                    elif num_groups % 2 == 0 and (in_features % (num_groups // 2)) == 0:
                        group_size = in_features // (num_groups // 2)
                else:
                    num_groups = int(qzeros.shape[0]) if getattr(qzeros, "numel", lambda: 1)() > 0 else 0
                    if num_groups > 0 and in_features % num_groups == 0:
                        group_size = in_features // num_groups
                    elif len(scales.shape) == 2 and int(scales.shape[0]) > 0 and in_features % int(scales.shape[0]) == 0:
                        group_size = in_features // int(scales.shape[0])

            # Dummy qzeros for gptq_marlin
            if (format == "gptq" and getattr(qzeros, "numel", lambda: 1)() == 0
                and (want_gptq_marlin or is_gptq_marlin_ckpt) and ckpt_bits in (2, 4, 8)):
                group_size_norm = in_features if group_size == -1 else group_size
                if group_size_norm <= 0 or (in_features % group_size_norm) != 0:
                    logger.warning(f"Invalid group_size={group_size} for {module_name}. Skipping.")
                    skipped += 1
                    continue
                num_groups = in_features // group_size_norm
                try:
                    qzeros = _make_packed_qzeros_constant(
                        num_groups=num_groups, out_features=out_features, bits=int(ckpt_bits), device=qweight.device
                    )
                except Exception as e:
                    logger.warning(f"Failed to create dummy qzeros for {module_name}: {e}. Skipping.")
                    skipped += 1
                    continue

            # Handle TP sharding
            result = _apply_tp_sharding(
                module, qweight, qzeros, scales, g_idx, format,
                in_features, out_features, group_size, pack_factor,
                ckpt_bits, is_gptq_marlin_ckpt, module_name, logger
            )
            if result is None:
                skipped += 1
                continue
            qweight, qzeros, scales, g_idx, in_features, out_features = result

            if g_idx is not None and getattr(g_idx, "numel", lambda: 1)() == 0:
                g_idx = None

            try:
                if format == "gptq" and is_gptq_marlin_ckpt:
                    if ckpt_bits not in (4, 8):
                        raise ValueError(f"gptq_marlin checkpoint requires bits=4/8, got bits={ckpt_bits}")
                    _set_offline_gptq_marlin_weight(
                        module, qweight=qweight, scales=scales,
                        out_features=out_features, in_features=in_features,
                        group_size=group_size, bits=int(ckpt_bits), g_idx=g_idx,
                    )
                else:
                    module.set_offline_quantized_weight(
                        format=format, qweight=qweight, qzeros=qzeros, scales=scales,
                        out_features=out_features, in_features=in_features,
                        group_size=group_size, g_idx=g_idx,
                    )
                if format == "gptq":
                    loaded_gptq += 1
                else:
                    loaded_awq += 1
            except Exception as e:
                logger.exception(f"Failed to load offline quantized weights for {module_name}: {e}")
                skipped += 1

        except Exception as e:
            logger.exception(f"Error loading offline quantized weights for {prefix}: {e}")
            skipped += 1

    return loaded_gptq, loaded_awq, skipped


def _apply_tp_sharding(
    module, qweight, qzeros, scales, g_idx, format,
    in_features, out_features, group_size, pack_factor,
    ckpt_bits, is_gptq_marlin_ckpt, module_name, logger
):
    """Apply tensor parallel sharding to quantized weights."""
    tp_size = int(getattr(module, "tp_size", 1) or 1)
    tp_rank = int(getattr(module, "tp_rank", 0) or 0)
    tp_dim = getattr(module, "tp_dim", None)
    
    if tp_size <= 1:
        return qweight, qzeros, scales, g_idx, in_features, out_features
        
    if tp_dim not in (0, 1):
        logger.warning(f"Unsupported tp_dim={tp_dim} for offline quantized weights. Skipping {module_name}.")
        return None

    if tp_dim == 0:  # ColumnParallel - shard N
        if out_features % tp_size != 0:
            logger.warning(f"out_features={out_features} not divisible by TP={tp_size}. Skipping {module_name}.")
            return None
        out_per = out_features // tp_size
        out_start = tp_rank * out_per
        out_end = out_start + out_per
        if out_per % pack_factor != 0:
            logger.warning(f"out_per={out_per} not divisible by pack_factor={pack_factor}. Skipping {module_name}.")
            return None
        out_packed_per = out_per // pack_factor
        out_packed_start = out_start // pack_factor

        if format == "gptq":
            if is_gptq_marlin_ckpt:
                n_factor = int(ckpt_bits) // 2
                if n_factor <= 0:
                    logger.warning(f"Invalid gptq_marlin n_factor for bits={ckpt_bits}. Skipping {module_name}.")
                    return None
                qweight = qweight[:, (out_start * n_factor) : (out_end * n_factor)]
                scales = scales[:, out_start:out_end]
                out_features = out_per
            else:
                qweight = qweight[:, out_start:out_end]
                qzeros = qzeros[:, out_packed_start:out_packed_start + out_packed_per]
                scales = scales[:, out_start:out_end]
                out_features = out_per
        else:  # awq
            qweight = qweight[:, out_packed_start:out_packed_start + out_packed_per]
            qzeros = qzeros[:, out_packed_start:out_packed_start + out_packed_per]
            scales = scales[:, out_start:out_end]
            out_features = out_per
            
    else:  # RowParallel - shard K
        if in_features % tp_size != 0:
            logger.warning(f"in_features={in_features} not divisible by TP={tp_size}. Skipping {module_name}.")
            return None
        in_per = in_features // tp_size
        in_start = tp_rank * in_per
        in_end = in_start + in_per
        if group_size <= 0 or (in_per % group_size) != 0 or (in_start % group_size) != 0:
            logger.warning(f"group_size={group_size} incompatible with TP sharding. Skipping {module_name}.")
            return None
        g_start = in_start // group_size
        g_end = in_end // group_size

        if format == "gptq":
            if is_gptq_marlin_ckpt:
                if in_start % 16 != 0:
                    logger.warning(f"gptq_marlin requires in_start divisible by 16. Skipping {module_name}.")
                    return None
                q_start = in_start // 16
                q_end = in_end // 16
                qweight = qweight[q_start:q_end, :]
                group_size_norm = in_features if group_size == -1 else group_size
                expected_num_groups = in_features // group_size_norm if group_size_norm > 0 else 0
                if expected_num_groups <= 0:
                    logger.warning(f"Invalid expected_num_groups. Skipping {module_name}.")
                    return None
                if int(scales.shape[0]) == expected_num_groups:
                    scales = scales[g_start:g_end, :]
                elif int(scales.shape[0]) == 2 * expected_num_groups:
                    scales = scales[(2 * g_start) : (2 * g_end), :]
                else:
                    logger.warning(f"Unexpected gptq_marlin scales.shape. Skipping {module_name}.")
                    return None
                if g_idx is not None and getattr(g_idx, "numel", lambda: 1)() > 0:
                    g_idx = g_idx[in_start:in_end]
                in_features = in_per
            else:
                if in_start % pack_factor != 0:
                    logger.warning(f"in_start={in_start} not divisible by pack_factor. Skipping {module_name}.")
                    return None
                q_start = in_start // pack_factor
                qweight = qweight[q_start:q_start + (in_per // pack_factor), :]
                qzeros = qzeros[g_start:g_end, :]
                scales = scales[g_start:g_end, :]
                if g_idx is not None and getattr(g_idx, "numel", lambda: 1)() > 0:
                    g_idx = g_idx[in_start:in_end]
                in_features = in_per
        else:  # awq
            qweight = qweight[in_start:in_end, :]
            qzeros = qzeros[g_start:g_end, :]
            scales = scales[g_start:g_end, :]
            in_features = in_per

    return qweight, qzeros, scales, g_idx, in_features, out_features
