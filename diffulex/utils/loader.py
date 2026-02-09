import os
import json
import torch
import torch.nn as nn

from tqdm import tqdm
from glob import glob
from functools import partial
from safetensors import safe_open
from diffulex.config import Config
from diffulex.logger import get_logger

logger = get_logger(__name__)

def _read_quantize_config(model_dir: str) -> dict:
    """Read vLLM-style quantization metadata if present.

    We use this to detect checkpoint formats like `gptq_marlin` which reuse the same
    tensor keys (qweight/qzeros/scales[/g_idx]) but have different semantics.
    """
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
    *,
    num_groups: int,
    out_features: int,
    bits: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Create a GPTQ-style packed qzeros tensor filled with a constant.

    For vLLM GPTQ v1 checkpoints, zeros are stored as (zeros - 1) and then bit-packed
    along the output dimension (N). For symmetric quantization, zeros is typically
    bias=2^(bits-1), thus stored constant becomes (2^(bits-1) - 1).

    This is primarily used as a *shape-compatible dummy* when loading gptq_marlin
    checkpoints where runtime zero-points are intentionally unused (qzeros may be empty).
    """
    if bits not in (2, 4, 8):
        raise ValueError(f"Unsupported bits={bits} for packed qzeros (expected 2/4/8)")
    pack_factor = 32 // bits
    if out_features % pack_factor != 0:
        raise ValueError(
            f"out_features={out_features} not divisible by pack_factor={pack_factor} for bits={bits}"
        )
    out_packed = out_features // pack_factor

    # Stored constant for GPTQ v1: bias - 1, where bias = 2^(bits-1).
    z = (1 << (bits - 1)) - 1
    packed_val = 0
    for i in range(pack_factor):
        packed_val |= (z & ((1 << bits) - 1)) << (bits * i)

    return torch.full(
        (int(num_groups), int(out_packed)),
        int(packed_val),
        dtype=torch.int32,
        device=device,
    )


def _infer_module_device(module: nn.Module) -> torch.device:
    w = getattr(module, "weight", None)
    if isinstance(w, torch.Tensor):
        return w.device
    for p in module.parameters(recurse=False):
        return p.device
    for b in module.buffers(recurse=False):
        return b.device
    return torch.device("cpu")


def _set_offline_gptq_marlin_weight(
    module: nn.Module,
    *,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    out_features: int,
    in_features: int,
    group_size: int,
    bits: int,
    g_idx: torch.Tensor | None,
) -> None:
    """Directly set GPTQ-Marlin-ready offline weights into a Diffulex Linear module.

    This bypasses `set_offline_quantized_weight` because marlin-exported `scales`
    use a different layout (e.g. (2*num_groups, out_features/2)) and would fail
    the standard GPTQ shape validation.

    We still populate minimal GPTQ metadata/buffers so Diffulex forward chooses
    the offline path, and then `LinearBase._maybe_prepare_offline_gptq_marlin`
    will only allocate workspace / g_idx metadata (and not repack/permute again).
    """
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

    # Minimal qzeros to satisfy offline presence checks. (Marlin GPTQ symmetric doesn't use runtime zp.)
    qzeros = _make_packed_qzeros_constant(
        num_groups=num_groups,
        out_features=out_features,
        bits=int(bits),
        device=module_device,
    )

    # Populate GPTQ buffers (note: scales here are marlin layout; gptq kernels should not be used).
    module.gptq_qweight = qweight
    module.gptq_qzeros = qzeros
    module.gptq_scales = scales.to(dtype=torch.float16)
    if g_idx is None:
        module.gptq_g_idx = torch.empty(0, dtype=torch.int32, device=module_device)
    else:
        if getattr(g_idx, "numel", lambda: 1)() == 0:
            module.gptq_g_idx = torch.empty(0, dtype=torch.int32, device=module_device)
        else:
            module.gptq_g_idx = g_idx.to(dtype=torch.int32)

    # Also mark as marlin-ready so LinearBase won't repack/permute again.
    module.gptq_marlin_qweight = qweight
    module.gptq_marlin_scales = module.gptq_scales

    module._offline_quant_format = torch.tensor(1, dtype=torch.int8, device=module_device)
    module._offline_quant_bits = torch.tensor(int(bits), dtype=torch.int32, device=module_device)
    module._offline_quant_group_size = torch.tensor(group_size, dtype=torch.int32, device=module_device)
    module._offline_quant_out_features = torch.tensor(out_features, dtype=torch.int32, device=module_device)
    module._offline_quant_in_features = torch.tensor(in_features, dtype=torch.int32, device=module_device)
    module._gptq_is_shuffled = torch.tensor(False, dtype=torch.bool, device=module_device)
    # Keep Python-side mirrors in sync; runtime fast paths rely on these and
    # must not `.item()` from CUDA tensors (graph capture / perf).
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

    # Reset marlin-prep caches (workspace/zp/g_idx meta will be created on first forward).
    module._gptq_marlin_is_prepared = torch.tensor(False, dtype=torch.bool, device=module_device)
    module.gptq_marlin_zp = torch.empty(0, dtype=torch.int32, device=module_device)
    module.gptq_marlin_g_idx = torch.empty(0, dtype=torch.int32, device=module_device)
    module.gptq_marlin_g_idx_sort_indices = torch.empty(0, dtype=torch.int32, device=module_device)
    module.gptq_marlin_workspace = torch.empty(0, dtype=torch.int32, device=module_device)

    # Drop bf16 weight Parameter if present (to free memory and avoid accidental fallback).
    if hasattr(module, "_parameters") and "weight" in module._parameters:
        module._parameters.pop("weight", None)
        setattr(module, "weight", None)


def load_lora_config(lora_path: str) -> dict:
    """Load LoRA configuration from adapter_config.json."""
    config_path = os.path.join(lora_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def enable_lora_for_model(model: nn.Module, lora_config: dict):
    """Enable LoRA for existing linear layers in the model."""
    r = lora_config.get('r', 16)
    lora_alpha = lora_config.get('lora_alpha', 32.0)
    lora_dropout = lora_config.get('lora_dropout', 0.0)
    target_modules = lora_config.get('target_modules', [])
    
    for name, module in model.named_modules():
        if hasattr(module, '__init_lora__'):
            should_apply = True
            if target_modules:
                leaf = name.split('.')[-1] if name else name
                should_apply = any(target == leaf for target in target_modules)
            if should_apply:
                module.__init_lora__(r, lora_alpha, lora_dropout)
    return model


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _load_gptq_awq_weights(model: nn.Module, config: Config):
    """Load GPTQ/AWQ offline quantized weights from checkpoint.
    
    Args:
        model: Model module
        config: Config with model path
        
    Returns:
        Tuple of (loaded_gptq_count, loaded_awq_count, skipped_count)
    """
    loaded_gptq = 0
    loaded_awq = 0
    skipped = 0
    
    # Check if model is configured for GPTQ or AWQ
    weight_attn_dtype = getattr(config, "linear_attn_weight_dtype", "bf16") or "bf16"
    weight_mlp_dtype = getattr(config, "linear_mlp_weight_dtype", "bf16") or "bf16"
    quantize_cfg = _read_quantize_config(getattr(config, "model", ""))
    checkpoint_format = (quantize_cfg.get("checkpoint_format") or "").strip().lower()
    ckpt_bits = int(quantize_cfg.get("bits", 0) or 0)
    ckpt_group_size = int(quantize_cfg.get("group_size", 0) or 0)
    
    # NOTE: marlin variants reuse the same offline GPTQ/AWQ checkpoint keys
    # (qweight/qzeros/scales[/g_idx]) and are repacked lazily in `LinearBase`
    # on first forward.
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
    
    all_files = list(glob(os.path.join(config.model, "*.safetensors")))

    # Scan keys once and remember which file contains each key.
    # This avoids the O(num_modules * num_files) "search every file for every module" pattern below.
    key_to_file: dict[str, str] = {}
    module_keys: dict[str, dict[str, str]] = {}
    offline_suffixes = (".qweight", ".qzeros", ".scales", ".g_idx")
    for file in all_files:
        with safe_open(file, "pt", "cpu") as f:
            for key in f.keys():
                if not key.endswith(offline_suffixes):
                    continue
                key_to_file[key] = file
                # Group by module prefix: {prefix}.qweight, {prefix}.qzeros, {prefix}.scales, {prefix}.g_idx (GPTQ only)
                if key.endswith(".qweight"):
                    prefix = key[:-8]
                    module_keys.setdefault(prefix, {})["qweight"] = key
                elif key.endswith(".qzeros"):
                    prefix = key[:-7]
                    module_keys.setdefault(prefix, {})["qzeros"] = key
                elif key.endswith(".scales"):
                    prefix = key[:-7]
                    module_keys.setdefault(prefix, {})["scales"] = key
                else:  # .g_idx
                    prefix = key[:-6]
                    module_keys.setdefault(prefix, {})["g_idx"] = key

    # Cache modules lookup to avoid rebuilding dict(model.named_modules()) repeatedly.
    named_modules = dict(model.named_modules())
    offline_capable_modules: dict[str, nn.Module] = {
        name: m for name, m in named_modules.items() if hasattr(m, "set_offline_quantized_weight")
    }
    
    def _find_offline_capable_module(module_name: str) -> nn.Module | None:
        """Best-effort resolve module_name to a module with offline quant support."""
        m = offline_capable_modules.get(module_name)
        if m is not None:
            return m

        # Try a few naming fallbacks (keep behavior compatible with the previous implementation).
        leaf = module_name.split(".")[-1] if module_name else module_name
        for name, cand in offline_capable_modules.items():
            if (
                name == module_name
                or name.endswith("." + module_name)
                or module_name.endswith("." + name)
                or (name.split(".")[-1] == leaf)
            ):
                return cand
        return None

    def _load_tensors_for_prefix(key_dict: dict[str, str], *, want_g_idx: bool) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
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

    # Load GPTQ/AWQ weights for each module
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    for prefix, key_dict in module_keys.items():
        if "qweight" not in key_dict or "qzeros" not in key_dict or "scales" not in key_dict:
            continue  # Skip incomplete sets
        
        # Map prefix to module name
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
            
            # Determine format: check if g_idx exists (GPTQ) or not (AWQ)
            has_g_idx = "g_idx" in key_dict
            is_gptq_keyset = has_g_idx or is_gptq_marlin_ckpt
            if is_gptq_keyset and use_gptq:
                format = "gptq"
            elif (not is_gptq_keyset) and use_awq:
                format = "awq"
            else:
                # Prefer GPTQ if both are enabled and g_idx exists
                format = "gptq" if (use_gptq and is_gptq_keyset) else ("awq" if use_awq else None)
            
            if format is None:
                skipped += 1
                continue
            
            # Load tensors from the minimal set of safetensors files.
            qweight, qzeros, scales, g_idx = _load_tensors_for_prefix(
                key_dict, want_g_idx=(format == "gptq")
            )
            
            if qweight is None or qzeros is None or scales is None:
                skipped += 1
                continue
            
            # Infer dimensions from tensor shapes (vLLM standard format) WITHOUT
            # assuming bits=4. This enables GPTQ W2/W4/W8 checkpoints.
            if format == "gptq":
                if is_gptq_marlin_ckpt:
                    # gptq_marlin export uses Marlin repacked qweight/scales layouts.
                    # Empirically (vLLM marlin): qweight is packed on K in tiles of 16,
                    # so qweight.shape[0] == in_features / 16; and scales carries original N.
                    out_features = int(scales.shape[1]) if scales.ndim == 2 else int(qweight.shape[1])
                    in_features = int(qweight.shape[0]) * 16
                    if ckpt_bits not in (4, 8):
                        logger.warning(
                            f"gptq_marlin requires bits=4/8, got bits={ckpt_bits} for {module_name}. Skipping."
                        )
                        skipped += 1
                        continue
                    # Keep pack_factor for dummy qzeros creation later.
                    pack_factor = 32 // int(ckpt_bits)
                else:
                    # Standard GPTQ: qweight [K/pack, N]
                    out_features = int(qweight.shape[1])
                    # qzeros: [K/group, N/pack] (may be empty for some checkpoints)
                    if getattr(qzeros, "numel", lambda: 1)() == 0:
                        if ckpt_bits not in (2, 4, 8):
                            logger.warning(
                                f"qzeros is empty and cannot infer bits for {module_name}. "
                                "Please ensure quantize_config.json contains bits (2/4/8). Skipping."
                            )
                            skipped += 1
                            continue
                        pack_factor = 32 // int(ckpt_bits)
                    else:
                        if int(qzeros.shape[1]) <= 0 or out_features % int(qzeros.shape[1]) != 0:
                            logger.warning(
                                f"Cannot infer GPTQ pack_factor from qzeros for {module_name}: "
                                f"qzeros.shape={tuple(qzeros.shape)}, qweight.shape={tuple(qweight.shape)}. Skipping."
                            )
                            skipped += 1
                            continue
                        pack_factor = out_features // int(qzeros.shape[1])  # 32 / bits
                    in_features = int(qweight.shape[0]) * pack_factor
            else:
                # awq: qweight: [K, N/pack], scales: [K/group, N]
                out_features = int(scales.shape[1]) if scales.ndim == 2 else int(qweight.shape[1])
                if int(qweight.shape[1]) <= 0 or out_features % int(qweight.shape[1]) != 0:
                    logger.warning(
                        f"Cannot infer AWQ pack_factor from scales/qweight for {module_name}: "
                        f"scales.shape={tuple(scales.shape)}, qweight.shape={tuple(qweight.shape)}. Skipping."
                    )
                    skipped += 1
                    continue
                pack_factor = out_features // int(qweight.shape[1])  # 32 / bits (expected 8 for AWQ 4-bit)
                in_features = int(qweight.shape[0])

            # Infer group_size from qzeros/scales.
            # qzeros/scales are groupwise on K (in_features).
            group_size = 128
            if ckpt_group_size not in (0, None):
                # quantize_config.json stores actual group_size (may be -1)
                group_size = int(ckpt_group_size)
            else:
                if is_gptq_marlin_ckpt and len(scales.shape) == 2 and int(scales.shape[0]) > 0:
                    # vLLM marlin_permute_scales keeps shape [num_groups, N] for most cases.
                    # Some older/alternate layouts may use [2*num_groups, N/2].
                    num_groups = int(scales.shape[0])
                    if num_groups > 0 and in_features % num_groups == 0:
                        group_size = in_features // num_groups
                    elif num_groups % 2 == 0 and (in_features % (num_groups // 2)) == 0:
                        # Fallback for legacy 2*num_groups layouts.
                        group_size = in_features // (num_groups // 2)
                else:
                    num_groups = int(qzeros.shape[0]) if getattr(qzeros, "numel", lambda: 1)() > 0 else 0
                    if num_groups > 0 and in_features % num_groups == 0:
                        group_size = in_features // num_groups
                    elif len(scales.shape) == 2 and int(scales.shape[0]) > 0 and in_features % int(scales.shape[0]) == 0:
                        group_size = in_features // int(scales.shape[0])

            # For gptq_marlin checkpoints qzeros may be empty; create a shape-compatible dummy
            # packed qzeros so LinearBase considers offline weights present.
            if (
                format == "gptq"
                and getattr(qzeros, "numel", lambda: 1)() == 0
                and (want_gptq_marlin or is_gptq_marlin_ckpt)
                and ckpt_bits in (2, 4, 8)
            ):
                group_size_norm = in_features if group_size == -1 else group_size
                if group_size_norm <= 0 or (in_features % group_size_norm) != 0:
                    logger.warning(
                        f"Invalid group_size={group_size} for {module_name} with in_features={in_features}. Skipping."
                    )
                    skipped += 1
                    continue
                num_groups = in_features // group_size_norm
                try:
                    qzeros = _make_packed_qzeros_constant(
                        num_groups=num_groups,
                        out_features=out_features,
                        bits=int(ckpt_bits),
                        device=qweight.device,
                    )
                except Exception as e:
                    logger.warning(f"Failed to create dummy qzeros for {module_name}: {e}. Skipping.")
                    skipped += 1
                    continue
            
            # Handle tensor parallel sharding (TP>1).
            # ColumnParallelLinear: tp_dim=0 (shard N/out_features)
            # RowParallelLinear   : tp_dim=1 (shard K/in_features)
            tp_size = int(getattr(module, "tp_size", 1) or 1)
            tp_rank = int(getattr(module, "tp_rank", 0) or 0)
            tp_dim = getattr(module, "tp_dim", None)
            if tp_size > 1:
                if tp_dim not in (0, 1):
                    logger.warning(
                        f"Unsupported tp_dim={tp_dim} for offline quantized weights. Skipping {module_name}."
                    )
                    skipped += 1
                    continue

                # Shard along output features (N) for column-parallel modules.
                if tp_dim == 0:
                    if out_features % tp_size != 0:
                        logger.warning(
                            f"out_features={out_features} not divisible by TP={tp_size} for {module_name}. "
                            "Skipping offline quant weights for this module."
                        )
                        skipped += 1
                        continue
                    out_per = out_features // tp_size
                    out_start = tp_rank * out_per
                    out_end = out_start + out_per
                    if out_per % pack_factor != 0:
                        logger.warning(
                            f"out_features_per_partition={out_per} not divisible by pack_factor={pack_factor} "
                            f"for {module_name}. Skipping."
                        )
                        skipped += 1
                        continue
                    out_packed_per = out_per // pack_factor
                    out_packed_start = out_start // pack_factor
                    out_packed_end = out_packed_start + out_packed_per

                    if format == "gptq":
                        if is_gptq_marlin_ckpt:
                            # Marlin qweight packs N by a factor (bits/2): N_packed = N * (bits/2)
                            n_factor = int(ckpt_bits) // 2
                            if n_factor <= 0:
                                logger.warning(
                                    f"invalid gptq_marlin n_factor for bits={ckpt_bits} ({module_name}). Skipping."
                                )
                                skipped += 1
                                continue
                            qweight = qweight[:, (out_start * n_factor):(out_end * n_factor)]
                            # scales keep original N
                            scales = scales[:, out_start:out_end]
                            # qzeros stays dummy/empty; g_idx stays on K.
                            out_features = out_per
                        else:
                            # qweight: [K/pack, N]
                            qweight = qweight[:, out_start:out_end]
                            # qzeros: [K/group, N/pack]
                            qzeros = qzeros[:, out_packed_start:out_packed_end]
                            # scales: [K/group, N]
                            scales = scales[:, out_start:out_end]
                            out_features = out_per
                    else:
                        # awq qweight: [K, N/pack]
                        qweight = qweight[:, out_packed_start:out_packed_end]
                        qzeros = qzeros[:, out_packed_start:out_packed_end]
                        scales = scales[:, out_start:out_end]
                        out_features = out_per

                # Shard along input features (K) for row-parallel modules.
                elif tp_dim == 1:
                    if in_features % tp_size != 0:
                        logger.warning(
                            f"in_features={in_features} not divisible by TP={tp_size} for {module_name}. "
                            "Skipping offline quant weights for this module."
                        )
                        skipped += 1
                        continue
                    in_per = in_features // tp_size
                    in_start = tp_rank * in_per
                    in_end = in_start + in_per
                    if group_size <= 0 or (in_per % group_size) != 0 or (in_start % group_size) != 0:
                        logger.warning(
                            f"group_size={group_size} incompatible with TP sharding for {module_name} "
                            f"(in_per={in_per}, in_start={in_start}). Skipping."
                        )
                        skipped += 1
                        continue
                    g_start = in_start // group_size
                    g_end = in_end // group_size

                    if format == "gptq":
                        if is_gptq_marlin_ckpt:
                            # Marlin qweight packs K in tiles of 16: K_packed = K / 16
                            if in_start % 16 != 0:
                                logger.warning(
                                    f"gptq_marlin requires in_start divisible by 16, got in_start={in_start} "
                                    f"for {module_name}. Skipping."
                                )
                                skipped += 1
                                continue
                            q_start = in_start // 16
                            q_end = in_end // 16
                            qweight = qweight[q_start:q_end, :]
                            # Shard scales on group dimension (K/group).
                            # vLLM marlin_permute_scales typically returns [num_groups, N].
                            group_size_norm = in_features if group_size == -1 else group_size
                            expected_num_groups = in_features // group_size_norm if group_size_norm > 0 else 0
                            if expected_num_groups <= 0:
                                logger.warning(
                                    f"invalid expected_num_groups={expected_num_groups} for {module_name}. Skipping."
                                )
                                skipped += 1
                                continue
                            if int(scales.shape[0]) == expected_num_groups:
                                scales = scales[g_start:g_end, :]
                            elif int(scales.shape[0]) == 2 * expected_num_groups:
                                # Legacy/alternate layout: [2*num_groups, N/2]
                                scales = scales[(2 * g_start):(2 * g_end), :]
                            else:
                                logger.warning(
                                    f"unexpected gptq_marlin scales.shape[0]={int(scales.shape[0])} "
                                    f"(expected {expected_num_groups} or {2*expected_num_groups}) for {module_name}. Skipping."
                                )
                                skipped += 1
                                continue
                            if g_idx is not None and getattr(g_idx, "numel", lambda: 1)() > 0:
                                g_idx = g_idx[in_start:in_end]
                            in_features = in_per
                        else:
                            # qweight: [K/pack, N] (packed on K)
                            if in_start % pack_factor != 0:
                                logger.warning(
                                    f"in_start={in_start} not divisible by pack_factor={pack_factor} "
                                    f"for {module_name}. Skipping."
                                )
                                skipped += 1
                                continue
                            q_start = in_start // pack_factor
                            q_end = in_end // pack_factor
                            qweight = qweight[q_start:q_end, :]
                            qzeros = qzeros[g_start:g_end, :]
                            scales = scales[g_start:g_end, :]
                            if g_idx is not None and getattr(g_idx, "numel", lambda: 1)() > 0:
                                g_idx = g_idx[in_start:in_end]
                            in_features = in_per
                    else:
                        # awq qweight: [K, N/pack]
                        qweight = qweight[in_start:in_end, :]
                        qzeros = qzeros[g_start:g_end, :]
                        scales = scales[g_start:g_end, :]
                        in_features = in_per
            
            # Treat empty g_idx as "not provided" for GPTQ (desc_act=False checkpoints often store empty).
            if g_idx is not None and getattr(g_idx, "numel", lambda: 1)() == 0:
                g_idx = None

            # Set offline quantized weight
            try:
                if format == "gptq" and is_gptq_marlin_ckpt:
                    if ckpt_bits not in (4, 8):
                        raise ValueError(f"gptq_marlin checkpoint requires bits=4/8, got bits={ckpt_bits}")
                    _set_offline_gptq_marlin_weight(
                        module,
                        qweight=qweight,
                        scales=scales,
                        out_features=out_features,
                        in_features=in_features,
                        group_size=group_size,
                        bits=int(ckpt_bits),
                        g_idx=g_idx,
                    )
                else:
                    module.set_offline_quantized_weight(
                        format=format,
                        qweight=qweight,
                        qzeros=qzeros,
                        scales=scales,
                        out_features=out_features,
                        in_features=in_features,
                        group_size=group_size,
                        g_idx=g_idx,
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


def load_model(model: nn.Module, config: Config):
    """Load model weights and optionally LoRA weights."""
    # Enable LoRA for linear layers if LoRA is enabled
    if config.use_lora and config.lora_path:
        lora_config = load_lora_config(config.lora_path)
        if lora_config:
            logger.info(f"LoRA Config Loaded: {lora_config}")
            model = enable_lora_for_model(model, lora_config)
        else:
            logger.info("No adapter_config.json found, using default LoRA parameters")
            default_config = {'r': 16, 'lora_alpha': 32.0, 'lora_dropout': 0.0}
            model = enable_lora_for_model(model, default_config)
    
    # First, try to load offline quantized weights (GPTQ/AWQ)
    loaded_gptq, loaded_awq, skipped_offline = _load_gptq_awq_weights(model, config)
    if loaded_gptq > 0 or loaded_awq > 0:
        print(f"Loaded offline quantized weights: GPTQ={loaded_gptq}, AWQ={loaded_awq}, skipped={skipped_offline}")
    
    # Load base model weights (only for non-offline-quantized layers)
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in tqdm(glob(os.path.join(config.model, "*.safetensors")), desc="Loading base model"):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Skip GPTQ/AWQ keys (already loaded)
                if any(
                    weight_name.endswith(suffix)
                    for suffix in [".qweight", ".qzeros", ".scales", ".g_idx"]
                ):
                    continue
                
                for k in packed_modules_mapping:
                    if k in weight_name:
                        
                        if config.model_name == "llada" and k == "ff_out" and "transformer.ff_out" in weight_name:
                            continue
                        elif config.model_name == "llada" and k == "transformer.ff_out":
                            v, shard_id = packed_modules_mapping[k]
                            assert v == "lm_head"
                            param_name = "lm_head.weight"
                        else:
                            v, shard_id = packed_modules_mapping[k]
                            param_name = weight_name.replace(k, v)
                            
                        if "layernorm" in param_name:
                            try:
                                param = model.get_parameter(param_name)
                                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                                weight_loader(param, f.get_tensor(weight_name))
                            except (AttributeError, KeyError):
                                # Try buffer fallback for non-parameter weights
                                try:
                                    buffer = model.get_buffer(param_name)
                                    buffer.copy_(f.get_tensor(weight_name))
                                except (AttributeError, KeyError):
                                    pass
                        else:
                            try:
                                param = model.get_parameter(param_name)
                                weight_loader = partial(getattr(param, "weight_loader"), param, f.get_tensor(weight_name)) 
                                if shard_id is None:
                                    weight_loader()
                                else:
                                    weight_loader(shard_id)
                            except (AttributeError, KeyError):
                                # Parameter might not exist if offline quantized weights were loaded
                                # Skip it silently
                                pass
                        break
                else:
                    try:
                        param = model.get_parameter(weight_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name))
                    except (AttributeError, KeyError):
                        # Try buffer fallback for non-parameter weights
                        try:
                            buffer = model.get_buffer(weight_name)
                            buffer.copy_(f.get_tensor(weight_name))
                        except (AttributeError, KeyError):
                            pass
    
    # Load LoRA weights if enabled
    if config.use_lora and config.lora_path:
        if os.path.exists(config.lora_path):
            logger.info(f"Loading LoRA weights from {config.lora_path}")
            load_lora_weights_fn = partial(load_lora_weights, model, config.lora_path)
            packed_modules_mapping = packed_modules_mapping if config.model_name == "llada" else None
            model = load_lora_weights_fn(packed_modules_mapping=packed_modules_mapping)
        else:
            logger.warning(f"LoRA path {config.lora_path} does not exist, skipping LoRA loading")
    
    return model


def load_lora_weights(model: nn.Module, lora_path: str, packed_modules_mapping: dict | None = None):
    """Load LoRA weights into LoRA-enabled layers."""
    try:
        lora_config = load_lora_config(lora_path)
        target_modules = lora_config.get('target_modules', [])
        
        lora_weights = {}
        
        for file in tqdm(glob(os.path.join(lora_path, "*.safetensors")), desc="Loading LoRA"):
            with safe_open(file, "pt", "cpu") as f:
                for weight_name in f.keys():
                    lora_weights[weight_name] = f.get_tensor(weight_name)
        
        applied_count = 0

        modified_modules = None
        if packed_modules_mapping is not None:
            modified_modules = [v for k, (v, _) in packed_modules_mapping.items() if k in target_modules]
            rev_mapping = {v: k for k, (v, _) in packed_modules_mapping.items()}
            
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                should_apply = True
                
                if modified_modules is not None:
                    modified_module_type = '.'.join(name.split('.')[-2:])  
                    org_module_type = rev_mapping[modified_module_type]
                    org_name = name.replace(modified_module_type, org_module_type)
                    should_apply = any(target in modified_module_type for target in modified_modules)
                elif target_modules:
                    module_type = name.split('.')[-1] if '.' in name else name
                    should_apply = any(target in module_type for target in target_modules)
                 
                if not should_apply:
                    continue
                
                base_patterns = [
                    name,
                    f"base_model.model.{name}",
                    f"model.{name}",
                ] if modified_modules is None else [
                    org_name,
                    f"base_model.model.{org_name}",
                    f"model.{org_name}",
                ]
                
                found_a = found_b = None
                for base_name in base_patterns:
                    lora_a_keys = [
                        f"{base_name}.lora_A.weight",
                        f"{base_name}.lora_A.default.weight",
                        f"{base_name}.lora_A",
                    ]
                    lora_b_keys = [
                        f"{base_name}.lora_B.weight", 
                        f"{base_name}.lora_B.default.weight",
                        f"{base_name}.lora_B",
                    ]
                    
                    for key in lora_a_keys:
                        if key in lora_weights:
                            found_a = lora_weights[key]
                            break
                    for key in lora_b_keys:
                        if key in lora_weights:
                            found_b = lora_weights[key]
                            break
                    
                    if found_a is not None and found_b is not None:
                        break
                
                if found_a is not None and found_b is not None:
                    if hasattr(module, 'tp_size') and module.tp_size > 1:
                        if hasattr(module, 'tp_dim') and module.tp_dim == 0:
                            shard_size = found_b.size(0) // module.tp_size
                            start_idx = module.tp_rank * shard_size
                            found_b = found_b[start_idx:start_idx + shard_size]
                        elif hasattr(module, 'tp_dim') and module.tp_dim == 1:
                            shard_size = found_a.size(1) // module.tp_size
                            start_idx = module.tp_rank * shard_size
                            found_a = found_a[:, start_idx:start_idx + shard_size]
                    
                    try:
                        module.lora_A.data.copy_(found_a)
                        module.lora_B.data.copy_(found_b)
                        applied_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to load LoRA weights for {name}: {e}")
        
        for module in model.modules():
            if hasattr(module, 'merge_lora'):
                module.merge_lora()
        
        logger.info(f"LoRA weights applied to {applied_count} layers and merged")
        
    except Exception as e:
        logger.error(f"Error loading LoRA weights: {e}")
        logger.warning("Continuing with base model only")
    
    return model
