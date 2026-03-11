"""LoRA weight loading utilities."""

from __future__ import annotations

import os
from glob import glob

import torch
import torch.nn as nn
from safetensors import safe_open
from tqdm import tqdm

from diffulex.logger import get_logger
from .core import load_lora_config

logger = get_logger(__name__)


def load_lora_weights(
    model: nn.Module,
    lora_path: str,
    packed_modules_mapping: dict | None = None,
    pre_merge_lora: bool = False,
):
    """Load LoRA weights into LoRA-enabled layers.

    Args:
        model: The model with LoRA-enabled linear layers.
        lora_path: Path to LoRA checkpoint.
        packed_modules_mapping: Optional mapping for packed modules (e.g. llada lm_head).
        pre_merge_lora: If True, merge LoRA into base weights after loading.
    """
    try:
        lora_config = load_lora_config(lora_path)
        target_modules = lora_config.get("target_modules", [])

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
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                should_apply = True

                if modified_modules is not None:
                    modified_module_type = ".".join(name.split(".")[-2:])
                    org_module_type = rev_mapping[modified_module_type]
                    org_name = name.replace(modified_module_type, org_module_type)
                    should_apply = any(target in modified_module_type for target in modified_modules)
                elif target_modules:
                    module_type = name.split(".")[-1] if "." in name else name
                    should_apply = any(target in module_type for target in target_modules)

                if not should_apply:
                    continue

                base_patterns = (
                    [name, f"base_model.model.{name}", f"model.{name}"]
                    if modified_modules is None
                    else [org_name, f"base_model.model.{org_name}", f"model.{org_name}"]
                )

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
                    if hasattr(module, "tp_size") and module.tp_size > 1:
                        if hasattr(module, "tp_dim") and module.tp_dim == 0:
                            shard_size = found_b.size(0) // module.tp_size
                            start_idx = module.tp_rank * shard_size
                            found_b = found_b[start_idx : start_idx + shard_size]
                        elif hasattr(module, "tp_dim") and module.tp_dim == 1:
                            shard_size = found_a.size(1) // module.tp_size
                            start_idx = module.tp_rank * shard_size
                            found_a = found_a[:, start_idx : start_idx + shard_size]

                    try:
                        module.lora_A.data.copy_(found_a)
                        module.lora_B.data.copy_(found_b)
                        applied_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to load LoRA weights for {name}: {e}")

        if pre_merge_lora:
            for m in model.modules():
                if hasattr(m, "merge_lora"):
                    m.merge_lora()
            logger.info(f"LoRA weights applied to {applied_count} layers and merged into base")
        else:
            logger.info(f"LoRA weights applied to {applied_count} layers (unmerged)")

    except Exception as e:
        logger.error(f"Error loading LoRA weights: {e}")
        logger.warning("Continuing with base model only")

    return model
