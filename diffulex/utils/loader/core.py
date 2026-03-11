"""Core loading utilities."""

from __future__ import annotations

import os
import json
from functools import partial
from glob import glob

import torch
import torch.nn as nn
from safetensors import safe_open
from tqdm import tqdm

from diffulex.config import Config
from diffulex.logger import get_logger

logger = get_logger(__name__)


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


def load_lora_config(lora_path: str) -> dict:
    """Load LoRA configuration from adapter_config.json."""
    config_path = os.path.join(lora_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def enable_lora_for_model(model: nn.Module, lora_config: dict):
    """Enable LoRA for existing linear layers in the model."""
    r = lora_config.get("r", 16)
    lora_alpha = lora_config.get("lora_alpha", 32.0)
    lora_dropout = lora_config.get("lora_dropout", 0.0)
    target_modules = lora_config.get("target_modules", [])

    for name, module in model.named_modules():
        if hasattr(module, "__init_lora__"):
            should_apply = True
            if target_modules:
                leaf = name.split(".")[-1] if name else name
                should_apply = any(target == leaf for target in target_modules)
            if should_apply:
                module.__init_lora__(r, lora_alpha, lora_dropout)
    return model


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """Default weight loader that simply copies data."""
    param.data.copy_(loaded_weight)


def load_base_weights(model: nn.Module, config: Config, loaded_quant_keys: set[str]) -> None:
    """Load base model weights (non-quantized)."""
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    for file in tqdm(glob(os.path.join(config.model, "*.safetensors")), desc="Loading base model"):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Skip already loaded quantized keys
                if weight_name in loaded_quant_keys:
                    continue

                handled = False
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
                                try:
                                    buffer = model.get_buffer(param_name)
                                    buffer.copy_(f.get_tensor(weight_name))
                                except (AttributeError, KeyError):
                                    pass
                        else:
                            try:
                                param = model.get_parameter(param_name)
                                weight_loader = partial(
                                    getattr(param, "weight_loader"),
                                    param, f.get_tensor(weight_name)
                                )
                                if shard_id is None:
                                    weight_loader()
                                else:
                                    weight_loader(shard_id)
                            except (AttributeError, KeyError):
                                pass
                        handled = True
                        break
                        
                if not handled:
                    try:
                        param = model.get_parameter(weight_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name))
                    except (AttributeError, KeyError):
                        try:
                            buffer = model.get_buffer(weight_name)
                            buffer.copy_(f.get_tensor(weight_name))
                        except (AttributeError, KeyError):
                            pass
