"""Main model loading entry point."""

import os

import torch
import torch.nn as nn

from diffulex.config import Config
from diffulex.logger import get_logger

from .core import load_lora_config, enable_lora_for_model, load_base_weights
from .offline_quant import load_gptq_awq_weights
from .lora import load_lora_weights

logger = get_logger(__name__)


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
            default_config = {"r": 16, "lora_alpha": 32.0, "lora_dropout": 0.0}
            model = enable_lora_for_model(model, default_config)

    # First, try to load offline quantized weights (GPTQ/AWQ)
    loaded_gptq, loaded_awq, skipped_offline = load_gptq_awq_weights(model, config)
    if loaded_gptq > 0 or loaded_awq > 0:
        print(f"Loaded offline quantized weights: GPTQ={loaded_gptq}, AWQ={loaded_awq}, skipped={skipped_offline}")

    # Collect all quantized keys to skip during base weight loading
    loaded_quant_keys = set()
    for name, module in model.named_modules():
        if hasattr(module, "gptq_qweight") and getattr(module.gptq_qweight, "numel", lambda: 0)() > 0:
            prefix = name.replace(".", "-")  # approximate match
            loaded_quant_keys.update([f"{prefix}.qweight", f"{prefix}.qzeros", f"{prefix}.scales", f"{prefix}.g_idx"])

    # Load base model weights (only for non-offline-quantized layers)
    load_base_weights(model, config, loaded_quant_keys)

    # Load LoRA weights if enabled
    if config.use_lora and config.lora_path:
        if os.path.exists(config.lora_path):
            logger.info(f"Loading LoRA weights from {config.lora_path}")
            packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
            model = load_lora_weights(
                model,
                config.lora_path,
                packed_modules_mapping=packed_modules_mapping if config.model_name == "llada" else None,
                pre_merge_lora=getattr(config, "pre_merge_lora", False),
            )
        else:
            logger.warning(f"LoRA path {config.lora_path} does not exist, skipping LoRA loading")

    return model
