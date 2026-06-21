from __future__ import annotations

import json
import os
from dataclasses import dataclass

from diffulex.engine.strategy_config_registry import StrategyConfigRegistry
from diffulex.logger import get_logger

logger = get_logger(__name__)

GEMMA_BLOCK_SIZE = 256


@dataclass
class DiffusionGemmaStrategyConfig:
    max_denoising_steps: int = 48
    stability_threshold: int = 2
    t_min: float = 0.0
    t_max: float = 1.0
    confidence_threshold: float = 0.1
    entropy_bound: float = 1.0
    canvas_length: int = GEMMA_BLOCK_SIZE


def _load_generation_config(config) -> None:
    path = os.path.join(config.model, "generation_config.json")
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            gen_config = json.load(f)
    except Exception as exc:
        logger.warning("Failed to load DiffusionGemma generation_config.json: %s", exc)
        return

    defaults = {
        "diffusion_gemma_max_denoising_steps": 48,
        "diffusion_gemma_stability_threshold": 2,
        "diffusion_gemma_t_min": 0.0,
        "diffusion_gemma_t_max": 1.0,
        "diffusion_gemma_confidence_threshold": 0.1,
        "diffusion_gemma_entropy_bound": 1.0,
    }
    mapping = {
        "max_denoising_steps": "diffusion_gemma_max_denoising_steps",
        "stability_threshold": "diffusion_gemma_stability_threshold",
        "t_min": "diffusion_gemma_t_min",
        "t_max": "diffusion_gemma_t_max",
        "confidence_threshold": "diffusion_gemma_confidence_threshold",
    }
    for src, dst in mapping.items():
        if src in gen_config and getattr(config, dst) == defaults[dst]:
            setattr(config, dst, gen_config[src])

    sampler_config = gen_config.get("sampler_config") or {}
    entropy_bound = sampler_config.get("entropy_bound")
    if (
        entropy_bound is not None
        and config.diffusion_gemma_entropy_bound == defaults["diffusion_gemma_entropy_bound"]
    ):
        config.diffusion_gemma_entropy_bound = float(entropy_bound)


@StrategyConfigRegistry.register("diffusion_gemma")
def normalize_diffusion_gemma_config(config) -> DiffusionGemmaStrategyConfig:
    if config.decoding_strategy != "diffusion_gemma":
        logger.warning("Forcing decoding_strategy='diffusion_gemma' for model_name='diffusion_gemma'.")
        config.decoding_strategy = "diffusion_gemma"
    if config.block_size != GEMMA_BLOCK_SIZE:
        logger.warning(
            "Forcing block_size=%s for model_name='diffusion_gemma' (got %s).",
            GEMMA_BLOCK_SIZE,
            config.block_size,
        )
        config.block_size = GEMMA_BLOCK_SIZE
    if config.page_size != GEMMA_BLOCK_SIZE:
        logger.warning(
            "Forcing page_size=%s for model_name='diffusion_gemma' (got %s).",
            GEMMA_BLOCK_SIZE,
            config.page_size,
        )
        config.page_size = GEMMA_BLOCK_SIZE
    if config.buffer_size != 1:
        logger.warning("Forcing buffer_size=1 for model_name='diffusion_gemma' (got %s).", config.buffer_size)
        config.buffer_size = 1
    if config.multi_block_prefix_full:
        logger.warning("Forcing multi_block_prefix_full=False for decoding_strategy=diffusion_gemma.")
    config.multi_block_prefix_full = False
    if config.enable_prefix_caching:
        logger.info("Enabling prefix caching for decoding_strategy=diffusion_gemma.")

    _load_generation_config(config)

    return DiffusionGemmaStrategyConfig(
        max_denoising_steps=config.diffusion_gemma_max_denoising_steps,
        stability_threshold=config.diffusion_gemma_stability_threshold,
        t_min=config.diffusion_gemma_t_min,
        t_max=config.diffusion_gemma_t_max,
        confidence_threshold=config.diffusion_gemma_confidence_threshold,
        entropy_bound=config.diffusion_gemma_entropy_bound,
        canvas_length=config.block_size,
    )
