from __future__ import annotations

import os

from dataclasses import dataclass

from diffulex.engine.strategy_config_registry import StrategyConfigRegistry
from diffulex.logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class FastDLLMV2StrategyConfig:
    name: str = "fast_dllm_v2"
    sub_block_size: int = 8
    block_size: int = 32
    use_block_cache: bool = True


@StrategyConfigRegistry.register("fast_dllm_v2")
def normalize_fast_dllm_v2_config(config) -> FastDLLMV2StrategyConfig:
    if config.block_size * config.buffer_size != 32:
        logger.warning(
            "Fast-dLLM v2 paper defaults map Diffulex block_size * buffer_size to 32; "
            "got block_size=%s, buffer_size=%s.",
            config.block_size,
            config.buffer_size,
        )
    if config.buffer_size <= 1:
        raise ValueError("decoding_strategy='fast_dllm_v2' requires buffer_size > 1.")

    if config.multi_block_prefix_full:
        logger.warning("Forcing multi_block_prefix_full=False for decoding_strategy=fast_dllm_v2.")
    config.multi_block_prefix_full = False

    if not config.enable_prefix_caching:
        logger.info("Enabling prefix caching for decoding_strategy=fast_dllm_v2.")
    config.enable_prefix_caching = True

    use_block_cache = bool(getattr(config, "fdv2_use_block_cache", True))
    env_value = os.environ.get("DIFFULEX_FDV2_USE_BLOCK_CACHE")
    if env_value is not None:
        use_block_cache = env_value.strip().lower() not in {"0", "false", "no", "off"}

    return FastDLLMV2StrategyConfig(
        sub_block_size=int(config.block_size),
        block_size=int(config.block_size) * int(config.buffer_size),
        use_block_cache=use_block_cache,
    )
