from __future__ import annotations

from dataclasses import dataclass

from diffulex.engine.strategy_config_registry import StrategyConfigRegistry
from diffulex.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MultiBDStrategyConfig:
    multi_block_prefix_full: bool = False


@StrategyConfigRegistry.register("multi_bd")
def normalize_multi_bd_config(config) -> MultiBDStrategyConfig:
    if config.multi_block_prefix_full:
        logger.warning("Forcing multi_block_prefix_full=False for decoding_strategy=multi_bd.")
    config.multi_block_prefix_full = False
    if config.enable_prefix_caching:
        logger.info("Enabling prefix caching for decoding_strategy=multi_bd.")
    return MultiBDStrategyConfig()
