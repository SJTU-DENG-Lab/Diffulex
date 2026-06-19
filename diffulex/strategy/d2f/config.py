from __future__ import annotations

from dataclasses import dataclass

from diffulex.engine.strategy_config_registry import StrategyConfigRegistry
from diffulex.logger import get_logger

logger = get_logger(__name__)


@dataclass
class D2FStrategyConfig:
    multi_block_prefix_full: bool = True
    enable_prefix_caching: bool = False


@StrategyConfigRegistry.register("d2f", is_default=True)
def normalize_d2f_config(config) -> D2FStrategyConfig:
    if not config.multi_block_prefix_full:
        logger.warning("Forcing multi_block_prefix_full=True for decoding_strategy=d2f.")
    if config.enable_prefix_caching:
        logger.warning("Disabling prefix caching for decoding_strategy=d2f.")
    config.multi_block_prefix_full = True
    config.enable_prefix_caching = False
    return D2FStrategyConfig()
