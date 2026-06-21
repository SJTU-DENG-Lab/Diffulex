from __future__ import annotations

from dataclasses import dataclass

from diffulex.engine.strategy_config_registry import StrategyConfigRegistry
from diffulex.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DMaxStrategyConfig:
    token_merge_mode: str
    token_merge_top_k: int
    token_merge_renormalize: bool
    token_merge_weight: float
    dmax_sampler_fast_path: bool
    dmax_force_prefill_active: bool
    multi_block_prefix_full: bool = False


@StrategyConfigRegistry.register("dmax")
def normalize_dmax_config(config) -> DMaxStrategyConfig:
    if config.multi_block_prefix_full:
        logger.warning("Forcing multi_block_prefix_full=False for decoding_strategy=dmax.")
    config.multi_block_prefix_full = False
    if config.enable_prefix_caching:
        logger.info("Enabling prefix caching for decoding_strategy=dmax.")
    return DMaxStrategyConfig(
        token_merge_mode=config.token_merge_mode,
        token_merge_top_k=config.token_merge_top_k,
        token_merge_renormalize=config.token_merge_renormalize,
        token_merge_weight=config.token_merge_weight,
        dmax_sampler_fast_path=config.dmax_sampler_fast_path,
        dmax_force_prefill_active=config.dmax_force_prefill_active,
    )
