from __future__ import annotations

from diffulex.config import Config
from diffulex.engine.kv_cache_manager import AutoKVCacheManager
from diffulex.strategy.templates.token_merge.engine.kv_cache_manager import (
    TokenMergeKVCacheManagerTemplate,
)


@AutoKVCacheManager.register("dmax")
class DMaxKVCacheManager(TokenMergeKVCacheManagerTemplate):
    def __init__(self, config: Config):
        super().__init__(config)
