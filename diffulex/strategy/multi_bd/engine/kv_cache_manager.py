from __future__ import annotations

from diffulex.config import Config
from diffulex.engine.kv_cache_manager import AutoKVCacheManager, KVCacheManagerBase


@AutoKVCacheManager.register("multi_bd", is_default=True)
class MultiBDKVCacheManager(KVCacheManagerBase):
    def __init__(self, config: Config):
        super().__init__(config)
