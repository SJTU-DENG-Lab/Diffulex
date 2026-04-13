from __future__ import annotations

from typing import TYPE_CHECKING

from diffulex.config import Config
from diffulex.engine.kv_cache_manager import AutoKVCacheManager
from diffulex.strategy_template.multi_block.engine.kv_cache_manager import MultiBlockKVCacheManagerTemplate

if TYPE_CHECKING:
    from diffulex.engine.request import DllmReq


@AutoKVCacheManager.register("multi_bd", is_default=True)
class MultiBDKVCacheManager(MultiBlockKVCacheManagerTemplate):
    def __init__(self, config: Config):
        super().__init__(config)

    def can_append(self, req: "DllmReq") -> bool:
        return self.can_append_multi_block(req)

    def may_append(self, req: "DllmReq") -> None:
        self.may_append_multi_block(req)
