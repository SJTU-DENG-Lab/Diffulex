from __future__ import annotations

from diffulex.config import Config
from diffulex.engine.kv_cache_manager import AutoKVCacheManager, KVCacheManagerBase


@AutoKVCacheManager.register("fast_dllm_v2")
class FastDLLMV2KVCacheManager(KVCacheManagerBase):
    def __init__(self, config: Config):
        super().__init__(config)

    def _missing_cache_pages(self, req) -> int:
        if getattr(req, "is_decoding", False) and hasattr(req, "fdv2_read_cache_pages"):
            return max(0, int(req.fdv2_read_cache_pages) - len(req.page_table))
        return super()._missing_cache_pages(req)

    def may_append(self, req) -> None:
        if not (getattr(req, "is_decoding", False) and hasattr(req, "fdv2_read_cache_pages")):
            return super().may_append(req)

        missing_pages = self._missing_cache_pages(req)
        if missing_pages > len(self.free_page_ids):
            raise RuntimeError(
                "Insufficient free KV cache pages for Fast-dLLM v2 current block: "
                f"missing_pages={missing_pages}, free_pages={len(self.free_page_ids)}, "
                f"fdv2_read_cache_len={req.fdv2_read_cache_len}, req_id={req.req_id}"
            )

        for _ in range(missing_pages):
            page_id = self.free_page_ids[0]
            self._allocate_page(page_id)
            req.page_table.append(page_id)
