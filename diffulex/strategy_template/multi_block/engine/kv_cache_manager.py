from __future__ import annotations

from diffulex.engine.request import DllmReq
from diffulex.engine.kv_cache_manager import KVCacheManagerBase


class MultiBlockKVCacheManagerTemplate(KVCacheManagerBase):
    def _missing_cache_pages_multi_block(self: KVCacheManagerBase, req: DllmReq) -> int:
        total_cache_pages = req.num_pages_with_seq_len(req.cache_len)
        return max(0, total_cache_pages - len(req.page_table))

    def can_append_multi_block(self: KVCacheManagerBase, req: DllmReq) -> bool:
        return len(self.free_page_ids) >= self._missing_cache_pages_multi_block(req)

    def _finalize_last_unhashed_page_multi_block(self: KVCacheManagerBase, req: DllmReq) -> None:
        page_table = req.page_table
        if not page_table:
            return

        last_page = self.pages[page_table[-1]]
        if last_page.hash != -1:
            return

        last_rel_page_id = len(page_table) - 1
        if not (0 <= last_rel_page_id < req.num_pages):
            return

        token_ids: list[int] = req.page(last_rel_page_id)
        prefix = self.pages[page_table[-2]].hash if len(page_table) > 1 else -1
        h = self.compute_hash(token_ids, prefix)
        last_page.update(h, token_ids)
        if self.enable_prefix_caching:
            self.hash_to_page_id[h] = last_page.page_id

    def may_append_multi_block(self: KVCacheManagerBase, req: DllmReq) -> None:
        if req.cache_len == 0:
            return

        page_table = req.page_table
        missing_pages = self._missing_cache_pages_multi_block(req)
        if missing_pages > len(self.free_page_ids):
            raise RuntimeError(
                "Insufficient free KV cache pages for may_append_multi_block: "
                f"missing_pages={missing_pages}, free_pages={len(self.free_page_ids)}, "
                f"cache_len={req.cache_len}, to_cache_len={req.to_cache_len}, "
                f"page_table_len={len(page_table)}, req_id={req.req_id}"
            )

        for _ in range(missing_pages):
            self._finalize_last_unhashed_page_multi_block(req)
            page_id = self.free_page_ids[0]
            self._allocate_page(page_id)
            page_table.append(page_id)
