from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffulex.engine.request import DllmReq
    from diffulex.engine.kv_cache_manager import KVCacheManagerBase


class MultiBlockKVCacheManagerMixin:
    def can_append_multi_block(self: KVCacheManagerBase, req: DllmReq) -> bool:
        return len(self.free_page_ids) * self.page_size >= min(req.to_cache_len, req.chunk_size)

    def may_append_multi_block(self: KVCacheManagerBase, req: DllmReq) -> None:
        if req.cache_len == 0:
            return
        
        page_table = req.page_table
        last_page = self.pages[page_table[-1]]
        allocate_num_pages = req.to_cache_len // self.page_size
        for _ in range(allocate_num_pages):
            if req.cache_len // self.page_size >= len(req.page_table):
                if last_page.hash == -1:
                    prev_end_token_pos = req.in_cache_len - 1
                    prev_rel_page_id = prev_end_token_pos // self.page_size
                    if 0 <= prev_rel_page_id < req.num_pages:
                        token_ids: list[int] = req.page(prev_rel_page_id)
                        prefix = self.pages[page_table[-2]].hash if len(page_table) > 1 else -1
                        h = self.compute_hash(token_ids, prefix)
                        last_page.update(h, token_ids)
                        self.hash_to_page_id[h] = last_page.page_id
                page_id = self.free_page_ids[0]
                self._allocate_page(page_id)
                page_table.append(page_id)
