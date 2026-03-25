import xxhash
import weakref

import numpy as np

from typing import Callable
from collections import deque
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from diffulex.config import Config
from diffulex.engine.request import DllmReq
from diffulex.engine.strategy_registry import DiffulexStrategyRegistry
from diffulex.mixin.multi_block.engine.kv_cache_manager import (
    MultiBlockKVCacheManagerMixin,
)


@dataclass
class Page:
    page_id: int
    ref_count: int = 0
    hash: int = -1
    token_ids: list[int] = field(default_factory=list)

    req: DllmReq | None = None

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
        self.req = None

    def set_req(self, req: DllmReq):
        self.req = weakref.ref(req)


class KVCacheManagerBase(ABC, MultiBlockKVCacheManagerMixin):
    def __init__(self, config: Config):
        num_pages = config.num_pages
        page_size = config.page_size
        assert num_pages > 0
        self.config = config
        self.page_size = page_size
        self.pages: list[Page] = [Page(page_id=i) for i in range(num_pages)]
        self.hash_to_page_id: dict[int, int] = dict()
        self.free_page_ids: deque[int] = deque(range(num_pages))
        self.used_page_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()

        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))

        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_page(self, page_id: int) -> Page:
        page = self.pages[page_id]
        assert page.ref_count == 0
        page.reset()
        self.free_page_ids.remove(page_id)
        self.used_page_ids.add(page_id)
        return self.pages[page_id]

    def _free_page(self, page_id: int) -> Page:
        assert self.pages[page_id].ref_count == 0
        self.used_page_ids.remove(page_id)
        self.free_page_ids.append(page_id)

    def can_allocate(self, req: DllmReq) -> bool:
        return len(self.free_page_ids) >= req.num_pages

    def allocate(self, req: DllmReq):
        assert not req.page_table
        req.page_cache_missed.clear()
        req.num_cached_tokens = 0
        h = -1
        cache_miss = False

        for i in range(req.num_pages):
            token_ids = req.page(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.page_size else -1
            page_id = self.hash_to_page_id.get(h, -1)

            if page_id == -1 or self.pages[page_id].token_ids != token_ids:
                cache_miss = True

            req.page_cache_missed.append(cache_miss)

            if cache_miss:
                page_id = self.free_page_ids[0]
                page = self._allocate_page(page_id)
            else:
                req.num_cached_tokens += self.page_size
                if page_id in self.used_page_ids:
                    page = self.pages[page_id]
                    page.ref_count += 1
                else:
                    page = self._allocate_page(page_id)

            if h != -1:
                page.update(h, token_ids)
                self.hash_to_page_id[h] = page_id

            req.page_table.append(page_id)

    def free(self, req: DllmReq):
        for page_id in reversed(req.page_table):
            page = self.pages[page_id]
            page.ref_count -= 1
            if page.ref_count == 0:
                self._free_page(page_id)

        req.num_cached_tokens = 0
        req.page_cache_missed.clear()
        req.page_table.clear()

    @abstractmethod
    def can_append(self, req: DllmReq) -> bool:
        pass

    @abstractmethod
    def may_append(self, req: DllmReq) -> None:
        pass


KVCacheManagerFactory = Callable[[Config], "KVCacheManagerBase"]


class AutoKVCacheManager(DiffulexStrategyRegistry):
    """Registry-driven factory for page manager implementations."""

    @classmethod
    def from_config(cls, config: Config) -> KVCacheManagerBase:
        cls._ensure_strategies_loaded()
        cls._MODULE_MAPPING: dict[str, KVCacheManagerFactory]
        candidates: list[str] = []
        for attr in ("decoding_strategy",):
            value = getattr(config, attr, None)
            if isinstance(value, str) and value:
                candidates.append(value)
        candidates.append(cls._DEFAULT_KEY)

        for key in candidates:
            factory = cls._MODULE_MAPPING.get(key)
            if factory is not None:
                return factory(config)

        available = ", ".join(cls.available_modules()) or "<none>"
        raise ValueError(
            "No page manager registered for decoding_strategy="
            f"'{getattr(config, 'decoding_strategy', None)}'. Available page managers: {available}."
        )
