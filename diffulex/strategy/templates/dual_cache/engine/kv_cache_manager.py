from __future__ import annotations

from diffulex.engine.kv_cache_manager import KVCacheManagerBase


class DualCacheKVCacheManagerTemplate(KVCacheManagerBase):
    """Placeholder dual-cache KV manager template.

    Dual cache refers to prefix/suffix caching while leaving a middle decoding
    subblock uncached. This class intentionally inherits standard block-cache behavior
    unchanged until that dedicated mechanism is implemented.
    """

    pass
