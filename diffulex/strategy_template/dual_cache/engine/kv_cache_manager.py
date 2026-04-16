from __future__ import annotations

from diffulex.strategy_template.multi_block.engine.kv_cache_manager import MultiBlockKVCacheManagerTemplate


class DualCacheKVCacheManagerTemplate(MultiBlockKVCacheManagerTemplate):
    """Placeholder dual-cache KV manager template.

    Dual cache refers to prefix/suffix caching while leaving a middle decoding
    subblock uncached. This class intentionally inherits multi-block behavior
    unchanged until that dedicated mechanism is implemented.
    """

    pass
