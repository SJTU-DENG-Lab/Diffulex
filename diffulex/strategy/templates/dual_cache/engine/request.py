from __future__ import annotations

from diffulex.engine.request import MultiBlockReqTemplate


class DualCacheReqTemplate(MultiBlockReqTemplate):
    """Placeholder dual-cache request template.

    Intended for future prefix/suffix cache bookkeeping with an uncached
    middle decoding subblock.
    """

    pass
