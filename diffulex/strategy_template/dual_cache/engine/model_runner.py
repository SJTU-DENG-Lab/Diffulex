from __future__ import annotations

from diffulex.strategy_template.multi_block.engine.model_runner import MultiBlockModelRunnerTemplate


class DualCacheModelRunnerTemplate(MultiBlockModelRunnerTemplate):
    """Placeholder dual-cache runner template.

    Intended for a prefix/suffix cache layout with an uncached decoding
    subblock in the middle. No dual-cache-specific execution path is
    implemented yet.
    """

    pass
