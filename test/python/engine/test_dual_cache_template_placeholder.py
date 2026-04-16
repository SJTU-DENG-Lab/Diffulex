from diffulex.strategy_template.dual_cache import (
    DualCacheKVCacheManagerTemplate,
    DualCacheModelRunnerTemplate,
    DualCacheReqTemplate,
    DualCacheSchedulerTemplate,
)


def test_dual_cache_template_exports_placeholder_classes() -> None:
    assert DualCacheKVCacheManagerTemplate.__name__ == "DualCacheKVCacheManagerTemplate"
    assert DualCacheModelRunnerTemplate.__name__ == "DualCacheModelRunnerTemplate"
    assert DualCacheReqTemplate.__name__ == "DualCacheReqTemplate"
    assert DualCacheSchedulerTemplate.__name__ == "DualCacheSchedulerTemplate"
