from diffulex.attention.metadata import AttnMetaDataBase
from diffulex.engine.kv_cache_manager import KVCacheManagerBase
from diffulex.engine.model_runner import ModelRunnerBase
from diffulex.engine.request import DllmReq
from diffulex.engine.scheduler import SchedulerBase
from diffulex.strategy.multi_bd.attention.metadata import MultiBDAttnMetaData
from diffulex.strategy.multi_bd.engine.kv_cache_manager import MultiBDKVCacheManager
from diffulex.strategy.multi_bd.engine.model_runner import MultiBDModelRunner
from diffulex.strategy.multi_bd.engine.request import MultiBDReq
from diffulex.strategy.multi_bd.engine.scheduler import MultiBDScheduler


def test_multi_block_methods_live_on_strategy_template_not_base() -> None:
    assert not hasattr(DllmReq, "init_multi_block")
    assert hasattr(MultiBDReq, "init_multi_block")

    assert not hasattr(SchedulerBase, "schedule_multi_block")
    assert hasattr(MultiBDScheduler, "schedule_multi_block")

    assert not hasattr(KVCacheManagerBase, "can_append_multi_block")
    assert hasattr(MultiBDKVCacheManager, "can_append_multi_block")

    assert not hasattr(ModelRunnerBase, "run_multi_block")
    assert hasattr(MultiBDModelRunner, "run_multi_block")

    assert not hasattr(AttnMetaDataBase, "init_multi_block")
    assert hasattr(MultiBDAttnMetaData, "init_multi_block")
