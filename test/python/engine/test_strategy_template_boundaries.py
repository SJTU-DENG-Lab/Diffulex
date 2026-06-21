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


def test_block_runtime_methods_are_core_contracts() -> None:
    assert hasattr(DllmReq, "init_multi_block")
    assert hasattr(MultiBDReq, "init_multi_block")

    assert hasattr(SchedulerBase, "schedule")
    assert not hasattr(SchedulerBase, "schedule_multi_block")
    assert not hasattr(MultiBDScheduler, "schedule_multi_block")

    assert hasattr(KVCacheManagerBase, "can_append")
    assert hasattr(KVCacheManagerBase, "may_append")
    assert not hasattr(KVCacheManagerBase, "can_append_multi_block")
    assert not hasattr(MultiBDKVCacheManager, "can_append_multi_block")

    assert hasattr(ModelRunnerBase, "run_multi_block")
    assert hasattr(MultiBDModelRunner, "run_multi_block")

    assert not hasattr(AttnMetaDataBase, "init_multi_block")
    assert hasattr(MultiBDAttnMetaData, "init_multi_block")
