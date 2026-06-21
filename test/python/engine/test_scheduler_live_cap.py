from __future__ import annotations

from collections import deque
from types import SimpleNamespace

from diffulex.engine.scheduler import SchedulerBase
from diffulex.engine.status import DllmReqStatus


class _FakeKVCacheManager:
    def __init__(self) -> None:
        self.allocated = []
        self.appended = []

    def can_allocate(self, req) -> bool:
        return True

    def allocate(self, req) -> None:
        self.allocated.append(req.req_id)

    def can_append(self, req) -> bool:
        return True

    def may_append(self, req) -> None:
        self.appended.append(req.req_id)


class _FakeReq:
    def __init__(self, req_id: int, status: DllmReqStatus = DllmReqStatus.WAITING) -> None:
        self.req_id = req_id
        self.status = status
        self.num_cached_tokens = 0
        self.new_tokens = 0

    def __len__(self) -> int:
        return 1

    @property
    def is_preempted(self) -> bool:
        return False

    def apply_cached_prefix_pages(self) -> None:
        return None

    def make_pending(self) -> None:
        self.status = DllmReqStatus.PENDING


def _scheduler(max_num_reqs: int, running, waiting) -> SchedulerBase:
    scheduler = object.__new__(SchedulerBase)
    scheduler.config = SimpleNamespace()
    scheduler.max_num_reqs = max_num_reqs
    scheduler.max_num_batched_tokens = 1024
    scheduler.eos = None
    scheduler.block_size = 1
    scheduler.kv_cache_manager = _FakeKVCacheManager()
    scheduler.running_reqs = deque(running)
    scheduler.waiting_reqs = deque(waiting)
    return scheduler


def test_scheduler_does_not_prefill_when_live_reqs_at_cap() -> None:
    running = [
        _FakeReq(1, DllmReqStatus.DECODING),
        _FakeReq(2, DllmReqStatus.DECODING),
    ]
    waiting = [_FakeReq(3)]
    scheduler = _scheduler(max_num_reqs=2, running=running, waiting=waiting)

    scheduled, is_prefill = SchedulerBase.schedule(scheduler)

    assert is_prefill is False
    assert [req.req_id for req in scheduled] == [1, 2]
    assert [req.req_id for req in scheduler.waiting_reqs] == [3]
    assert scheduler.kv_cache_manager.allocated == []


def test_scheduler_prefills_only_until_live_reqs_reach_cap() -> None:
    running = [_FakeReq(1, DllmReqStatus.DECODING)]
    waiting = [_FakeReq(2), _FakeReq(3)]
    scheduler = _scheduler(max_num_reqs=2, running=running, waiting=waiting)

    scheduled, is_prefill = SchedulerBase.schedule(scheduler)

    assert is_prefill is True
    assert [req.req_id for req in scheduled] == [2]
    assert [req.req_id for req in scheduler.running_reqs] == [1, 2]
    assert [req.req_id for req in scheduler.waiting_reqs] == [3]
    assert scheduler.kv_cache_manager.allocated == [2]
