from collections import deque
from types import SimpleNamespace

from diffulex.engine.status import DllmReqStatus
from diffulex.strategy_template.multi_block.engine.request import MultiBlockReqTemplate
from diffulex.strategy_template.multi_block.engine.scheduler import MultiBlockSchedulerTemplate


class _Scheduler(MultiBlockSchedulerTemplate):
    def __init__(self):
        self.kv_cache_manager = SimpleNamespace(free=self._free)
        self.running_reqs = deque()
        self.freed_req_ids: list[int] = []

    def _free(self, req):
        self.freed_req_ids.append(req.req_id)

    def add(self, req):
        pass

    def schedule(self):
        pass

    def preempt(self, req):
        pass

    def postprocess(self, reqs, sample_output):
        pass


class _Req:
    def __init__(self, req_id: int, max_nfe: int | None = None, max_repetition_run: int | None = None):
        self.req_id = req_id
        self.max_nfe = max_nfe
        self.max_repetition_run = max_repetition_run
        self.nfe = 0
        self.new_tokens = 0
        self.dllm_blocks = []
        self.status = DllmReqStatus.DECODING
        self.repetition_run_length = 0
        self.eos_token_generated = False
        self.max_new_tokens_reached = False
        self.max_model_len_reached = False
        self.truncated_response = []
        self.completion_reason = None

    @property
    def max_nfe_reached(self) -> bool:
        return self.max_nfe is not None and self.nfe >= self.max_nfe

    @property
    def max_repetition_run_reached(self) -> bool:
        return self.max_repetition_run is not None and self.repetition_run_length >= self.max_repetition_run

    @property
    def is_completed(self) -> bool:
        return self.status == DllmReqStatus.COMPLETED

    def reset_new_tokens(self):
        self.new_tokens = 0

    def postprocess(self):
        pass

    def force_deactivate(self, reason=None):
        self.completion_reason = reason
        self.status = DllmReqStatus.COMPLETED


def test_scheduler_postprocess_kills_req_when_max_nfe_is_reached() -> None:
    scheduler = _Scheduler()
    req = _Req(req_id=7, max_nfe=2)
    scheduler.running_reqs.append(req)
    sample_output = SimpleNamespace(
        true_local_ids_map={},
        accepted_ids_map={},
        sampled_tokens_map={},
    )

    scheduler.postprocess_multi_block([req], sample_output)
    assert req.nfe == 1
    assert req.status == DllmReqStatus.DECODING
    assert scheduler.freed_req_ids == []

    scheduler.postprocess_multi_block([req], sample_output)
    assert req.nfe == 2
    assert req.status == DllmReqStatus.FINISHED
    assert scheduler.freed_req_ids == [7]
    assert req not in scheduler.running_reqs


def test_scheduler_postprocess_kills_req_when_repetition_run_is_too_long() -> None:
    scheduler = _Scheduler()
    req = _Req(req_id=9, max_repetition_run=3)
    req.repetition_run_length = 3
    scheduler.running_reqs.append(req)
    sample_output = SimpleNamespace(
        true_local_ids_map={},
        accepted_ids_map={},
        sampled_tokens_map={},
    )

    scheduler.postprocess_multi_block([req], sample_output)

    assert req.nfe == 1
    assert req.status == DllmReqStatus.FINISHED
    assert scheduler.freed_req_ids == [9]
    assert req not in scheduler.running_reqs


def test_repetition_run_length_counts_trailing_identical_tokens() -> None:
    req = SimpleNamespace(truncated_response=[11, 22, 22, 22])

    run_length = MultiBlockReqTemplate.repetition_run_length.fget(req)

    assert run_length == 3
