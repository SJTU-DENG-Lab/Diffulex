from collections import deque
from types import SimpleNamespace

import pytest

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
        self.token_ids = []
        self.auto_max_nfe_enabled = max_nfe is None
        self.auto_max_nfe_warmup_steps = 2
        self.auto_max_nfe_tpf_floor = 1.0
        self.auto_max_nfe_token_count = 0
        self.auto_max_nfe_value = None
        self.max_new_tokens = 8

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

    def update_auto_max_nfe(self):
        if not self.auto_max_nfe_enabled or self.max_nfe is not None:
            return
        self.auto_max_nfe_token_count += max(0, int(self.new_tokens))
        if self.nfe < self.auto_max_nfe_warmup_steps:
            return
        import math

        avg_tpf = self.auto_max_nfe_token_count / max(1, self.nfe)
        effective_tpf = max(avg_tpf, self.auto_max_nfe_tpf_floor)
        self.auto_max_nfe_value = max(1, int(math.ceil(self.max_new_tokens / effective_tpf)))
        self.max_nfe = max(self.nfe, self.auto_max_nfe_value)


def test_scheduler_postprocess_kills_req_when_max_nfe_is_reached() -> None:
    scheduler = _Scheduler()
    req = _Req(req_id=7, max_nfe=2)
    scheduler.running_reqs.append(req)
    sample_output = SimpleNamespace(
        true_local_ids_map={"7": {}},
        accepted_ids_map={"7": {}},
        sampled_tokens_map={"7": {}},
        edit_writes_map={"7": {}},
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


def test_scheduler_derives_max_nfe_from_request_average_tpf_when_unset() -> None:
    scheduler = _Scheduler()
    req = _Req(req_id=8, max_nfe=None)
    req.dllm_blocks = [SimpleNamespace(write_token=lambda token, rel_idx: None)]
    scheduler.running_reqs.append(req)
    sample_output = SimpleNamespace(
        true_local_ids_map={"8": {"0": [0, 1, 2, 3]}},
        accepted_ids_map={"8": {"0": [0, 1, 2, 3]}},
        sampled_tokens_map={"8": {"0": [10, 11, 12, 13]}},
        edit_writes_map={"8": {}},
    )

    scheduler.postprocess_multi_block([req], sample_output)
    assert req.max_nfe is None
    assert req.status == DllmReqStatus.DECODING

    scheduler.postprocess_multi_block([req], sample_output)

    assert req.auto_max_nfe_value == 2
    assert req.max_nfe == 2
    assert req.status == DllmReqStatus.FINISHED
    assert req.completion_reason == "max_nfe_reached"


def test_scheduler_does_not_override_explicit_max_nfe() -> None:
    scheduler = _Scheduler()
    req = _Req(req_id=18, max_nfe=10)
    req.new_tokens = 4
    scheduler.running_reqs.append(req)
    sample_output = SimpleNamespace(
        true_local_ids_map={"18": {}},
        accepted_ids_map={"18": {}},
        sampled_tokens_map={"18": {}},
        edit_writes_map={"18": {}},
    )

    scheduler.postprocess_multi_block([req], sample_output)

    assert req.max_nfe == 10
    assert req.auto_max_nfe_value is None
    assert req.status == DllmReqStatus.DECODING


def test_scheduler_postprocess_kills_req_when_repetition_run_is_too_long() -> None:
    scheduler = _Scheduler()
    req = _Req(req_id=9, max_repetition_run=3)
    req.repetition_run_length = 3
    scheduler.running_reqs.append(req)
    sample_output = SimpleNamespace(
        true_local_ids_map={"9": {}},
        accepted_ids_map={"9": {}},
        sampled_tokens_map={"9": {}},
        edit_writes_map={"9": {}},
    )

    scheduler.postprocess_multi_block([req], sample_output)

    assert req.nfe == 1
    assert req.status == DllmReqStatus.FINISHED
    assert scheduler.freed_req_ids == [9]
    assert req not in scheduler.running_reqs


def test_scheduler_postprocess_kills_req_when_max_new_tokens_is_reached() -> None:
    scheduler = _Scheduler()
    req = _Req(req_id=10)
    req.max_new_tokens_reached = True
    scheduler.running_reqs.append(req)
    sample_output = SimpleNamespace(
        true_local_ids_map={"10": {}},
        accepted_ids_map={"10": {}},
        sampled_tokens_map={"10": {}},
        edit_writes_map={"10": {}},
    )

    scheduler.postprocess_multi_block([req], sample_output)

    assert req.nfe == 1
    assert req.status == DllmReqStatus.FINISHED
    assert req.completion_reason == "max_new_tokens_reached"
    assert scheduler.freed_req_ids == [10]
    assert req not in scheduler.running_reqs


def test_scheduler_postprocess_kills_req_when_max_model_len_is_reached() -> None:
    scheduler = _Scheduler()
    req = _Req(req_id=12)
    req.max_model_len_reached = True
    scheduler.running_reqs.append(req)
    sample_output = SimpleNamespace(
        true_local_ids_map={"12": {}},
        accepted_ids_map={"12": {}},
        sampled_tokens_map={"12": {}},
        edit_writes_map={"12": {}},
    )

    scheduler.postprocess_multi_block([req], sample_output)

    assert req.nfe == 1
    assert req.status == DllmReqStatus.FINISHED
    assert req.completion_reason == "max_model_len_reached"
    assert scheduler.freed_req_ids == [12]
    assert req not in scheduler.running_reqs


def test_repetition_run_length_counts_trailing_identical_tokens() -> None:
    req = SimpleNamespace(truncated_response=[11, 22, 22, 22])

    run_length = MultiBlockReqTemplate.repetition_run_length.fget(req)

    assert run_length == 3


def test_scheduler_postprocess_applies_edit_writes_map() -> None:
    scheduler = _Scheduler()
    req = _Req(req_id=11)
    req.token_ids = [9, 0, 0]

    class _Block:
        def __init__(self, req):
            self.req = req
            self.start = 0
            self.mask_token_id = 0

        @property
        def token_ids(self):
            return self.req.token_ids

        def write_token(self, token_id, rel_idx):
            self.req.token_ids[rel_idx] = token_id

    req.dllm_blocks = [_Block(req)]
    sample_output = SimpleNamespace(
        true_local_ids_map={"11": {}},
        accepted_ids_map={"11": {}},
        sampled_tokens_map={"11": {}},
        edit_writes_map={"11": {"0": {0: 0, 1: 5, 2: 6}}},
    )

    scheduler.postprocess_multi_block([req], sample_output)

    assert req.token_ids == [0, 5, 6]
    assert req.new_tokens == 2


def test_scheduler_postprocess_raises_when_req_id_map_is_missing() -> None:
    scheduler = _Scheduler()
    req = _Req(req_id=13)
    scheduler.running_reqs.append(req)
    sample_output = SimpleNamespace(
        true_local_ids_map={},
        accepted_ids_map={},
        sampled_tokens_map={},
        edit_writes_map={},
    )

    with pytest.raises(KeyError, match="13"):
        scheduler.postprocess_multi_block([req], sample_output)


def test_scheduler_postprocess_observes_block_edit_state_from_sample_output() -> None:
    scheduler = _Scheduler()
    req = _Req(req_id=19)

    class _Block:
        def __init__(self):
            self.block_id = 0
            self.start = 0
            self.end = 2
            self.editable_start = 0
            self.status = SimpleNamespace(name="ACTIVE")
            self.mask_token_id = 0
            self.observed_state = None

        @property
        def is_active(self):
            return True

        @property
        def num_mask_tokens(self):
            return 0

        @property
        def token_ids(self):
            return [1, 2]

        def write_token(self, token_id, rel_idx):
            raise AssertionError("write_token should not be called in this test")

        def observe_edit_state(self, token_ids, confidences=None):
            self.observed_state = {
                "token_ids": token_ids,
                "confidences": confidences,
            }

    block = _Block()
    req.dllm_blocks = [block]
    sample_output = SimpleNamespace(
        true_local_ids_map={"19": {}},
        accepted_ids_map={"19": {}},
        sampled_tokens_map={"19": {}},
        edit_writes_map={"19": {}},
        block_state_map={"19": {"0": {"token_ids": [1, 2], "confidences": [0.95, 0.97]}}},
    )

    scheduler.postprocess_multi_block([req], sample_output)

    assert block.observed_state == {
        "token_ids": [1, 2],
        "confidences": [0.95, 0.97],
    }


def test_multiblock_req_postprocess_requires_commit_ready_before_to_cache() -> None:
    class _Block:
        def __init__(self, commit_ready: bool):
            self.prev_block = None
            self.commit_ready = commit_ready
            self._is_to_cache = False

        @property
        def is_active(self):
            return not self._is_to_cache

        @property
        def is_complete(self):
            return True

        @property
        def is_to_cache(self):
            return self._is_to_cache

        @property
        def is_dummy(self):
            return False

        @property
        def is_in_cache(self):
            return False

        def to_cache(self):
            self._is_to_cache = True

    def _build_req(commit_ready: bool):
        block = _Block(commit_ready=commit_ready)
        req = SimpleNamespace(
            maybe_postprocess_prefix_blocks=lambda: None,
            dllm_block_buffer=SimpleNamespace(buffer_size=1, dllm_blocks=[block]),
            push_back_dummy_block=lambda: None,
            eos_token_generated=False,
            max_new_tokens_reached=False,
            max_model_len_reached=False,
            max_nfe_reached=False,
            max_repetition_run_reached=False,
        )
        return req, block

    req, block = _build_req(commit_ready=False)
    MultiBlockReqTemplate.postprocess(req)
    assert block.is_to_cache is False

    req, block = _build_req(commit_ready=True)
    MultiBlockReqTemplate.postprocess(req)
    assert block.is_to_cache is True
