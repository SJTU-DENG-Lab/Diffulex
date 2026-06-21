from __future__ import annotations

from abc import ABC
from collections import deque, defaultdict
from typing import Callable

from diffulex.config import Config
from diffulex.engine.kv_cache_manager import AutoKVCacheManager
from diffulex.engine.request import DllmReq
from diffulex.engine.status import DllmReqStatus
from diffulex.engine.strategy_registry import DiffulexStrategyRegistry
from diffulex.logger import get_logger
from diffulex.mixin.block_rewrite.scheduler import BlockRewriteSchedulerMixin


class SchedulerBase(BlockRewriteSchedulerMixin, ABC):
    _logger = get_logger(__name__)

    def __init__(self, config: Config):
        self.config = config
        self.max_num_reqs = config.max_num_reqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.init_multi_block()
        self.kv_cache_manager = AutoKVCacheManager.from_config(config)
        self.waiting_reqs: deque[DllmReq] = deque()
        self.running_reqs: deque[DllmReq] = deque()

    def is_finished(self) -> bool:
        return not self.waiting_reqs and not self.running_reqs

    def init_multi_block(self) -> None:
        self.block_size = self.config.block_size

    def abort_request(self, req_id: int) -> bool:
        for req in list(self.waiting_reqs):
            if req.req_id == req_id:
                self.waiting_reqs.remove(req)
                req.status = DllmReqStatus.FINISHED
                setattr(req, "completion_reason", "aborted")
                return True

        for req in list(self.running_reqs):
            if req.req_id == req_id:
                self.running_reqs.remove(req)
                setattr(req, "completion_reason", "aborted")
                req.status = DllmReqStatus.FINISHED
                self.kv_cache_manager.free(req)
                return True

        return False

    def add(self, req: DllmReq) -> None:
        req.init_multi_block(self.config)
        self.waiting_reqs.append(req)

    def schedule(self) -> tuple[list[DllmReq], bool]:
        scheduled: list[DllmReq] = []
        num_reqs = 0
        num_batched_tokens = 0

        while (
            self.waiting_reqs
            and num_reqs < self.max_num_reqs
            and len(self.running_reqs) < self.max_num_reqs
        ):
            req = self.waiting_reqs[0]

            projected = len(req) + self.block_size
            if num_batched_tokens + projected > self.max_num_batched_tokens or not self.kv_cache_manager.can_allocate(
                req
            ):
                break

            num_reqs += 1
            self.kv_cache_manager.allocate(req)
            req.apply_cached_prefix_pages()
            if req.is_preempted:
                if not self.kv_cache_manager.can_append(req):
                    self.kv_cache_manager.free(req)
                    num_reqs -= 1
                    break
                self.kv_cache_manager.may_append(req)

            num_batched_tokens += projected - req.num_cached_tokens
            req.make_pending()
            self.waiting_reqs.popleft()
            self.running_reqs.append(req)
            scheduled.append(req)

        if scheduled:
            return scheduled, True

        while self.running_reqs and num_reqs < self.max_num_reqs:
            req = self.running_reqs.popleft()
            while not self.kv_cache_manager.can_append(req):
                if self.running_reqs:
                    self.preempt(self.running_reqs.pop())
                else:
                    self.preempt(req)
                    break
            else:
                num_reqs += 1
                self.kv_cache_manager.may_append(req)
                scheduled.append(req)

        if not scheduled:
            diag = dict(
                phase="decode",
                waiting=len(self.waiting_reqs),
                running=len(self.running_reqs),
                max_num_reqs=self.max_num_reqs,
                max_num_batched_tokens=self.max_num_batched_tokens,
                block_size=self.block_size,
            )
            candidates = list(self.running_reqs)[:3] + list(self.waiting_reqs)[:2]
            details = []
            for idx, candidate in enumerate(candidates):
                try:
                    can_append = self.kv_cache_manager.can_append(candidate)
                except Exception:
                    can_append = "error"
                details.append(
                    f"[{idx}] status={candidate.status.name}, len={len(candidate)}, "
                    f"block_size={self.block_size}, "
                    f"new_tokens={candidate.new_tokens}, "
                    f"cached={candidate.num_cached_tokens}, "
                    f"can_append={can_append}"
                )
            raise RuntimeError(
                "Scheduler: unable to schedule any req in decode; "
                f"state={diag}; details={' | '.join(details)}"
            )

        self.running_reqs.extendleft(reversed(scheduled))
        return scheduled, False

    def preempt(self, req: DllmReq) -> None:
        req.preempt()
        self.kv_cache_manager.free(req)
        self.waiting_reqs.appendleft(req)

    def postprocess(self, reqs: list[DllmReq], sampler_output):
        for req in reqs:
            req.reset_new_tokens()

            req_id_str = str(req.req_id)
            true_ids_map = sampler_output.true_local_ids_map[req_id_str]
            accepted_ids_map = sampler_output.accepted_ids_map[req_id_str]
            sampled_tokens_map = sampler_output.sampled_tokens_map[req_id_str]
            for block_id, accepted_ids in accepted_ids_map.items():
                if not accepted_ids:
                    continue

                dllm_block = req.dllm_blocks[int(block_id)]
                sampled_tokens = sampled_tokens_map[block_id]
                true_local_ids = true_ids_map[block_id]
                for true_local_id, accepted_id in zip(true_local_ids, accepted_ids):
                    token = sampled_tokens[accepted_id]
                    dllm_block.write_token(token, true_local_id)
                req.new_tokens += len(accepted_ids)

            self.apply_block_writes_map(req, sampler_output)
            block_state_map = getattr(sampler_output, "block_state_map", {}).get(req_id_str, {})
            for block in req.dllm_blocks:
                if not getattr(block, "is_active", False):
                    continue
                state = block_state_map.get(str(block.block_id))
                if state is not None:
                    block.commit_ready = bool(state.get("committable", False))
                    block.same_as_previous = bool(state.get("same_as_previous", False))
                    block.same_token_ratio = float(state.get("same_token_ratio", 0.0))
                    block.all_confident = bool(state.get("all_confident", False))
                    if "valid_commit_len" in state:
                        block.valid_commit_len = int(state["valid_commit_len"])
                elif block.is_complete:
                    block.commit_ready = True
                    block.same_token_ratio = 1.0

            req.postprocess()
            req.nfe += 1
            update_auto_max_nfe = getattr(req, "update_auto_max_nfe", None)
            if callable(update_auto_max_nfe):
                update_auto_max_nfe()
            if (
                req.max_new_tokens_reached
                or req.max_model_len_reached
                or req.max_nfe_reached
                or req.max_repetition_run_reached
            ):
                if req.max_new_tokens_reached:
                    reason = "max_new_tokens_reached"
                elif req.max_model_len_reached:
                    reason = "max_model_len_reached"
                elif req.max_nfe_reached:
                    reason = "max_nfe_reached"
                else:
                    reason = "max_repetition_run_reached"
                req.force_deactivate(reason=reason)
            if req.is_completed:
                if req.completion_reason is None:
                    req.completion_reason = "completed_without_reason"
                self._logger.debug(
                    "Req %s marked FINISHED (reason=%s, eos=%s, max_new=%s, max_model_len=%s, max_nfe=%s, max_repeat=%s, nfe=%s, gen_tokens=%s, auto_max_nfe=%s, avg_tpf=%s)",
                    req.req_id,
                    req.completion_reason,
                    req.eos_token_generated,
                    req.max_new_tokens_reached,
                    req.max_model_len_reached,
                    req.max_nfe_reached,
                    req.max_repetition_run_reached,
                    req.nfe,
                    len(req.truncated_response) if req.truncated_response is not None else -1,
                    getattr(req, "auto_max_nfe_value", None),
                    (
                        f"{getattr(req, 'auto_max_nfe_avg_tpf', 0.0):.2f}"
                        if getattr(req, "auto_max_nfe_avg_tpf", None) is not None
                        else "n/a"
                    ),
                )
                req.status = DllmReqStatus.FINISHED
                self.kv_cache_manager.free(req)
                if req in self.running_reqs:
                    self.running_reqs.remove(req)


class MultiBlockSchedulerTemplate(SchedulerBase):
    """Compatibility alias for the core block-aware scheduler base."""


class DataParallelScheduler:
    def __init__(self, config: Config, scheduler_factory: Callable[[Config], SchedulerBase]):
        self.config = config
        self.dp_size = config.data_parallel_size
        self.schedulers = [scheduler_factory(config) for _ in range(self.dp_size)]

    def _owner_for_req(self, req: DllmReq) -> int:
        owner_assigned = bool(getattr(req, "_dp_owner_assigned", False))
        owner = getattr(req, "dp_rank", None)
        if not owner_assigned or owner is None or not (0 <= owner < self.dp_size):
            owner = req.req_id % self.dp_size
            assign_fn = getattr(req, "assign_dp_rank", None)
            if callable(assign_fn):
                assign_fn(owner)
            else:
                req.dp_rank = owner
        return owner

    def add(self, req: DllmReq) -> None:
        owner = self._owner_for_req(req)
        self.schedulers[owner].add(req)

    def schedule(self) -> tuple[list[DllmReq], bool]:
        scheduled: list[DllmReq] = []
        saw_prefill = False
        for scheduler in self.schedulers:
            if scheduler.is_finished():
                continue
            local_reqs, is_prefill = scheduler.schedule()
            scheduled.extend(local_reqs)
            saw_prefill = saw_prefill or is_prefill
        return scheduled, saw_prefill

    def postprocess(self, reqs: list[DllmReq], sampler_output) -> None:
        reqs_by_owner: dict[int, list[DllmReq]] = defaultdict(list)
        for req in reqs:
            reqs_by_owner[self._owner_for_req(req)].append(req)

        for owner, local_reqs in reqs_by_owner.items():
            if local_reqs:
                self.schedulers[owner].postprocess(local_reqs, sampler_output)

    def is_finished(self) -> bool:
        return all(scheduler.is_finished() for scheduler in self.schedulers)

    def abort_request(self, req_id: int) -> bool:
        for scheduler in self.schedulers:
            if scheduler.abort_request(req_id):
                return True
        return False


SchedulerFactory = Callable[[Config], SchedulerBase]


class AutoScheduler(DiffulexStrategyRegistry):
    """Registry-driven factory for scheduler implementations."""

    @classmethod
    def _from_single_config(cls, config: Config) -> SchedulerBase:
        cls._ensure_strategies_loaded()
        cls._MODULE_MAPPING: dict[str, SchedulerFactory]
        candidates: list[str] = []
        if config.decoding_strategy:
            candidates.append(config.decoding_strategy)
        candidates.append(cls._DEFAULT_KEY)

        for key in candidates:
            factory = cls._MODULE_MAPPING.get(key)
            if factory is not None:
                return factory(config)

        available = ", ".join(cls.available_modules()) or "<none>"
        raise ValueError(
            "No scheduler registered for decoding_strategy="
            f"'{config.decoding_strategy}'. Available schedulers: {available}."
        )

    @classmethod
    def from_config(cls, config: Config) -> SchedulerBase | DataParallelScheduler:
        if config.data_parallel_size > 1:
            return DataParallelScheduler(config, scheduler_factory=cls._from_single_config)
        return cls._from_single_config(config)
