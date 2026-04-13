from typing import Callable
from collections import deque
from abc import ABC, abstractmethod

from diffulex.config import Config
from diffulex.engine.request import DllmReq
from diffulex.engine.status import DllmReqStatus
from diffulex.engine.kv_cache_manager import AutoKVCacheManager
from diffulex.engine.strategy_registry import DiffulexStrategyRegistry


class SchedulerBase(ABC):
    def __init__(self, config: Config):
        self.config = config
        self.max_num_reqs = config.max_num_reqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.kv_cache_manager = AutoKVCacheManager.from_config(config)
        self.waiting_reqs: deque[DllmReq] = deque()
        self.running_reqs: deque[DllmReq] = deque()

    def is_finished(self) -> bool:
        return not self.waiting_reqs and not self.running_reqs

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

    @abstractmethod
    def add(self, req: DllmReq) -> None:
        pass

    @abstractmethod
    def schedule(self) -> tuple[list[DllmReq], bool]:
        pass

    @abstractmethod
    def preempt(self, req: DllmReq) -> None:
        pass

    @abstractmethod
    def postprocess(self, reqs: list[DllmReq], sampler_output):
        pass


SchedulerFactory = Callable[[Config], "SchedulerBase"]


class AutoScheduler(DiffulexStrategyRegistry):
    """Registry-driven factory for scheduler implementations."""

    @classmethod
    def from_config(cls, config: Config) -> SchedulerBase:
        cls._ensure_strategies_loaded()
        cls._MODULE_MAPPING: dict[str, SchedulerFactory]
        candidates: list[str] = []
        for attr in ("decoding_strategy",):
            value = getattr(config, attr, None)
            if isinstance(value, str) and value:
                candidates.append(value)
        candidates.append(cls._DEFAULT_KEY)

        for key in candidates:
            factory = cls._MODULE_MAPPING.get(key)
            if factory is not None:
                return factory(config)

        available = ", ".join(cls.available_modules()) or "<none>"
        raise ValueError(
            "No scheduler registered for decoding_strategy="
            f"'{getattr(config, 'decoding_strategy', None)}'. Available schedulers: {available}."
        )
