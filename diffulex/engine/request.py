"""Req base class and registry."""

from __future__ import annotations

from copy import copy
from itertools import count
from typing import Callable

from diffulex.config import Config
from diffulex.sampling_params import SamplingParams
from diffulex.engine.strategy_registry import DiffulexStrategyRegistry
from diffulex.engine.status import DllmReqStatus
from diffulex.mixin.multi_block.engine.request import DllmReqMultiBlockMixin


class DllmReq(DllmReqMultiBlockMixin):
    """Minimal base class that tracks prompt tokens and cache bookkeeping."""

    page_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams = SamplingParams()):
        self.req_id = next(DllmReq.counter)
        self.status = DllmReqStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.page_table: list[int] = []
        self.page_cache_missed: list[bool] = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.new_tokens = 0
        self.meet_eos = False
        self.is_multi_block = False

    def __len__(self) -> int:
        return self.num_tokens

    def __getitem__(self, key) -> int:
        return self.token_ids[key]

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def is_finished(self) -> bool:
        return self.status == DllmReqStatus.FINISHED

    @property
    def prompt_token_ids(self) -> list[int]:
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def num_pages(self) -> int:
        if self.is_multi_block:
            return (self.running_len + self.page_size - 1) // self.page_size
        else:
            return (self.num_tokens + self.page_size - 1) // self.page_size

    @property
    def last_page_num_tokens(self) -> int:
        return self.num_tokens - (self.num_pages - 1) * self.page_size

    def reset_new_tokens(self):
        self.new_tokens = 0

    def page(self, index: int) -> list[int]:
        assert 0 <= index < self.num_pages
        return self.token_ids[index * self.page_size : (index + 1) * self.page_size]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1


ReqFactory = Callable[[list[int], SamplingParams, Config], DllmReq]


class AutoReq(DiffulexStrategyRegistry):
    """Registry-driven factory for req implementations."""

    @classmethod
    def create(
        cls,
        config: Config,
        token_ids: list[int],
        sampling_params: SamplingParams = SamplingParams(),
    ) -> DllmReq:
        cls._MODULE_MAPPING: dict[str, ReqFactory]
        candidates: list[str] = []
        for attr in ("decoding_strategy",):
            value = getattr(config, attr, None)
            if isinstance(value, str) and value:
                candidates.append(value)
        candidates.append(cls._DEFAULT_KEY)

        for key in candidates:
            factory = cls._MODULE_MAPPING.get(key)
            if factory is not None:
                return factory(token_ids, sampling_params, config)

        available = ", ".join(cls.available_modules()) or "<none>"
        raise ValueError(
            "No req registered for decoding_strategy="
            f"'{getattr(config, 'decoding_strategy', None)}'. Available reqs: {available}."
        )
