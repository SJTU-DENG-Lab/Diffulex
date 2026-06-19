"""Req base class and registry."""

from __future__ import annotations

import math
from copy import copy
from itertools import count
from typing import Callable

from diffulex.config import Config
from diffulex.sampling_params import SamplingParams
from diffulex.attention.metadata import is_warming_up
from diffulex.engine.dllm_block import DllmBlock, DllmBlockBuffer
from diffulex.engine.strategy_registry import DiffulexStrategyRegistry
from diffulex.engine.status import DllmBlockStatus, DllmReqStatus
from diffulex.mixin.request_state import ReqStateMixin


class DllmReq(ReqStateMixin):
    """Minimal base class that tracks prompt tokens and cache bookkeeping."""

    page_size = 32
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams = SamplingParams()):
        self.req_id = next(DllmReq.counter)
        self.status = DllmReqStatus.WAITING
        self.dp_rank = 0
        self._dp_owner_assigned = False
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.page_table: list[int] = []
        self.page_cache_missed: list[bool] = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.max_nfe = sampling_params.max_nfe
        self.max_repetition_run = sampling_params.max_repetition_run
        self.ignore_eos = sampling_params.ignore_eos
        self.new_tokens = 0
        self.nfe = 0
        self.meet_eos = False
        self.is_multi_block = False
        self._execution_prepared = False

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
            # return (self.running_len + self.page_size - 1) // self.page_size
            return (self.to_cache_len + self.page_size - 1) // self.page_size
        else:
            return (self.num_tokens + self.page_size - 1) // self.page_size

    @property
    def last_page_num_tokens(self) -> int:
        return self.num_tokens - (self.num_pages - 1) * self.page_size

    def reset_new_tokens(self):
        self.new_tokens = 0

    def on_block_token_rewrite(self, block: DllmBlock, rel_idx: int, old_token: int, new_token: int) -> None:
        del rel_idx
        if old_token == block.mask_token_id and new_token != block.mask_token_id:
            self.new_tokens += 1

    @property
    def is_execution_prepared(self) -> bool:
        return bool(self._execution_prepared)

    def mark_execution_prepared(self) -> None:
        self._execution_prepared = True

    def clear_execution_prepared(self) -> None:
        self._execution_prepared = False

    def assign_dp_rank(self, dp_rank: int) -> None:
        self.dp_rank = dp_rank
        self._dp_owner_assigned = True

    def page(self, index: int) -> list[int]:
        assert 0 <= index < self.num_pages
        return self.token_ids[index * self.page_size : (index + 1) * self.page_size]

    def _restore_req_runtime_state(self):
        super()._restore_req_runtime_state()

        if not self.is_multi_block:
            return

        dllm_blocks = self.dllm_blocks
        dllm_block_buffer = self.dllm_block_buffer
        if dllm_block_buffer is not None:
            dllm_block_buffer.bind_req(self)

        buffer_block_ids = {id(block) for block in dllm_block_buffer.dllm_blocks} if dllm_block_buffer is not None else set()
        for block in dllm_blocks:
            block.bind_req(self)
            block.bind_buffer(dllm_block_buffer if id(block) in buffer_block_ids else None)

    def init_multi_block(self, config: Config):
        self.is_multi_block = True
        self.status_history = [self.status]
        self.completion_reason = None

        self.block_size = config.block_size
        self.buffer_size = config.buffer_size
        self.mask_token_id = config.mask_token_id
        self.thresholds = config.decoding_thresholds
        self.eos_token_id = config.eos
        self.max_model_len = config.max_model_len
        self.max_new_tokens = self.max_tokens
        self.auto_max_nfe_enabled = self.max_nfe is None
        self.auto_max_nfe_warmup_steps = int(getattr(config, "auto_max_nfe_warmup_steps", 8))
        self.auto_max_nfe_tpf_floor = float(getattr(config, "auto_max_nfe_tpf_floor", 1.0))
        self.auto_max_nfe_token_count = 0
        self.auto_max_nfe_value: int | None = None
        self.auto_max_nfe_avg_tpf: float | None = None

        self.dllm_blocks: list[DllmBlock] = []
        self.dllm_block_buffer: DllmBlockBuffer = None

        if self.max_model_len_reached and not is_warming_up():
            self.force_deactivate()
            return

        self.prefix_len = len(self.token_ids)

        if self.prefix_len % self.block_size != 0:
            padding_len = self.block_size - self.prefix_len % self.block_size
            self.pad_tokens(padding_len)
            self.padded_prefix_len = self.prefix_len + padding_len
        else:
            self.padded_prefix_len = self.prefix_len

        for i in range(self.padded_prefix_len // self.block_size):
            start = i * self.block_size
            end = start + self.block_size
            editable_start = 0
            if self.is_padded and i == (self.padded_prefix_len // self.block_size) - 1:
                editable_start = self.prefix_len - start
            dllm_block = DllmBlock(
                block_id=i,
                start=start,
                end=end,
                block_size=self.block_size,
                mask_token_id=self.mask_token_id,
                thresholds=self.thresholds,
                status=None,
                prev_block=None if i == 0 else self.dllm_blocks[-1],
                editable_start=editable_start,
            )
            dllm_block.post_init_dllm_block(self, None)
            self.dllm_blocks.append(dllm_block)

        remain_buffer_size = self.buffer_size - 1 if self.is_padded else self.buffer_size
        for _ in range(remain_buffer_size):
            block_id = len(self.dllm_blocks)
            start = block_id * self.block_size
            end = start + self.block_size
            self.extend_block_tokens()
            dllm_block = DllmBlock(
                block_id=block_id,
                start=start,
                end=end,
                block_size=self.block_size,
                mask_token_id=self.mask_token_id,
                thresholds=self.thresholds,
                status=DllmBlockStatus.DUMMY,
                prev_block=self.dllm_blocks[-1],
            )
            dllm_block.post_init_dllm_block(self, None)
            self.dllm_blocks.append(dllm_block)

        self.dllm_block_buffer = DllmBlockBuffer(
            buffer_size=self.buffer_size,
            dllm_blocks=self.dllm_blocks[-self.buffer_size :],
        )
        self.dllm_block_buffer.post_init_dllm_block_buffer(self)

    @property
    def eos_token_generated(self) -> bool:
        if self.ignore_eos:
            return False
        generated_seq = self.token_ids[self.prefix_len :]
        return self.eos_token_id in generated_seq

    @property
    def num_prefix_blocks(self) -> int:
        return self.prefix_len // self.block_size

    @property
    def num_prefix_pages(self) -> int:
        return self.num_pages_with_seq_len(self.prefix_len)

    @property
    def prefix_len_truncate_padding(self) -> int:
        return self.padded_prefix_len - self.prefix_len if self.is_padded else self.padded_prefix_len

    @property
    def max_new_tokens_reached(self) -> bool:
        return len(self.token_ids) - self.prefix_len >= self.max_new_tokens

    @property
    def max_model_len_reached(self) -> bool:
        return len(self.token_ids) >= self.max_model_len

    @property
    def max_nfe_reached(self) -> bool:
        return self.max_nfe is not None and self.nfe >= self.max_nfe

    def update_auto_max_nfe(self) -> None:
        if not self.auto_max_nfe_enabled or self.max_nfe is not None:
            return
        self.auto_max_nfe_token_count += max(0, int(self.new_tokens))
        if self.nfe < max(1, self.auto_max_nfe_warmup_steps):
            return

        avg_tpf = self.auto_max_nfe_token_count / max(1, self.nfe)
        self.auto_max_nfe_avg_tpf = avg_tpf
        effective_tpf = max(avg_tpf, self.auto_max_nfe_tpf_floor)
        self.auto_max_nfe_value = max(1, int(math.ceil(self.max_new_tokens / effective_tpf)))
        self.max_nfe = max(self.nfe, self.auto_max_nfe_value)

    @property
    def repetition_run_length(self) -> int:
        generated = self.truncated_response
        if not generated:
            return 0

        last_token = generated[-1]
        run_length = 1
        for token in reversed(generated[:-1]):
            if token != last_token:
                break
            run_length += 1
        return run_length

    @property
    def max_repetition_run_reached(self) -> bool:
        return self.max_repetition_run is not None and self.repetition_run_length >= self.max_repetition_run

    @property
    def running_sequence(self) -> list[int]:
        if self.is_prefilling:
            return self.token_ids[self.contiguous_in_cache_prefix_len : self.running_len]
        if self.is_decoding or self.is_completed:
            return self.dllm_block_buffer.buffer_sequence
        return []

    @property
    def running_position_ids(self) -> range | list[int]:
        if self.is_prefilling:
            return range(self.contiguous_in_cache_prefix_len, self.running_len)
        if self.is_decoding or self.is_completed:
            return self.dllm_block_buffer.buffer_position_ids
        return []

    @property
    def truncated_response(self) -> list[int]:
        if self.eos_token_generated:
            generated_seq = self.token_ids[self.prefix_len :]
            if self.eos_token_id in generated_seq:
                first_eos_pos = generated_seq.index(self.eos_token_id)
                truncate_pos = self.prefix_len + first_eos_pos
                return self.token_ids[self.prefix_len : truncate_pos]
            return self.token_ids[self.prefix_len :]
        if self.max_model_len_reached:
            return self.token_ids[self.prefix_len : self.max_model_len]
        if self.max_new_tokens_reached:
            return self.token_ids[self.prefix_len : self.prefix_len + self.max_new_tokens]
        dummy_len = self.chunk_size - len(self.dllm_block_buffer.valid_blocks) * self.block_size
        if dummy_len <= 0:
            return self.token_ids[self.prefix_len :]
        return self.token_ids[self.prefix_len : -dummy_len]

    @property
    def is_truncated(self) -> bool:
        return (
            self.eos_token_generated
            or self.max_model_len_reached
            or self.max_new_tokens_reached
            or self.max_nfe_reached
            or self.max_repetition_run_reached
        )

    @property
    def full_response(self) -> list[int]:
        return self.token_ids[self.prefix_len :]

    @property
    def is_padded(self):
        return self.prefix_len != self.padded_prefix_len

    @property
    def is_waiting(self) -> bool:
        return self.status == DllmReqStatus.WAITING

    @property
    def is_pending(self) -> bool:
        return self.status == DllmReqStatus.PENDING

    @property
    def is_prefilling(self) -> bool:
        return self.status == DllmReqStatus.PREFILLING

    @property
    def is_decoding(self) -> bool:
        return self.status == DllmReqStatus.DECODING

    @property
    def is_running(self) -> bool:
        return self.is_prefilling or self.is_decoding

    @property
    def is_completed(self) -> bool:
        return self.status == DllmReqStatus.COMPLETED

    @property
    def chunk_size(self) -> int:
        return self.block_size * self.buffer_size

    @property
    def running_len(self) -> int:
        if self.is_prefilling:
            return (
                (self.padded_prefix_len - self.block_size) + self.dllm_block_buffer.num_valid_blocks * self.block_size
                if self.is_padded
                else self.prefix_len
            )
        if self.is_pending or self.is_waiting:
            padded_running_len = (
                self.padded_prefix_len - self.block_size
            ) + self.dllm_block_buffer.num_valid_blocks * self.block_size
            if self.dllm_block_buffer.dllm_blocks[0].should_add_block:
                padded_running_len += self.block_size
            return padded_running_len if self.is_padded else self.prefix_len + self.block_size
        if self.is_decoding:
            return self.dllm_block_buffer.num_running_blocks * self.block_size
        return 0

    @property
    def valid_len(self) -> int:
        if self.is_prefilling:
            return self.running_len
        if self.is_decoding:
            return self.dllm_block_buffer.num_valid_blocks * self.block_size
        return 0

    @property
    def to_cache_len(self) -> int:
        if self.is_prefilling or self.is_pending or self.is_waiting:
            return sum(b.is_to_cache for b in self.dllm_blocks) * self.block_size
        if self.is_decoding or self.is_preempted:
            return len(self.dllm_block_buffer.to_cache_blocks) * self.block_size
        return 0

    @property
    def in_cache_len(self) -> int:
        return sum(block.block_size for block in self.dllm_blocks if block.is_in_cache)

    @property
    def contiguous_in_cache_prefix_len(self) -> int:
        total = 0
        for block in self.dllm_blocks:
            if block.start != total:
                break
            if not block.is_in_cache:
                break
            total += block.block_size
        return total

    @property
    def cache_len(self) -> int:
        return sum(block.block_size for block in self.dllm_blocks if block.is_to_cache or block.is_in_cache)

    @property
    def running_seq_start(self) -> int:
        if self.is_prefilling:
            return 0
        if self.is_decoding:
            return self.dllm_block_buffer.first_running_block.start
        return 0

    @property
    def running_seq_end(self) -> int:
        return self.running_seq_start + self.running_len

    @property
    def to_cache_seq_start(self) -> tuple[int, int]:
        return (0, self.running_seq_start)

    @property
    def to_cache_seq_end(self) -> tuple[int, int]:
        if self.is_prefilling:
            return (self.to_cache_len, self.to_cache_len)
        if self.is_decoding:
            return (self.to_cache_len, self.to_cache_len + self.running_seq_start)
        return (0, 0)

    @property
    def has_to_cache_blocks(self) -> bool:
        if self.is_prefilling:
            return True
        if self.is_decoding:
            return len(self.dllm_block_buffer.to_cache_blocks) > 0
        return False

    @property
    def has_to_cache_block(self) -> bool:
        return self.has_to_cache_blocks

    @property
    def to_cache_last_token_id(self) -> int:
        if self.is_prefilling:
            return self.to_cache_len - 1 if self.to_cache_len > 0 else 0
        n = len(self.dllm_block_buffer.to_cache_blocks) * self.block_size
        return n - 1 if n > 0 else 0

    @property
    def last_block_finished(self) -> bool:
        inspected_block = self.dllm_block_buffer.first_running_block.prev_block
        return inspected_block is not None and inspected_block.is_complete and inspected_block.is_last_in_context

    @property
    def pure_prefill_without_mask_token(self) -> bool:
        return self.is_prefilling and not self.is_padded

    def num_pages_with_seq_len(self, seq_len: int) -> int:
        return (seq_len + self.page_size - 1) // self.page_size

    def num_blocks_with_seq_len(self, seq_len: int) -> int:
        return (seq_len + self.block_size - 1) // self.block_size

    def pad_tokens(self, cnt: int):
        self.token_ids.extend([self.mask_token_id] * cnt)

    def extend_block_tokens(self):
        self.pad_tokens(self.block_size)

    def make_pending(self):
        if self.status == DllmReqStatus.WAITING:
            self.status = DllmReqStatus.PENDING

    def preempt(self):
        self.lazy_activate()
        self.log_status()
        self.status = DllmReqStatus.WAITING

    @property
    def is_preempted(self) -> bool:
        return self.status_history[-1] in [
            DllmReqStatus.PREFILLING,
            DllmReqStatus.DECODING,
        ]

    def lazy_activate(self):
        self.log_status()

        self.status = self.status_history[-1]
        if self.is_pending:
            self.status = DllmReqStatus.PREFILLING
        elif self.is_prefilling:
            self.status = DllmReqStatus.DECODING

    def log_status(self):
        if self.status not in self.status_history:
            self.status_history.append(self.status)

    def preempt_time_prefilling(self):
        return self.status_history[-1] in [
            DllmReqStatus.WAITING,
            DllmReqStatus.PENDING,
            DllmReqStatus.PREFILLING,
        ]

    def force_deactivate(self, reason: str | None = None):
        if reason is not None:
            self.completion_reason = reason
        self.status = DllmReqStatus.COMPLETED

    def deactivate(self, reason: str | None = None):
        if self.is_running:
            if reason is not None:
                self.completion_reason = reason
            self.status = DllmReqStatus.COMPLETED

    def step(self):
        self.lazy_activate()

        for block in self.dllm_block_buffer.active_blocks:
            block.total_steps += 1

        activate_cond = self.dllm_block_buffer.should_add_block and not self.dllm_block_buffer.is_overflow
        activate_cond_all_dummy_buffer_backup = (
            not self.dllm_block_buffer.active_blocks
            and not self.eos_token_generated
            and self.dllm_block_buffer.dllm_blocks[0].is_dummy
            and self.dllm_block_buffer.dllm_blocks[0].prev_block.is_in_cache
        )
        if activate_cond or activate_cond_all_dummy_buffer_backup:
            self.dllm_block_buffer.activate_cursor_slot_block()

    def push_back_dummy_block(self):
        self.extend_block_tokens()
        dllm_block = DllmBlock(
            block_id=len(self.dllm_blocks),
            start=self.dllm_blocks[-1].end,
            end=self.dllm_blocks[-1].end + self.block_size,
            block_size=self.block_size,
            mask_token_id=self.mask_token_id,
            thresholds=self.thresholds,
            status=DllmBlockStatus.DUMMY,
            prev_block=self.dllm_blocks[-1],
            editable_start=0,
        )

        dllm_block.post_init_dllm_block(self, self.dllm_block_buffer)
        if (self.max_new_tokens_reached or self.max_model_len_reached) and dllm_block.prev_block.is_in_context:
            dllm_block.make_last_in_context()
        elif dllm_block.prev_block.is_out_of_context or dllm_block.prev_block.is_last_in_context:
            dllm_block.make_out_of_context()

        self.dllm_blocks.append(dllm_block)
        self.dllm_block_buffer.push_back(dllm_block)

    def maybe_postprocess_prefix_blocks(self):
        if not self.is_prefilling:
            return

        for block_id in range(self.num_prefix_blocks):
            self.dllm_blocks[block_id].in_cache()

    def apply_cached_prefix_pages(self):
        if not self.is_multi_block:
            return
        if not self.page_cache_missed:
            return

        cached_pages = 0
        for missed in self.page_cache_missed:
            if missed:
                break
            cached_pages += 1

        if cached_pages == 0:
            return

        blocks_per_page = self.page_size // self.block_size
        cached_blocks = min(cached_pages * blocks_per_page, len(self.dllm_blocks))
        for block_id in range(cached_blocks):
            self.dllm_blocks[block_id].in_cache()

    def postprocess(self):
        self.maybe_postprocess_prefix_blocks()
        block_id = 0
        while block_id < self.dllm_block_buffer.buffer_size:
            block = self.dllm_block_buffer.dllm_blocks[block_id]
            prev_ready = block.prev_block is None or block.prev_block.is_to_cache or block.prev_block.is_in_cache
            if block.is_active and block.is_complete and bool(getattr(block, "commit_ready", False)) and prev_ready:
                block.to_cache()
                block_id += 1
            elif block.is_to_cache:
                block.in_cache()
                self.dllm_block_buffer.pop_front()
                self.push_back_dummy_block()
            elif block.is_dummy or block.is_active or block.is_in_cache:
                block_id += 1

        if self.eos_token_generated:
            self.dllm_block_buffer.last_valid_block.make_last_in_context()
            self.meet_eos = True
            self.dllm_block_buffer.maybe_fix_context_management()

        if (
            self.eos_token_generated
            or self.max_new_tokens_reached
            or self.max_model_len_reached
            or self.max_nfe_reached
            or self.max_repetition_run_reached
        ) and self.last_block_finished:
            completed_blocks = [block.is_complete for block in self.dllm_block_buffer.valid_blocks]
            if all(completed_blocks):
                if self.eos_token_generated:
                    reason = "eos_token_generated"
                elif self.max_new_tokens_reached:
                    reason = "max_new_tokens_reached"
                elif self.max_model_len_reached:
                    reason = "max_model_len_reached"
                elif self.max_nfe_reached:
                    reason = "max_nfe_reached"
                elif self.max_repetition_run_reached:
                    reason = "max_repetition_run_reached"
                else:
                    reason = "unknown_postprocess_deactivate"
                self.deactivate(reason=reason)


class MultiBlockReqTemplate(DllmReq):
    """Compatibility alias for the core block-aware request base."""


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
        if config.decoding_strategy:
            candidates.append(config.decoding_strategy)
        candidates.append(cls._DEFAULT_KEY)

        for key in candidates:
            factory = cls._MODULE_MAPPING.get(key)
            if factory is not None:
                return factory(token_ids, sampling_params, config)

        available = ", ".join(cls.available_modules()) or "<none>"
        raise ValueError(
            "No req registered for decoding_strategy="
            f"'{config.decoding_strategy}'. Available reqs: {available}."
        )
