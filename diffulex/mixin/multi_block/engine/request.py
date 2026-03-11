from __future__ import annotations

from typing import TYPE_CHECKING

from diffulex.config import Config
from diffulex.engine.dllm_block import DllmBlock, DllmBlockBuffer
from diffulex.engine.status import DllmBlockStatus, DllmReqStatus
from diffulex.attention.metadata import is_warming_up

if TYPE_CHECKING:
    from diffulex.engine.request import DllmReq


class DllmReqMultiBlockMixin:
    def init_multi_block(self: DllmReq, config: Config):
        self.is_multi_block = True
        self.status_history = [self.status]

        self.block_size = config.block_size
        self.buffer_size = config.buffer_size
        self.mask_token_id = config.mask_token_id
        self.thresholds = config.decoding_thresholds
        self.eos_token_id = config.eos
        self.max_model_len = config.max_model_len
        self.max_new_tokens = self.max_tokens  # from sampling_params in __init__

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
            dllm_block = DllmBlock(
                block_id=i,
                start=start,
                end=end,
                block_size=self.block_size,
                mask_token_id=self.mask_token_id,
                thresholds=self.thresholds,
                status=None,  # let Block __post_init__ to set the status
                prev_block=None if i == 0 else self.dllm_blocks[-1],
            )
            dllm_block.post_init_dllm_block(self, None)
            self.dllm_blocks.append(dllm_block)

        remain_buffer_size = self.buffer_size - 1 if self.is_padded else self.buffer_size
        for i in range(remain_buffer_size):
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
        seq = self.running_sequence
        if seq is None:
            return False
        eos_detect_fn = lambda x: self.eos_token_id in x
        last_in_cache_block = self.dllm_block_buffer.first_running_block.prev_block
        return eos_detect_fn(seq) or (
            last_in_cache_block.is_last_in_context and eos_detect_fn(last_in_cache_block.token_ids)
        )

    @property
    def num_prefix_blocks(self) -> int:
        # return self.prefix_len // self.block_size
        return self.num_blocks_with_seq_len(self.prefix_len)

    @property
    def num_prefix_pages(self: DllmReq) -> int:
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
    def running_sequence(self) -> list[int]:
        if self.is_prefilling:
            prefilling_len = self.running_len
            return self.token_ids[:prefilling_len]
        elif self.is_decoding or self.is_completed:
            return self.dllm_block_buffer.buffer_sequence

    @property
    def running_position_ids(self) -> list[int]:
        if self.is_prefilling:
            return range(self.in_cache_len, len(self))
        elif self.is_decoding or self.is_completed:
            return self.dllm_block_buffer.buffer_position_ids

    @property
    def truncated_response(self) -> list[int]:
        if self.eos_token_generated:
            generated_seq = self.token_ids[self.prefix_len :]
            if self.eos_token_id in generated_seq:
                first_eos_pos = generated_seq.index(self.eos_token_id)
                truncate_pos = self.prefix_len + first_eos_pos
                return self.token_ids[self.prefix_len : truncate_pos]
            return self.token_ids[self.prefix_len :]
        elif self.max_model_len_reached:
            return self.token_ids[self.prefix_len : self.max_model_len]
        elif self.max_new_tokens_reached:
            return self.token_ids[self.prefix_len : self.prefix_len + self.max_new_tokens]
        else:
            dummy_len = self.chunk_size - len(self.dllm_block_buffer.valid_blocks) * self.block_size
            return self.token_ids[self.prefix_len : -dummy_len]

    @property
    def is_truncated(self) -> bool:
        return self.eos_token_generated or self.max_model_len_reached or self.max_new_tokens_reached

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
        elif self.is_pending or self.is_waiting:
            padded_running_len = (
                self.padded_prefix_len - self.block_size
            ) + self.dllm_block_buffer.num_valid_blocks * self.block_size
            if self.dllm_block_buffer.dllm_blocks[0].should_add_block:
                padded_running_len += self.block_size
            return padded_running_len if self.is_padded else self.prefix_len + self.block_size
        elif self.is_decoding:
            return self.dllm_block_buffer.num_running_blocks * self.block_size

    @property
    def valid_len(self) -> int:
        if self.is_prefilling:
            return self.running_len
        elif self.is_decoding:
            return self.dllm_block_buffer.num_valid_blocks * self.block_size

    @property
    def to_cache_len(self) -> int:
        if self.is_prefilling:
            return self.padded_prefix_len - self.block_size if self.is_padded else self.padded_prefix_len
        elif self.is_decoding or self.is_preempted:
            return len(self.dllm_block_buffer.to_cache_blocks) * self.block_size

    @property
    def in_cache_len(self) -> int:
        return sum(block.block_size for block in self.dllm_blocks if block.is_in_cache)

    @property
    def cache_len(self) -> int:
        return sum(block.block_size for block in self.dllm_blocks if block.is_to_cache or block.is_in_cache)

    @property
    def running_seq_start(self) -> int:
        # return: absolute start
        if self.is_prefilling:
            return 0
        elif self.is_decoding:
            return self.dllm_block_buffer.first_running_block.start

    @property
    def running_seq_end(self) -> int:
        # return: absolute end
        return self.running_seq_start + self.running_len

    @property
    def to_cache_seq_start(self) -> int:
        # return: tuple(relative start, absolute start)
        return (0, self.running_seq_start)

    @property
    def to_cache_seq_end(self) -> tuple[int, int]:
        # return: tuple(relative end, absolute end)
        if self.is_prefilling:
            return (self.to_cache_len, self.to_cache_len)
        elif self.is_decoding:
            return (self.to_cache_len, self.to_cache_len + self.running_seq_start)

    @property
    def has_to_cache_blocks(self) -> bool:
        if self.is_prefilling:
            return True
        elif self.is_decoding:
            return len(self.dllm_block_buffer.to_cache_blocks) > 0

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
        return inspected_block.is_complete and inspected_block.is_last_in_context

    @property
    def pure_prefill_without_mask_token(self) -> bool:
        return self.is_prefilling and not self.is_padded

    def num_pages_with_seq_len(self: DllmReq, seq_len: int) -> int:
        return (seq_len + self.page_size - 1) // self.page_size

    def num_blocks_with_seq_len(self: DllmReq, seq_len: int) -> int:
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

    def force_deactivate(self):
        self.status = DllmReqStatus.COMPLETED

    def deactivate(self):
        if self.is_running:
            self.status = DllmReqStatus.COMPLETED

    def step(self):
        self.lazy_activate()
        # Condition to activate the next block, when buffer contains active blocks
        activate_cond = self.dllm_block_buffer.should_add_block and not self.dllm_block_buffer.is_overflow

        # Condition to activate the next block, when buffer is filled with dummy blocks
        # NOTE: maybe the status transfering logic is not robust enough, need to improve it
        #       this is a temporary solution to avoid the buffer being filled with dummy blocks
        #       which may cause infinite loop
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

    def postprocess(self):
        self.maybe_postprocess_prefix_blocks()
        block_id = 0
        while block_id < self.dllm_block_buffer.buffer_size:
            block = self.dllm_block_buffer.dllm_blocks[block_id]
            if block.is_active and block.is_complete and (block.prev_block.is_to_cache or block.prev_block.is_in_cache):
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

        if (
            self.eos_token_generated or self.max_new_tokens_reached or self.max_model_len_reached
        ) and self.last_block_finished:
            completed_blocks = [block.is_complete for block in self.dllm_block_buffer.valid_blocks]
            if all(completed_blocks):
                self.deactivate()
