from __future__ import annotations

from diffulex.config import Config
from diffulex.engine.dllm_block import DllmBlock, DllmBlockBuffer
from diffulex.engine.request import DllmReq
from diffulex.engine.status import DllmBlockStatus, DllmReqStatus
from diffulex.attention.metadata import is_warming_up


class MultiBlockReqTemplate(DllmReq):
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

    def init_multi_block(self: DllmReq, config: Config):
        self.is_multi_block = True
        self.status_history = [self.status]
        self.completion_reason = None
        self._resume_prefill_until = 0
        self._terminal_context_block_id: int | None = None

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
                status=None,  # let Block __post_init__ to set the status
                prev_block=None if i == 0 else self.dllm_blocks[-1],
                editable_start=editable_start,
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
        if self.ignore_eos:
            return False
        generated_seq = self.token_ids[self.prefix_len :]
        return self.eos_token_id in generated_seq

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
    def max_nfe_reached(self) -> bool:
        return self.max_nfe is not None and self.nfe >= self.max_nfe

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
        elif self.is_decoding or self.is_completed:
            return self.dllm_block_buffer.buffer_sequence

    @property
    def running_position_ids(self) -> list[int]:
        if self.is_prefilling:
            return range(self.contiguous_in_cache_prefix_len, self.running_len)
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
            if self._resume_prefill_until > 0:
                return self._resume_prefill_until
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
        if self.is_prefilling or self.is_pending or self.is_waiting:
            # return self.padded_prefix_len - self.block_size if self.is_padded else self.prefix_len
            return sum(b.is_to_cache for b in self.dllm_blocks) * self.block_size
        elif self.is_decoding or self.is_preempted:
            return len(self.dllm_block_buffer.to_cache_blocks) * self.block_size

    @property
    def in_cache_len(self) -> int:
        return sum(block.block_size for block in self.dllm_blocks if block.is_in_cache)

    @property
    def contiguous_in_cache_prefix_len(self) -> int:
        """Length of the contiguous cached prefix from block 0.

        `in_cache_len` counts all IN_CACHE blocks and can be larger than the true
        left-prefix when state transitions leave non-prefix cached blocks. For
        prefill q/pos alignment we require the contiguous prefix only.
        """
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
            return self._prefill_visible_to_cache_last_global_id() is not None
        elif self.is_decoding:
            return len(self.dllm_block_buffer.to_cache_blocks) > 0

    @property
    def has_to_cache_block(self) -> bool:
        return self.has_to_cache_blocks

    @property
    def to_cache_last_token_id(self) -> int:
        if self.is_prefilling:
            window_start = int(self.contiguous_in_cache_prefix_len)
            last_global = self._prefill_visible_to_cache_last_global_id()
            if last_global is None:
                return 0
            return int(last_global - window_start)
        n = len(self.dllm_block_buffer.to_cache_blocks) * self.block_size
        return n - 1 if n > 0 else 0

    def _prefill_visible_to_cache_last_global_id(self) -> int | None:
        if not self.is_prefilling:
            return None

        window_start = int(self.contiguous_in_cache_prefix_len)
        window_end = int(self.running_len)
        if window_end <= window_start:
            return None

        last_global = None
        for block in self.dllm_blocks:
            if block.end <= window_start:
                continue
            if block.start >= window_end:
                break
            if not block.is_to_cache:
                continue
            candidate = min(block.end, window_end) - 1
            if candidate < window_start:
                continue
            last_global = candidate if last_global is None else max(last_global, candidate)
        return last_global

    @property
    def last_block_finished(self) -> bool:
        terminal_block = self.terminal_context_block
        return terminal_block is not None and terminal_block.is_complete

    @property
    def terminal_context_block(self) -> DllmBlock | None:
        block_id = getattr(self, "_terminal_context_block_id", None)
        if block_id is None or not (0 <= int(block_id) < len(self.dllm_blocks)):
            return None
        return self.dllm_blocks[int(block_id)]

    def set_terminal_context_block(self, block: DllmBlock | None) -> None:
        # The terminal context boundary is request state and must always point to a
        # real block. Dummy blocks are only buffer-capacity placeholders.
        while block is not None and block.is_dummy:
            block = block.prev_block
        if block is None:
            self._terminal_context_block_id = None
            return

        terminal_block_id = int(block.block_id)
        self._terminal_context_block_id = terminal_block_id

        for dllm_block in self.dllm_blocks:
            if dllm_block.is_dummy or dllm_block.block_id > terminal_block_id:
                dllm_block.make_out_of_context()
            elif dllm_block.block_id < terminal_block_id:
                dllm_block.make_in_context()
            else:
                dllm_block.make_in_context()
                dllm_block.make_last_in_context()

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
        if self.is_multi_block:
            rebuild_until = 0
            if self.is_decoding:
                rebuild_until = int(self.running_seq_start)
                self._resume_prefill_until = rebuild_until
            # Scheduler preemption frees this req's page_table immediately after moving it
            # back to WAITING. Any block still marked IN_CACHE would then point to KV pages
            # that no longer belong to the req. Demote those blocks back to TO_CACHE so a
            # resumed req rebuilds its cached prefix instead of reusing stale block state.
            for block in self.dllm_blocks:
                if rebuild_until > 0 and block.end <= rebuild_until and block.is_to_cache:
                    continue
                if block.is_in_cache:
                    block.status = DllmBlockStatus.TO_CACHE
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
        if self._resume_prefill_until > 0:
            self.status = DllmReqStatus.PREFILLING
            return
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

        # Decode can livelock when the running buffer is fully occupied by
        # IN_CACHE blocks (no ACTIVE/TO_CACHE block left). In that state we keep
        # scheduling decode, but there is no writable frontier so no new token is
        # ever accepted. Recycle the cached head block to reopen one slot.
        if (
            self.is_decoding
            and not self.dllm_block_buffer.active_blocks
            and not self.dllm_block_buffer.to_cache_blocks
        ):
            head_block = self.dllm_block_buffer.first_running_block
            if head_block.is_in_cache and not head_block.is_last_in_context:
                head_block.status = DllmBlockStatus.TO_CACHE
        
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
            editable_start=0,
        )

        dllm_block.post_init_dllm_block(self, self.dllm_block_buffer)
        if (
            self.max_new_tokens_reached
            or self.max_model_len_reached
            or self.terminal_context_block is not None
            or dllm_block.prev_block.is_out_of_context
            or dllm_block.prev_block.is_last_in_context
        ):
            dllm_block.make_out_of_context()

        self.dllm_blocks.append(dllm_block)
        self.dllm_block_buffer.push_back(dllm_block)

    def maybe_postprocess_prefix_blocks(self):
        if not self.is_prefilling:
            return

        for block_id in range(self.num_prefix_blocks):
            self.dllm_blocks[block_id].in_cache()

        if self._resume_prefill_until > 0:
            for block in self.dllm_blocks:
                if block.end > self._resume_prefill_until:
                    break
                if block.is_to_cache and block.is_complete:
                    block.in_cache()

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

        if self._resume_prefill_until > 0 and self.contiguous_in_cache_prefix_len >= self._resume_prefill_until:
            self._resume_prefill_until = 0

        if self.is_truncated:
            self.set_terminal_context_block(self.dllm_block_buffer.last_valid_block)

        if self.eos_token_generated:
            self.meet_eos = True

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
