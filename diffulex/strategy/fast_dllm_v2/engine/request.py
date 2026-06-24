from __future__ import annotations

from enum import IntEnum

from diffulex.attention.metadata import is_warming_up
from diffulex.config import Config
from diffulex.engine.request import AutoReq, DllmReq
from diffulex.engine.dllm_block import DllmBlock, DllmBlockBuffer
from diffulex.engine.status import DllmBlockStatus
from diffulex.sampling_params import SamplingParams


class FastDLLMV2Mode(IntEnum):
    FULL_BUFFER_INIT = 0
    SUB_BLOCK_REFINE = 1
    FINAL_COMMIT = 2


@AutoReq.register("fast_dllm_v2")
class FastDLLMV2Req(DllmReq):
    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams = SamplingParams(),
        config: Config | None = None,
    ):
        super().__init__(token_ids, sampling_params)
        self.fdv2_current_buffer_initialized = False
        self.fdv2_current_sub_block_idx = 0
        self.fdv2_mode = FastDLLMV2Mode.FULL_BUFFER_INIT
        self.fdv2_prefix_cache_len = 0
        self.fdv2_pending_next_token_id: int | None = None
        self.fdv2_use_block_cache = True

    def init_multi_block(self, config: Config):
        self.is_multi_block = True
        self.status_history = [self.status]
        self.completion_reason = None
        self._resume_prefill_until = 0
        self._terminal_context_block_id: int | None = None

        self.block_size = config.block_size
        self.buffer_size = config.buffer_size
        strategy_config = getattr(config, "strategy", None)
        self.fdv2_use_block_cache = bool(getattr(strategy_config, "use_block_cache", True))
        self.mask_token_id = config.mask_token_id
        self.thresholds = config.decoding_thresholds
        self.eos_token_id = config.eos
        self.max_model_len = config.max_model_len
        self.max_new_tokens = self.max_tokens
        self.auto_max_nfe_enabled = self.max_nfe is None
        self.auto_max_nfe_warmup_steps = int(getattr(config, "auto_max_nfe_warmup_steps", 8))
        self.auto_max_nfe_tpf_floor = float(getattr(config, "auto_max_nfe_tpf_floor", 1.0))
        self.auto_max_nfe_token_count = 0
        self.auto_max_nfe_value = None
        self.auto_max_nfe_avg_tpf = None

        self.dllm_blocks: list[DllmBlock] = []
        self.dllm_block_buffer: DllmBlockBuffer | None = None

        if self.max_model_len_reached and not is_warming_up():
            self.force_deactivate()
            return

        self.prefix_len = len(self.token_ids)
        self.padded_prefix_len = self.prefix_len

        fdv2_block_size = self.block_size * self.buffer_size
        self.fdv2_prefix_cache_len = (self.prefix_len // fdv2_block_size) * fdv2_block_size
        buffer_start = self.fdv2_prefix_cache_len
        buffer_end = buffer_start + fdv2_block_size

        if len(self.token_ids) < buffer_end:
            self.pad_tokens(buffer_end - len(self.token_ids))

        for i in range(self.fdv2_prefix_cache_len // self.block_size):
            start = i * self.block_size
            end = start + self.block_size
            dllm_block = DllmBlock(
                block_id=i,
                start=start,
                end=end,
                block_size=self.block_size,
                mask_token_id=self.mask_token_id,
                thresholds=self.thresholds,
                status=None,
                prev_block=None if i == 0 else self.dllm_blocks[-1],
            )
            dllm_block.post_init_dllm_block(self, None)
            self.dllm_blocks.append(dllm_block)

        for _ in range(self.buffer_size):
            block_id = len(self.dllm_blocks)
            start = buffer_start + (block_id - self.fdv2_prefix_cache_len // self.block_size) * self.block_size
            end = start + self.block_size
            editable_start = max(0, min(self.block_size, self.prefix_len - start))
            dllm_block = DllmBlock(
                block_id=block_id,
                start=start,
                end=end,
                block_size=self.block_size,
                mask_token_id=self.mask_token_id,
                thresholds=self.thresholds,
                status=DllmBlockStatus.ACTIVE,
                prev_block=None if block_id == 0 else self.dllm_blocks[-1],
                editable_start=editable_start,
            )
            dllm_block.post_init_dllm_block(self, None)
            dllm_block.commit_ready = False
            self.dllm_blocks.append(dllm_block)

        self.dllm_block_buffer = DllmBlockBuffer(
            buffer_size=self.buffer_size,
            dllm_blocks=self.dllm_blocks[-self.buffer_size :],
        )
        self.dllm_block_buffer.post_init_dllm_block_buffer(self)
        for block in self.fdv2_buffer_blocks:
            block.commit_ready = False

    @property
    def num_prefix_blocks(self) -> int:
        return int(self.fdv2_prefix_cache_len) // int(self.block_size)

    @property
    def num_prefix_pages(self) -> int:
        return self.num_pages_with_seq_len(self.fdv2_prefix_cache_len)

    @property
    def fdv2_buffer_blocks(self):
        if self.dllm_block_buffer is None:
            return []
        return self.dllm_block_buffer.dllm_blocks

    @property
    def fdv2_current_sub_block(self):
        return self.fdv2_buffer_blocks[self.fdv2_current_sub_block_idx]

    @property
    def fdv2_buffer_start(self) -> int:
        return self.dllm_block_buffer.first_running_block.start

    @property
    def fdv2_buffer_end(self) -> int:
        return self.dllm_block_buffer.last_running_block.end

    @property
    def fdv2_buffer_token_ids(self) -> list[int]:
        return self.token_ids[self.fdv2_buffer_start : self.fdv2_buffer_end]

    @property
    def fdv2_buffer_position_ids(self) -> list[int]:
        return list(range(self.fdv2_buffer_start, self.fdv2_buffer_end))

    @property
    def fdv2_replace_position(self) -> int:
        return int(self.fdv2_current_sub_block_idx) * int(self.block_size)

    @property
    def fdv2_read_cache_len(self) -> int:
        return self.fdv2_buffer_end

    @property
    def fdv2_read_cache_pages(self) -> int:
        return self.num_pages_with_seq_len(self.fdv2_read_cache_len)

    @property
    def fdv2_committed_prefix_len(self) -> int:
        return self.contiguous_in_cache_prefix_len

    @property
    def fdv2_sub_block_complete(self) -> bool:
        return self.fdv2_current_sub_block.is_complete

    @property
    def fdv2_current_sub_block_first_token_is_mask(self) -> bool:
        return self.fdv2_current_sub_block.token_ids[0] == self.mask_token_id

    @property
    def fdv2_buffer_complete(self) -> bool:
        return all(block.is_complete for block in self.fdv2_buffer_blocks)

    def refresh_fdv2_mode(self) -> None:
        if not self.is_decoding:
            return
        while (
            self.fdv2_current_sub_block_idx < self.buffer_size - 1
            and self.fdv2_current_sub_block.is_complete
        ):
            self.fdv2_current_sub_block_idx += 1
        if not self.fdv2_current_buffer_initialized:
            self.fdv2_mode = FastDLLMV2Mode.FULL_BUFFER_INIT
            return
        if self.fdv2_buffer_complete:
            self.fdv2_mode = FastDLLMV2Mode.FINAL_COMMIT
            return
        if self.fdv2_current_sub_block_first_token_is_mask:
            self.fdv2_mode = FastDLLMV2Mode.FULL_BUFFER_INIT
        else:
            self.fdv2_mode = FastDLLMV2Mode.SUB_BLOCK_REFINE

    @property
    def running_sequence(self) -> list[int]:
        if self.is_prefilling:
            return super().running_sequence
        if self.is_decoding or self.is_completed:
            if self.fdv2_mode == FastDLLMV2Mode.SUB_BLOCK_REFINE and self.fdv2_use_block_cache:
                return self.fdv2_current_sub_block.token_ids
            return self.fdv2_buffer_token_ids
        return []

    @property
    def running_position_ids(self) -> range | list[int]:
        if self.is_prefilling:
            return super().running_position_ids
        if self.is_decoding or self.is_completed:
            if self.fdv2_mode == FastDLLMV2Mode.SUB_BLOCK_REFINE and self.fdv2_use_block_cache:
                return list(range(self.fdv2_current_sub_block.start, self.fdv2_current_sub_block.end))
            return self.fdv2_buffer_position_ids
        return []

    @property
    def running_len(self) -> int:
        if self.is_prefilling:
            return int(self.fdv2_prefix_cache_len)
        if self.is_decoding:
            if self.fdv2_mode == FastDLLMV2Mode.SUB_BLOCK_REFINE and self.fdv2_use_block_cache:
                return self.block_size
            return self.chunk_size
        return super().running_len

    @property
    def valid_len(self) -> int:
        if self.is_prefilling:
            return int(self.fdv2_prefix_cache_len)
        if self.is_decoding:
            if self.fdv2_mode == FastDLLMV2Mode.SUB_BLOCK_REFINE and self.fdv2_use_block_cache:
                return self.block_size
            return self.chunk_size
        return super().valid_len

    @property
    def cache_len(self) -> int:
        if self.is_decoding:
            return max(super().cache_len, self.fdv2_read_cache_len)
        return super().cache_len

    def step(self):
        if self.dllm_block_buffer is None:
            self.lazy_activate()
            return
        super().step()
        if not self.is_decoding:
            return
        for block in self.fdv2_buffer_blocks:
            if block.is_dummy:
                block.status = DllmBlockStatus.ACTIVE
        self.refresh_fdv2_mode()

    def _seed_next_block_from_pending_token(self) -> None:
        token_id = self.fdv2_pending_next_token_id
        if token_id is None:
            return
        if self.max_new_tokens_reached or self.max_model_len_reached:
            self.fdv2_pending_next_token_id = None
            return

        for block in self.fdv2_buffer_blocks:
            if int(getattr(block, "editable_start", 0)) > 0:
                continue
            if not block.token_ids or block.token_ids[0] != self.mask_token_id:
                continue
            block.write_token(int(token_id), 0)
            self.new_tokens += 1
            self.fdv2_pending_next_token_id = None
            return

    def postprocess(self):
        if self.dllm_block_buffer is None:
            return

        if self.is_decoding and self.fdv2_mode == FastDLLMV2Mode.FULL_BUFFER_INIT:
            self.fdv2_current_buffer_initialized = True
            self.refresh_fdv2_mode()
            return

        if self.is_decoding and self.fdv2_mode == FastDLLMV2Mode.SUB_BLOCK_REFINE:
            if self.fdv2_current_sub_block.is_complete and self.fdv2_current_sub_block_idx < self.buffer_size - 1:
                self.fdv2_current_sub_block_idx += 1
            self.refresh_fdv2_mode()
            return

        if self.is_decoding and self.fdv2_mode == FastDLLMV2Mode.FINAL_COMMIT:
            for block in self.fdv2_buffer_blocks:
                if block.is_active:
                    block.commit_ready = True
                    block.status = DllmBlockStatus.TO_CACHE
            super().postprocess()
            self.fdv2_current_buffer_initialized = False
            self.fdv2_current_sub_block_idx = 0
            self.fdv2_mode = FastDLLMV2Mode.FULL_BUFFER_INIT
            self._seed_next_block_from_pending_token()
            return

        super().postprocess()
        self._seed_next_block_from_pending_token()
