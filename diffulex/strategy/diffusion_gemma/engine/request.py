from __future__ import annotations

import random

from diffulex.attention.metadata import is_warming_up
from diffulex.config import Config
from diffulex.engine.dllm_block import DllmBlock, DllmBlockBuffer
from diffulex.engine.request import AutoReq, DllmReq
from diffulex.engine.status import DllmBlockStatus
from diffulex.sampling_params import SamplingParams


@AutoReq.register("diffusion_gemma")
class DiffusionGemmaReq(DllmReq):
    """Request state for DiffusionGemma's block/canvas decoding."""

    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams = SamplingParams(),
        config: Config | None = None,
    ):
        super().__init__(token_ids, sampling_params)

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
        self.gemma_block_vocab_size = int(
            getattr(config, "tokenizer_vocab_size", None)
            or getattr(config.hf_config, "vocab_size", 0)
            or getattr(getattr(config.hf_config, "text_config", None), "vocab_size", 0)
            or 0
        )

        self.dllm_blocks: list[DllmBlock] = []
        self.dllm_block_buffer: DllmBlockBuffer = None

        if self.max_model_len_reached and not is_warming_up():
            self.force_deactivate()
            return

        self.prefix_len = len(self.token_ids)
        self.padded_prefix_len = (
            ((self.prefix_len + self.block_size - 1) // self.block_size) * self.block_size
            if self.prefix_len > 0
            else 0
        )
        padding_len = self.padded_prefix_len - self.prefix_len
        if padding_len > 0:
            self.pad_tokens(padding_len)

        num_prefix_blocks = self.padded_prefix_len // self.block_size
        for i in range(num_prefix_blocks):
            start = i * self.block_size
            end = start + self.block_size
            dllm_block = DllmBlock(
                block_id=i,
                start=start,
                end=end,
                block_size=self.block_size,
                mask_token_id=self.mask_token_id,
                thresholds=self.thresholds,
                status=DllmBlockStatus.TO_CACHE,
                prev_block=None if i == 0 else self.dllm_blocks[-1],
                editable_start=self.block_size,
            )
            dllm_block.post_init_dllm_block(self, None)
            dllm_block.commit_ready = True
            self.dllm_blocks.append(dllm_block)

        block_id = len(self.dllm_blocks)
        start = block_id * self.block_size
        end = start + self.block_size
        self.extend_gemma_block_tokens()
        active_block = DllmBlock(
            block_id=block_id,
            start=start,
            end=end,
            block_size=self.block_size,
            mask_token_id=self.mask_token_id,
            thresholds=self.thresholds,
            status=DllmBlockStatus.ACTIVE,
            prev_block=None if not self.dllm_blocks else self.dllm_blocks[-1],
            editable_start=0,
        )
        active_block.post_init_dllm_block(self, None)
        active_block.commit_ready = False
        active_block.valid_commit_len = self._gemma_block_next_commit_len()
        self.dllm_blocks.append(active_block)

        self.dllm_block_buffer = DllmBlockBuffer(
            buffer_size=self.buffer_size,
            dllm_blocks=[active_block],
        )
        self.dllm_block_buffer.post_init_dllm_block_buffer(self)
        for block in self.dllm_block_buffer.dllm_blocks:
            block.commit_ready = False

    def _random_gemma_block_tokens(self, n: int) -> list[int]:
        vocab_size = int(getattr(self, "gemma_block_vocab_size", 0) or 0)
        if vocab_size <= 0:
            return [self.mask_token_id] * n
        return [random.randrange(vocab_size) for _ in range(n)]

    def extend_gemma_block_tokens(self):
        self.token_ids.extend(self._random_gemma_block_tokens(self.block_size))

    def on_block_token_rewrite(self, block, rel_idx: int, old_token: int, new_token: int) -> None:
        del block, rel_idx, old_token, new_token

    @property
    def gemma_block_committed_len(self) -> int:
        total = 0
        for block in self.dllm_blocks:
            if block.start < self.padded_prefix_len:
                continue
            if block.is_in_cache:
                total += int(getattr(block, "valid_commit_len", block.block_size))
        return total

    def _gemma_block_next_commit_len(self) -> int:
        committed = int(getattr(self, "gemma_block_committed_len", 0))
        remaining_new = max(0, int(self.max_new_tokens) - committed)
        remaining_model = max(0, int(self.max_model_len) - int(self.prefix_len) - committed)
        return min(int(self.block_size), remaining_new, remaining_model)

    @property
    def eos_token_generated(self) -> bool:
        if self.ignore_eos:
            return False
        return self.eos_token_id in self.full_response

    @property
    def num_prefix_blocks(self) -> int:
        return self.padded_prefix_len // self.block_size

    @property
    def max_new_tokens_reached(self) -> bool:
        return self.gemma_block_committed_len >= self.max_new_tokens

    @property
    def max_model_len_reached(self) -> bool:
        if hasattr(self, "prefix_len"):
            return self.prefix_len + self.gemma_block_committed_len >= self.max_model_len
        return len(self.token_ids) >= self.max_model_len

    @property
    def running_sequence(self) -> list[int]:
        if self.is_prefilling:
            context_len = min(self.contiguous_in_cache_prefix_len, self.prefix_len)
            return self.token_ids[context_len : self.prefix_len]
        if self.is_decoding or self.is_completed:
            return self.dllm_block_buffer.buffer_sequence

    @property
    def running_position_ids(self) -> list[int]:
        if self.is_prefilling:
            context_len = min(self.contiguous_in_cache_prefix_len, self.prefix_len)
            return list(range(context_len, self.prefix_len))
        if self.is_decoding or self.is_completed:
            # We keep generated canvases physically page-aligned in token_ids/KV
            # storage, but DiffusionGemma's RoPE positions should follow the
            # true sequence without the prompt-padding hole.
            prefix_padding = int(self.padded_prefix_len) - int(self.prefix_len)
            return [pos - prefix_padding for pos in self.dllm_block_buffer.buffer_position_ids]

    @property
    def truncated_response(self) -> list[int]:
        generated_seq = self.full_response
        if self.eos_token_id in generated_seq and not self.ignore_eos:
            generated_seq = generated_seq[: generated_seq.index(self.eos_token_id)]
        if self.max_new_tokens_reached:
            generated_seq = generated_seq[: self.max_new_tokens]
        return generated_seq

    @property
    def full_response(self) -> list[int]:
        tokens: list[int] = []
        for block in self.dllm_blocks:
            if block.start < self.padded_prefix_len:
                continue
            if not block.is_in_cache:
                continue
            valid_len = int(getattr(block, "valid_commit_len", block.block_size))
            if valid_len <= 0:
                continue
            tokens.extend(self.token_ids[block.start : block.start + valid_len])
        return tokens

    @property
    def running_len(self) -> int:
        if self.is_prefilling:
            return self.prefix_len
        if self.is_pending or self.is_waiting:
            padded_running_len = (
                self.padded_prefix_len - self.block_size
            ) + self.dllm_block_buffer.num_valid_blocks * self.block_size
            if self.dllm_block_buffer.dllm_blocks[0].should_add_block:
                padded_running_len += self.block_size
            return padded_running_len if self.is_padded else self.prefix_len + self.block_size
        if self.is_decoding:
            return self.dllm_block_buffer.num_running_blocks * self.block_size

    @property
    def valid_len(self) -> int:
        if self.is_prefilling:
            return len(self.running_sequence)
        if self.is_decoding:
            return self.dllm_block_buffer.num_valid_blocks * self.block_size

    def push_back_dummy_block(self):
        self.extend_gemma_block_tokens()
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
        dllm_block.commit_ready = False
        dllm_block.valid_commit_len = self._gemma_block_next_commit_len()
        if (self.max_new_tokens_reached or self.max_model_len_reached) and dllm_block.prev_block.is_in_context:
            dllm_block.make_last_in_context()
        elif dllm_block.prev_block.is_out_of_context or dllm_block.prev_block.is_last_in_context:
            dllm_block.make_out_of_context()

        self.dllm_blocks.append(dllm_block)
        self.dllm_block_buffer.push_back(dllm_block)

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
                if block.start >= self.padded_prefix_len:
                    self.new_tokens += int(getattr(block, "valid_commit_len", block.block_size))
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
