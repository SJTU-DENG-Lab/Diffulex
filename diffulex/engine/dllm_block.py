"""DllmBlock and DllmBlockBuffer - extracted to avoid circular import with mixin."""

from __future__ import annotations

import weakref
import torch

from dataclasses import dataclass, field

from diffulex.config import DecodingThresholds
from diffulex.engine.status import DllmBlockStatus, DllmBlockType
from diffulex.mixin.edit.dllm_block import DllmBlockEditMixin

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffulex.engine.request import DllmReq
    from diffulex.engine.dllm_block import DllmBlockBuffer

weakref_fn = lambda x: weakref.ref(x) if x is not None else None


@dataclass
class DllmBlock(DllmBlockEditMixin):
    block_id: int
    start: int
    end: int
    block_size: int
    mask_token_id: int
    thresholds: DecodingThresholds

    status: DllmBlockStatus = None
    prev_block: "DllmBlock" = None

    block_type: DllmBlockType = DllmBlockType.IN_CONTEXT

    def __repr__(self):
        prev_block_id = self.prev_block.block_id if self.prev_block is not None else None
        return f"DllmBlock(block_id={self.block_id}, start={self.start}, end={self.end}, block_size={self.block_size}, mask_token_id={self.mask_token_id}, thresholds={self.thresholds}, status={self.status}, prev_block_id={prev_block_id}, is_last_in_context={self.is_last_in_context})"

    def post_init_dllm_block(self, req: "DllmReq", dllm_block_buffer: "DllmBlockBuffer"):
        assert self.end - self.start == self.block_size
        if not 0 <= int(self.editable_start) <= self.block_size:
            raise ValueError(
                f"editable_start must be in [0, {self.block_size}], got: {self.editable_start}"
            )

        if req is not None:
            self.bind_req(req)

        if dllm_block_buffer is not None:
            self.bind_buffer(dllm_block_buffer)

        if self.status is None:
            self.status = DllmBlockStatus.TO_CACHE if self.is_complete else DllmBlockStatus.ACTIVE
        self.observe_edit_state(self.token_ids)

        self.make_in_context()

    def bind_req(self, req: "DllmReq" | None):
        self._req = weakref_fn(req)

    def bind_buffer(self, dllm_block_buffer: "DllmBlockBuffer" | None):
        self._dllm_block_buffer = weakref_fn(dllm_block_buffer)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_req", None)
        state.pop("_dllm_block_buffer", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def req(self) -> "DllmReq":
        ref = getattr(self, "_req", None)
        return ref() if ref else None

    @property
    def dllm_block_buffer(self) -> "DllmBlockBuffer":
        ref = getattr(self, "_dllm_block_buffer", None)
        return ref() if ref else None

    @property
    def rel_page_id(self) -> int:
        return self.start // self.req.page_size
    
    @property
    def token_ids(self) -> list[int]:
        return self.req[self.start : self.end]

    @property
    def mask_token_relative_ids(self) -> list[int]:
        editable_start = int(self.editable_start)
        return [
            i
            for i, token_id in enumerate(self.token_ids)
            if i >= editable_start and token_id == self.mask_token_id
        ]

    @property
    def mask_token_global_ids(self) -> list[int]:
        editable_start = int(self.editable_start)
        return [
            i + self.start
            for i, token_id in enumerate(self.token_ids)
            if i >= editable_start and token_id == self.mask_token_id
        ]

    @property
    def in_buffer_block_id(self) -> int:
        return self.dllm_block_buffer.block_ids.index(self.block_id)

    @property
    def num_mask_tokens(self):
        return sum(token_id == self.mask_token_id for token_id in self.token_ids)

    @property
    def progress(self):
        return (self.block_size - self.num_mask_tokens) / self.block_size

    @property
    def is_complete(self):
        return self.progress == 1.0

    @property
    def is_semi_complete(self):
        return self.progress >= self.thresholds.semi_complete_threshold

    @property
    def should_force_decode_topk(self):
        return self.prev_block is not None and self.prev_block.is_semi_complete

    @property
    def should_add_block(self):
        return (
            self.progress >= self.thresholds.add_block_threshold
            and self.same_token_ratio >= self.thresholds.token_stability_threshold
            and not self.is_last_in_context
        )

    @property
    def is_dummy(self):
        return self.status == DllmBlockStatus.DUMMY

    @property
    def is_active(self):
        return self.status == DllmBlockStatus.ACTIVE

    @property
    def is_to_cache(self):
        return self.status == DllmBlockStatus.TO_CACHE

    @property
    def is_in_cache(self):
        return self.status == DllmBlockStatus.IN_CACHE

    @property
    def is_in_context(self):
        return self.block_type == DllmBlockType.IN_CONTEXT

    @property
    def is_out_of_context(self):
        return self.block_type == DllmBlockType.OUT_OF_CONTEXT

    @property
    def is_last_in_context(self):
        return self.block_type == DllmBlockType.LAST_IN_CONTEXT

    def write_token(self, token_id: int, rel_idx: int):
        if int(rel_idx) < int(self.editable_start):
            raise ValueError(
                f"Cannot write non-editable token in block {self.block_id}: "
                f"rel_idx={rel_idx}, editable_start={self.editable_start}"
            )
        self.req.token_ids[self.start + rel_idx] = token_id

    def write_tokens_parallel(self, token_ids: torch.Tensor, abs_ids: torch.Tensor):
        token_ids_list = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
        abs_ids_list = abs_ids.tolist() if isinstance(abs_ids, torch.Tensor) else list(abs_ids)
        for abs_idx, token_id in zip(abs_ids_list, token_ids_list):
            if int(abs_idx) - self.start < int(self.editable_start):
                raise ValueError(
                    f"Cannot write non-editable token in block {self.block_id}: "
                    f"abs_idx={abs_idx}, editable_start={self.editable_start}"
                )
            self.req.token_ids[int(abs_idx)] = int(token_id)

    def to_cache(self):
        if self.is_active:
            self.status = DllmBlockStatus.TO_CACHE

    def in_cache(self):
        if self.is_to_cache:
            self.status = DllmBlockStatus.IN_CACHE

    def make_in_context(self):
        self.block_type = DllmBlockType.IN_CONTEXT

    def make_out_of_context(self):
        self.block_type = DllmBlockType.OUT_OF_CONTEXT

    def make_last_in_context(self):
        if self.is_in_context:
            self.block_type = DllmBlockType.LAST_IN_CONTEXT


@dataclass
class DllmBlockBuffer:
    buffer_size: int
    dllm_blocks: list[DllmBlock] = field(default_factory=list)

    def __repr__(self):
        return f"DllmBlockBuffer(buffer_size={self.buffer_size}, dllm_blocks={self.dllm_blocks})"

    def post_init_dllm_block_buffer(self, req: "DllmReq"):
        assert len(self.dllm_blocks) == self.buffer_size

        if req is not None:
            self.bind_req(req)

        if len(self.dllm_blocks) > 0:
            for block in self.dllm_blocks:
                block.post_init_dllm_block(None, self)

    def bind_req(self, req: "DllmReq" | None):
        self._req = weakref_fn(req)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_req", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def req(self) -> "DllmReq":
        ref = getattr(self, "_req", None)
        return ref() if ref else None

    @property
    def buffer_sequence(self) -> list[int]:
        return self.req[self.dllm_blocks[0].start : self.dllm_blocks[-1].end]

    @property
    def buffer_position_ids(self) -> list[int]:
        return list(range(self.dllm_blocks[0].start, self.dllm_blocks[-1].end))

    @property
    def block_ids(self) -> list[int]:
        return [block.block_id for block in self.dllm_blocks]

    @property
    def cursor_slot_idx(self) -> int:
        return len(self.valid_blocks)

    @property
    def valid_blocks(self) -> list[DllmBlock]:
        return [block for block in self.dllm_blocks if not block.is_dummy]

    @property
    def dummy_blocks(self) -> list[DllmBlock]:
        return [block for block in self.dllm_blocks if block.is_dummy]

    @property
    def active_blocks(self) -> list[DllmBlock]:
        return [block for block in self.dllm_blocks if block.is_active]

    @property
    def to_cache_blocks(self) -> list[DllmBlock]:
        return [block for block in self.dllm_blocks if block.is_to_cache]

    @property
    def in_cache_blocks(self) -> list[DllmBlock]:
        return [block for block in self.dllm_blocks if block.is_in_cache]

    @property
    def first_running_block(self) -> DllmBlock:
        return self.dllm_blocks[0]

    @property
    def last_running_block(self) -> DllmBlock:
        return self.dllm_blocks[-1]

    @property
    def first_valid_block(self) -> DllmBlock:
        return self.dllm_blocks[0]

    @property
    def last_valid_block(self) -> DllmBlock:
        return self.dllm_blocks[self.cursor_slot_idx - 1]

    @property
    def first_to_cache_block(self) -> DllmBlock:
        return self.to_cache_blocks[0]

    @property
    def last_to_cache_block(self) -> DllmBlock:
        return self.to_cache_blocks[-1]

    @property
    def slot_block(self) -> DllmBlock:
        return self.dllm_blocks[self.cursor_slot_idx]

    @property
    def num_valid_blocks(self) -> int:
        return self.cursor_slot_idx

    @property
    def num_running_blocks(self) -> int:
        return self.buffer_size

    @property
    def should_add_block(self) -> bool:
        return self.last_valid_block.should_add_block

    @property
    def is_overflow(self) -> bool:
        return self.cursor_slot_idx >= self.buffer_size

    @property
    def prev_step_popped(self) -> bool:
        if len(self.dllm_blocks) < 2:
            return False
        return self.dllm_blocks[-1].block_id == self.dllm_blocks[-2].block_id

    def push_back(self, block: DllmBlock):
        self.dllm_blocks[-1] = block

    def pop_front(self):
        for i in range(0, self.buffer_size - 1):
            self.dllm_blocks[i] = self.dllm_blocks[i + 1]

    def activate_cursor_slot_block(self):
        self.slot_block.status = DllmBlockStatus.ACTIVE
        
    def maybe_fix_context_management(self):
        if self.first_valid_block.is_dummy and self.first_valid_block.is_last_in_context:
            self.first_valid_block.prev_block.make_last_in_context()
