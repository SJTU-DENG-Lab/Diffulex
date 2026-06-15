from __future__ import annotations

import torch

from dataclasses import dataclass

from diffulex.config import GEMMA_BLOCK_SIZE, SUPPORTED_PAGE_BLOCK_SIZES
from diffulex.attention.metadata import AttnMetaDataBase


@dataclass
class MultiBlockAttnMetaDataTemplate(AttnMetaDataBase):
    def init_multi_block(
        self: AttnMetaDataBase,
        valid_slices: torch.Tensor | None = None,
        buffer_size: int = 1,
        is_prefix_full: bool = False,
        status_table: torch.Tensor | None = None,
        prefix_lens: torch.Tensor | None = None,
        padded_prefix_lens: torch.Tensor | None = None,
        mask_prefix_hole: bool = False,
        prefix_causal: bool = False,
    ):
        self.use_multi_block = True
        self.valid_slices = valid_slices
        self.buffer_size = buffer_size
        self.is_block_causal = True
        self.is_prefix_full = is_prefix_full
        self.status_table = status_table
        self.prefix_lens = prefix_lens
        self.padded_prefix_lens = padded_prefix_lens
        self.mask_prefix_hole = mask_prefix_hole
        self.prefix_causal = prefix_causal

        supported_sizes = SUPPORTED_PAGE_BLOCK_SIZES + (GEMMA_BLOCK_SIZE,)

        if self.page_size not in supported_sizes:
            raise ValueError(
                f"page_size must be one of {supported_sizes}, got {self.page_size}"
            )
        if self.block_size not in supported_sizes:
            raise ValueError(
                f"block_size must be one of {supported_sizes}, got {self.block_size}"
            )
        if self.block_size > self.page_size:
            raise ValueError(
                f"block_size {self.block_size} must be <= page_size {self.page_size}"
            )
