from __future__ import annotations

from multiprocessing.synchronize import Event

import torch

from diffulex.attention.metadata import set_fetch_fn_for_attn_metadata
from diffulex.config import Config
from diffulex.engine.model_runner import AutoModelRunner
from diffulex.strategy.dmax.attention.metadata import (
    fetch_dmax_attn_metadata,
    reset_dmax_attn_metadata,
    set_dmax_attn_metadata,
)
from diffulex.strategy.dmax.engine.request import DMaxReq
from diffulex.strategy_template.token_merging_multi_block.engine.model_runner import (
    TokenMergingMultiBlockModelRunnerTemplate,
)


@AutoModelRunner.register("dmax")
class DMaxModelRunner(TokenMergingMultiBlockModelRunnerTemplate):
    """DMax-style block diffusion runner with metadata-driven token merging."""

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        set_fetch_fn_for_attn_metadata(fetch_dmax_attn_metadata)
        self.init_attn_metadata_fn(
            set_dmax_attn_metadata,
            reset_dmax_attn_metadata,
            fetch_dmax_attn_metadata,
        )
        self.mask_token_id = config.mask_token_id
        self.is_prefix_full = config.multi_block_prefix_full
        config.enforce_eager = True
        super().__init__(config, rank, event)

    def prepare_prefill(self, reqs: list[DMaxReq]):
        self.prepare_chunked_prefill_token_merging_multi_block(reqs)

    def prepare_decode(self, reqs: list[DMaxReq]):
        self.prepare_decode_multi_block(reqs)

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor):
        self.run_model_multi_block(input_ids, positions)

    def run(self, reqs: list[DMaxReq]) -> list[int]:
        return self.run_multi_block(reqs)

    @torch.inference_mode()
    def capture_cudagraph(self):
        return None
