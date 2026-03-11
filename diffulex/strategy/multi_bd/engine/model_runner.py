from __future__ import annotations

import time

from multiprocessing.synchronize import Event

import torch
from tqdm import tqdm

from diffulex.config import Config
from diffulex.engine.request import DllmReq
from diffulex.attention.metadata import set_fetch_fn_for_attn_metadata, set_warming_up, reset_warming_up
from diffulex.engine.model_runner import AutoModelRunner, ModelRunnerBase
from diffulex.strategy.multi_bd.attention.metadata import (
    fetch_multi_bd_attn_metadata,
    set_multi_bd_attn_metadata,
    reset_multi_bd_attn_metadata,
)


@AutoModelRunner.register("multi_bd", is_default=True)
class MultiBDModelRunner(ModelRunnerBase):
    """Reference implementation of Multi-Block Diffusion decoding strategy."""

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        set_fetch_fn_for_attn_metadata(fetch_multi_bd_attn_metadata)
        self.init_attn_metadata_fn(
            set_multi_bd_attn_metadata, reset_multi_bd_attn_metadata, fetch_multi_bd_attn_metadata
        )
        self.mask_token_id = config.mask_token_id
        self.is_prefix_full = False

        super().__init__(config, rank, event)

    def prepare_prefill(self, reqs: list[DllmReq]):
        self.prepare_chunked_prefill_multi_block(reqs)

    def prepare_decode(self, reqs: list[DllmReq]):
        self.prepare_decode_multi_block(reqs)

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor):
        self.run_model_multi_block(input_ids, positions)

    def run(self, reqs: list[DllmReq]) -> list[int]:
        return self.run_multi_block(reqs)

    @torch.inference_mode()
    def capture_cudagraph(self):
        self.capture_cudagraph_multi_block()
