from __future__ import annotations


from multiprocessing.synchronize import Event

from diffulex.config import Config
from diffulex.attention.metadata import set_fetch_fn_for_attn_metadata
from diffulex.engine.model_runner import AutoModelRunner, ModelRunnerBase
from diffulex.strategy.d2f.attention.metadata import (
    fetch_d2f_attn_metadata,
    set_d2f_attn_metadata,
    reset_d2f_attn_metadata,
)


@AutoModelRunner.register("d2f", is_default=True)
class D2fModelRunner(ModelRunnerBase):
    """Reference implementation of Multi-Block Diffusion decoding strategy."""

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        set_fetch_fn_for_attn_metadata(fetch_d2f_attn_metadata)
        self.init_attn_metadata_fn(set_d2f_attn_metadata, reset_d2f_attn_metadata, fetch_d2f_attn_metadata)
        self.mask_token_id = config.mask_token_id
        self.is_prefix_full = True

        super().__init__(config, rank, event)
