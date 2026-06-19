from __future__ import annotations

from multiprocessing.synchronize import Event

import torch

from diffulex.attention.metadata import set_fetch_fn_for_attn_metadata
from diffulex.config import Config
from diffulex.engine.model_runner import AutoModelRunner, ModelRunnerBase
from diffulex.engine.request import DllmReq
from diffulex.logger import get_logger
from diffulex.profiling import record_function
from diffulex.strategy.diffusion_gemma.attention.metadata import (
    fetch_diffusion_gemma_attn_metadata,
    reset_diffusion_gemma_attn_metadata,
    set_diffusion_gemma_attn_metadata,
)

logger = get_logger(__name__)


@AutoModelRunner.register("diffusion_gemma")
class DiffusionGemmaModelRunner(ModelRunnerBase):
    """DiffusionGemma runner for block/canvas decoding."""

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        set_fetch_fn_for_attn_metadata(fetch_diffusion_gemma_attn_metadata)
        self.init_attn_metadata_fn(
            set_diffusion_gemma_attn_metadata,
            reset_diffusion_gemma_attn_metadata,
            fetch_diffusion_gemma_attn_metadata,
        )
        self.mask_token_id = config.mask_token_id
        self.is_prefix_full = config.multi_block_prefix_full
        self.mask_prefix_hole = True
        self.prefix_causal = True
        super().__init__(config, rank, event)

    def _prepare_prefill_req(self, req: DllmReq):
        context_len = min(self._cached_prefix_len(req), int(req.prefix_len))
        input_ids = list(req.token_ids[context_len : req.prefix_len])
        q_len = len(input_ids)
        positions = list(range(context_len, req.prefix_len))

        slot_mapping = []
        for pos in range(context_len, int(req.prefix_len)):
            rel_page_id = pos // self.page_size
            if rel_page_id >= len(req.page_table):
                slot_mapping.append(-1)
                continue
            abs_page_id = req.page_table[rel_page_id]
            slot_mapping.append(abs_page_id * self.page_size + pos % self.page_size)

        return dict(
            input_ids=input_ids,
            positions=positions,
            context_len=context_len,
            seqlen_q=q_len,
            seqlen_k=q_len,
            valid_slice=q_len,
            slot_mapping=slot_mapping,
            status=0,
            prefix_len=req.prefix_len,
            padded_prefix_len=req.padded_prefix_len,
        )

    @torch.inference_mode()
    def run_model_multi_block(self, input_ids: torch.Tensor, positions: torch.Tensor):
        with record_function("diffulex.diffusion_gemma.model_forward"):
            return self.model.compute_logits(self.model(input_ids, positions))

    def _before_multi_block_model_forward(self, local_reqs: list[DllmReq]) -> None:
        set_context = getattr(self.model, "set_self_conditioning_context", None)
        if set_context is None:
            return

        get_embeds = getattr(self.sampler, "get_self_conditioning_embeds", None)
        if get_embeds is None:
            set_context(None)
            return

        context: list[dict] = []
        token_offset = 0
        for req in local_reqs:
            seq_len = len(req.running_sequence)
            if req.is_decoding:
                first_start = int(req.dllm_block_buffer.first_running_block.start)
                for block in req.dllm_block_buffer.active_blocks:
                    soft_embeds = get_embeds(req.req_id, block.block_id)
                    if soft_embeds is None:
                        continue
                    local_start = token_offset + int(block.start) - first_start
                    local_end = local_start + int(block.block_size)
                    context.append(
                        {
                            "start": local_start,
                            "end": local_end,
                            "soft_embeds": soft_embeds,
                        }
                    )
            token_offset += seq_len

        set_context(context or None)

    def _sample_multi_block_outputs(
        self,
        local_reqs: list[DllmReq],
        logits: torch.Tensor,
        temperatures: torch.Tensor | None,
    ):
        # Self-conditioning requires every TP rank to consume full-vocab logits
        # and enter the sampler's all-reduce. Only rank 0 returns scheduler state.
        local_sample_output = self.sampler(local_reqs, logits, temperatures)
        return local_sample_output if self.is_model_parallel_root else None

    @torch.inference_mode()
    def capture_cudagraph_multi_block(self):
        logger.info("Skipping CUDA graph capture for DiffusionGemma until self-conditioning graph buffers are wired.")
        self.graphs = {}
        self.prefill_graphs = {}
        self.graph_bs = []
        self.graph_vars = {}
        self.graph_outputs_are_logits = False
