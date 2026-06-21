from __future__ import annotations

from multiprocessing.synchronize import Event

import torch

from diffulex.attention.metadata import AttnMetaDataBase, set_fetch_fn_for_attn_metadata
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

    def _prepare_decode_req(self, req: DllmReq):
        prepared = super()._prepare_decode_req(req)
        # DiffusionGemma pads the prompt to the 256-token canvas/page size but
        # only writes real prompt tokens during prefill. Decode attention must
        # keep the true prefix bounds so the Triton kernel masks those unwritten
        # padding slots instead of reading stale KV from a reused page.
        prepared["prefix_len"] = int(req.prefix_len)
        prepared["padded_prefix_len"] = int(req.padded_prefix_len)
        return prepared

    @torch.inference_mode()
    def run_model_multi_block(self, input_ids: torch.Tensor, positions: torch.Tensor):
        with record_function("diffulex.diffusion_gemma.model_forward"):
            attn_metadata: AttnMetaDataBase = self.fetch_attn_metadata()
            full_runner = self._full_static_runner()

            if attn_metadata.has_prefill:
                with record_function("diffulex.diffusion_gemma.eager_prefill"):
                    return self.model.compute_logits(self.model(input_ids, positions))

            if not full_runner.can_run_decode(input_ids):
                with record_function("diffulex.diffusion_gemma.eager_decode"):
                    return self.model.compute_logits(self.model(input_ids, positions))

            with record_function("diffulex.diffusion_gemma.full_static_decode"):
                return full_runner.run_decode(input_ids, positions, attn_metadata)

    def _before_multi_block_model_forward(self, local_reqs: list[DllmReq]) -> None:
        set_context = getattr(self.model, "set_self_conditioning_context", None)
        if set_context is None:
            self._pending_self_conditioning_context = []
            return

        get_embeds = getattr(self.sampler, "get_self_conditioning_embeds", None)
        if get_embeds is None:
            self._pending_self_conditioning_context = []
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

        self._pending_self_conditioning_context = context
        set_context(context or None)

    def _init_graph_capture_extra_metadata(
        self,
        attn_metadata: AttnMetaDataBase,
        graph_vars: dict[str, torch.Tensor],
        num_tokens: int,
    ) -> None:
        del attn_metadata
        if "self_conditioning_soft_embeds" not in graph_vars:
            hf_config = getattr(self.config, "hf_config")
            text_config = getattr(hf_config, "text_config", hf_config)
            hidden_size = int(text_config.hidden_size)
            device = self._cuda_graph_device()
            graph_vars["self_conditioning_soft_embeds"] = torch.zeros(
                int(graph_vars["input_ids"].numel()),
                hidden_size,
                dtype=torch.float32,
                device=device,
            )
            graph_vars["self_conditioning_mask"] = torch.zeros(
                int(graph_vars["input_ids"].numel()),
                dtype=torch.bool,
                device=device,
            )

        soft_embeds = graph_vars["self_conditioning_soft_embeds"][:num_tokens]
        mask = graph_vars["self_conditioning_mask"][:num_tokens]
        soft_embeds.zero_()
        mask.zero_()
        set_tensor_context = getattr(self.model, "set_self_conditioning_tensor_context", None)
        if set_tensor_context is not None:
            set_tensor_context(soft_embeds, mask, active=True)

    def _bind_decode_graph_extra_metadata(
        self,
        attn_metadata: AttnMetaDataBase,
        graph_vars: dict[str, torch.Tensor],
        num_tokens: int,
    ) -> None:
        del attn_metadata
        soft_buffer = graph_vars.get("self_conditioning_soft_embeds")
        mask_buffer = graph_vars.get("self_conditioning_mask")
        if soft_buffer is None or mask_buffer is None:
            return

        soft_buffer = soft_buffer[:num_tokens]
        mask_buffer = mask_buffer[:num_tokens]
        soft_buffer.zero_()
        mask_buffer.zero_()

        context = getattr(self, "_pending_self_conditioning_context", None) or []
        for item in context:
            start = max(0, int(item["start"]))
            end = min(num_tokens, int(item["end"]))
            if end <= start:
                continue
            soft_embeds = item.get("soft_embeds")
            if soft_embeds is None:
                continue
            length = min(end - start, int(soft_embeds.shape[0]))
            if length <= 0:
                continue
            soft_buffer[start : start + length].copy_(
                soft_embeds[:length].to(device=soft_buffer.device, dtype=soft_buffer.dtype)
            )
            mask_buffer[start : start + length] = True

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
        try:
            return super().capture_cudagraph_multi_block()
        finally:
            disable_tensor_context = getattr(self.model, "disable_self_conditioning_tensor_context", None)
            if disable_tensor_context is not None:
                disable_tensor_context()
