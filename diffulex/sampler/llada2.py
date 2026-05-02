from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

from diffulex.mixin.edit import EditSamplerMixin
from diffulex.mixin.token_merge import TokenMergeSamplerMixin
from diffulex.sampler.auto_sampler import AutoSampler
from diffulex.sampler.base import DllmSamplerNoShiftBase, SemiCompleteAcceptedIdsMixin

if TYPE_CHECKING:
    from diffulex.engine.dllm_block import DllmBlock


class LLaDA2Sampler(SemiCompleteAcceptedIdsMixin, DllmSamplerNoShiftBase):
    def __init__(self, config=None):
        del config
        super().__init__()


class LLaDA2dot1Sampler(EditSamplerMixin, LLaDA2Sampler):
    def __init__(self, config=None):
        del config
        super().__init__()


class LLaDA2DMaxSampler(TokenMergeSamplerMixin, LLaDA2dot1Sampler):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.token_merge_mode = str(getattr(config, "token_merge_mode", "dmax_topk"))
        self.enable_token_merge = bool(
            self.token_merge_mode in {"dmax_topk", "iter_smooth_topk"}
            and float(getattr(config, "token_merge_weight", 1.0)) > 0.0
        )
        self.last_block_state_map: dict[str, dict[str, dict]] = {}
        del config

    def _compute_accepted_ids(
        self,
        block,
        confidence: torch.Tensor,
        initial_confidence: torch.Tensor,
        sampled_tokens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del block, confidence, initial_confidence, kwargs
        return torch.empty(0, dtype=torch.long, device=sampled_tokens.device)

    def forward(
        self,
        reqs,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_p=None,
        top_k=None,
        margin_confidence=False,
        neg_entropy=False,
        **kwargs,
    ):
        # DMax derives writes/token-merge state from full block logits in
        # _build_edit_writes_map. Running the generic mask-token sampler first
        # duplicates argmax/softmax work and its accepted-id output is unused.
        del top_p, top_k, margin_confidence, neg_entropy
        attn_metadata = self.fetch_attn_metadata()
        split_logits = self._split_logits_per_req(attn_metadata, reqs, logits)

        empty_per_req = {str(req.req_id): {} for req in reqs}
        sample_output = self.output_cls(
            true_local_ids_map=dict(empty_per_req),
            accepted_ids_map=dict(empty_per_req),
            sampled_tokens_map=dict(empty_per_req),
            mask_token_rel_ids_map=dict(empty_per_req),
            confidence_map=dict(empty_per_req),
            initial_confidence_map=dict(empty_per_req),
        )
        return self._postprocess_sample_output(
            reqs=reqs,
            split_logits=split_logits,
            temperatures=temperatures,
            sample_output=sample_output,
            attn_metadata=attn_metadata,
            **kwargs,
        )

    @staticmethod
    def _sample_argmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature > 0:
            gumbel = -torch.log(-torch.log(torch.rand_like(logits, dtype=torch.float32).clamp_(1e-6, 1 - 1e-6)))
            logits = logits.to(torch.float32) + gumbel * temperature
        return torch.argmax(logits, dim=-1)

    def _extract_block_logits(self, req, req_logits: torch.Tensor, block, is_prefill: bool) -> torch.Tensor | None:
        if req_logits.shape[0] == 0:
            return None
        if is_prefill:
            prefix_offset = int(req.contiguous_in_cache_prefix_len)
            local_start = int(block.start - prefix_offset)
            local_end = int(block.end - prefix_offset)
            if local_start < 0 or local_end > req_logits.shape[0]:
                return None
            return req_logits[local_start:local_end, ...]

        buf_offset = int(block.start - req.dllm_block_buffer.first_running_block.start)
        local_start = buf_offset
        local_end = buf_offset + int(block.block_size)
        if local_start < 0 or local_end > req_logits.shape[0]:
            return None
        return req_logits[local_start:local_end, ...]

    def _build_dmax_block_outputs(
        self,
        block: DllmBlock,
        block_tokens: torch.Tensor,
        block_logits: torch.Tensor,
        temperature: float,
    ) -> tuple[dict[int, int], dict[int, dict | None], dict]:
        editable_start = int(getattr(block, "editable_start", 0) or 0)
        if editable_start >= int(block.block_size):
            return {}, {}, {
                "token_ids": block_tokens.tolist(),
                "confidences": [1.0] * int(block.block_size),
            }

        mask_id = int(block.mask_token_id)
        accept_threshold = float(block.thresholds.accept_threshold)
        full_block_before = block_tokens.clone()
        top1_tokens = self._sample_argmax(block_logits, temperature)
        mask_index = full_block_before == mask_id
        mask_positions = torch.nonzero(mask_index, as_tuple=False).flatten()

        top1_confidence = torch.ones(top1_tokens.shape, dtype=torch.float32, device=top1_tokens.device)
        if mask_positions.numel() > 0:
            mask_logits_fp32 = block_logits.index_select(0, mask_positions).to(torch.float32)
            mask_top1 = top1_tokens.index_select(0, mask_positions).unsqueeze(-1)
            mask_top1_logits = mask_logits_fp32.gather(dim=-1, index=mask_top1).squeeze(-1)
            mask_lse = torch.logsumexp(mask_logits_fp32, dim=-1)
            top1_confidence[mask_positions] = torch.exp(mask_top1_logits - mask_lse)

        target_tokens = full_block_before.clone()
        token_index = full_block_before != mask_id
        if bool(token_index.any().item()):
            target_tokens[token_index] = top1_tokens[token_index]

        decode_positions = torch.empty(0, dtype=torch.long, device=full_block_before.device)
        below_threshold_positions = torch.empty(0, dtype=torch.long, device=full_block_before.device)
        if bool(mask_index.any().item()):
            mask_confidence = top1_confidence[mask_positions]
            below_threshold = torch.nonzero(mask_confidence < accept_threshold, as_tuple=False).flatten()
            if below_threshold.numel() > 0:
                below_threshold_positions = mask_positions[below_threshold]
            if below_threshold.numel() == 0:
                decode_upto = int(mask_positions.numel())
            elif int(below_threshold[0].item()) == 0:
                decode_upto = 1
            else:
                decode_upto = int(below_threshold[0].item())
            decode_positions = mask_positions[:decode_upto]
            if decode_positions.numel() > 0:
                target_tokens[decode_positions] = top1_tokens[decode_positions]

        block_writes: dict[int, int] = {}
        token_merge_entries: dict[int, dict | None] = {}
        changed_positions = torch.nonzero(target_tokens != full_block_before, as_tuple=False).flatten()
        for rel_idx in changed_positions.tolist():
            if int(rel_idx) < editable_start:
                continue
            token = int(target_tokens[rel_idx].item())
            if token != int(full_block_before[rel_idx].item()):
                block_writes[int(rel_idx)] = token

        if self.enable_token_merge:
            non_mask_positions = torch.nonzero(target_tokens != mask_id, as_tuple=False).flatten()
            for rel_idx in non_mask_positions.tolist():
                if int(rel_idx) < editable_start:
                    continue
                token = int(target_tokens[rel_idx].item())
                if self.token_merge_mode == "dmax_topk":
                    descriptor = self._build_manual_token_merge_descriptor(
                        token=token,
                        confidence=float(top1_confidence[rel_idx].item()),
                        mask_id=mask_id,
                    )
                else:
                    row_probs = F.softmax(block_logits[rel_idx].to(torch.float32), dim=-1)
                    descriptor = self._build_token_merge_descriptor(
                        probs=row_probs,
                        token=token,
                        mask_id=mask_id,
                    )
                token_merge_entries[int(rel_idx)] = descriptor

        return block_writes, token_merge_entries, {
            "token_ids": target_tokens.tolist(),
            "confidences": top1_confidence.tolist(),
        }

    def _reset_block_state_map(self) -> None:
        self.last_block_state_map = {}

    def _build_edit_writes_map(
        self,
        reqs,
        split_logits,
        temperatures: torch.Tensor,
        sample_output,
        attn_metadata,
        **kwargs,
    ) -> dict[str, dict[str, dict[int, int]]]:
        del sample_output, kwargs
        edit_writes_map: dict[str, dict[str, dict[int, int]]] = {}
        self._reset_token_merge_map()
        self._reset_block_state_map()
        for req_idx, (req, req_logits) in enumerate(zip(reqs, split_logits)):
            req_id_str = str(req.req_id)
            req_edit_writes: dict[str, dict[int, int]] = {}
            req_token_merge: dict[int, dict | None] = {}
            req_block_states: dict[str, dict] = {}
            for block_id, block in enumerate(req.dllm_blocks):
                if not block.is_active:
                    continue
                block_logits = self._extract_block_logits(req, req_logits, block, attn_metadata.is_prefill[req_idx])
                if block_logits is None or block_logits.shape[0] != int(block.block_size):
                    continue
                for rel_idx in range(int(block.block_size)):
                    req_token_merge.setdefault(int(block.start + rel_idx), None)
                block_tokens = torch.tensor(block.token_ids, dtype=torch.long, device=block_logits.device)
                block_writes, token_merge_entries, block_state = self._build_dmax_block_outputs(
                    block=block,
                    block_tokens=block_tokens,
                    block_logits=block_logits,
                    temperature=float(temperatures[req_idx].item()),
                )
                req_block_states[str(block_id)] = block_state
                if block_writes:
                    req_edit_writes[str(block_id)] = block_writes
                for rel_idx, descriptor in token_merge_entries.items():
                    req_token_merge[int(block.start + rel_idx)] = descriptor
            edit_writes_map[req_id_str] = req_edit_writes
            self._set_token_merge_entries(req_id_str, req_token_merge)
            self.last_block_state_map[req_id_str] = req_block_states
        return edit_writes_map

    def _postprocess_sample_output(
        self,
        reqs,
        split_logits,
        temperatures: torch.Tensor,
        sample_output,
        attn_metadata,
        **kwargs,
    ):
        sample_output = super()._postprocess_sample_output(
            reqs=reqs,
            split_logits=split_logits,
            temperatures=temperatures,
            sample_output=sample_output,
            attn_metadata=attn_metadata,
            **kwargs,
        )
        sample_output.block_state_map = self.last_block_state_map
        return sample_output


def build_llada2_sampler(config=None):
    sampling_mode = str(getattr(config, "sampling_mode", "naive"))
    if getattr(config, "decoding_strategy", None) == "dmax":
        return LLaDA2DMaxSampler(config)
    if sampling_mode == "edit" and getattr(config, "model_name", None) in [
        "llada2",
        "llada2_moe",
        "llada2_mini",
        "llada2dot1_mini",
    ]:
        return LLaDA2dot1Sampler(config)
    if sampling_mode == "naive" and getattr(config, "model_name", None) in ["llada2", "llada2_moe", "llada2_mini"]:
        return LLaDA2Sampler(config)


AutoSampler.register("llada2", build_llada2_sampler, use_full_config=True)
AutoSampler.register("llada2_moe", build_llada2_sampler, use_full_config=True)
AutoSampler.register("llada2_mini", build_llada2_sampler, use_full_config=True)
AutoSampler.register("llada2dot1_mini", build_llada2_sampler, use_full_config=True)
AutoSampler.register("llada2_mini_dmax", build_llada2_sampler, use_full_config=True)
