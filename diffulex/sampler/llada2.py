from __future__ import annotations

import torch
import torch.nn.functional as F

from diffulex.mixin import EditSamplerMixin, TokenMergeSamplerMixin
from diffulex.sampler.auto_sampler import AutoSampler
from diffulex.sampler.base import DllmSamplerNoShiftBase


class LLaDA2AcceptedIdsMixin:
    def _compute_accepted_ids(
        self,
        block,
        confidence: torch.Tensor,
        initial_confidence: torch.Tensor,
        sampled_tokens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        accept_threshold = block.thresholds.accept_threshold
        pre_block_complete = block.prev_block.is_semi_complete if block.prev_block else True

        high_conf_indices = torch.where(initial_confidence > accept_threshold)[0]
        if pre_block_complete:
            if len(high_conf_indices) == 0:
                _, transfer_index = torch.topk(confidence, 1)
                return transfer_index
            transfer_index = torch.tensor([], device=sampled_tokens.device, dtype=torch.long)
            return torch.unique(torch.cat([transfer_index, high_conf_indices]))
        return high_conf_indices


class LLaDA2Sampler(LLaDA2AcceptedIdsMixin, DllmSamplerNoShiftBase):
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

    @staticmethod
    def _sample_argmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature > 0:
            gumbel = -torch.log(-torch.log(torch.rand_like(logits, dtype=torch.float32).clamp_(1e-6, 1 - 1e-6)))
            logits = logits.to(torch.float32) + gumbel * temperature
        return torch.argmax(logits, dim=-1)

    @staticmethod
    def _segment_max_transfer(mask_new: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        transfer_index = torch.zeros_like(mask_new, dtype=torch.bool)
        starts = None
        for idx, is_mask in enumerate(mask_new.tolist()):
            if is_mask and starts is None:
                starts = idx
            elif not is_mask and starts is not None:
                transfer_index[starts + int(torch.argmax(confidence[starts:idx]).item())] = True
                starts = None
        if starts is not None:
            transfer_index[starts + int(torch.argmax(confidence[starts:]).item())] = True
        return transfer_index

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
        block,
        block_tokens: torch.Tensor,
        block_logits: torch.Tensor,
        temperature: float,
    ) -> tuple[dict[int, int], dict[int, dict | None]]:
        mask_id = int(block.mask_token_id)
        accept_threshold = float(block.thresholds.accept_threshold)
        remask_threshold = float(block.thresholds.remask_threshold)
        mask_index = block_tokens.eq(mask_id)
        x0 = self._sample_argmax(block_logits, temperature)
        probs = F.softmax(block_logits.to(torch.float32), dim=-1)
        x0_p = probs.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

        lower_index = x0_p < remask_threshold
        remask_index = torch.logical_and(lower_index, torch.logical_not(mask_index))
        mask_new = torch.logical_or(lower_index, mask_index)
        if not bool(mask_new.any().item()):
            return {}, {}

        neg_inf = torch.full_like(x0_p, torch.finfo(x0_p.dtype).min)
        confidence = torch.where(mask_new, x0_p, neg_inf)
        transfer_index = self._segment_max_transfer(mask_new, confidence)

        transfer_index = torch.logical_and(transfer_index, confidence > accept_threshold)
        transfer_index = torch.logical_or(transfer_index, confidence > accept_threshold)

        gap = int(remask_index.sum().item() + 1 - transfer_index.sum().item())
        if gap > 0:
            candidate_conf = confidence.clone()
            candidate_conf[transfer_index] = torch.finfo(candidate_conf.dtype).min
            extra_k = min(gap, int(mask_new.sum().item()))
            if extra_k > 0:
                _, indices = torch.topk(candidate_conf, extra_k, largest=True, sorted=False)
                transfer_index[indices] = True

        remask_index = torch.logical_and(remask_index, torch.logical_not(transfer_index))
        x0[remask_index] = mask_id
        transfer_index[remask_index] = True

        block_writes: dict[int, int] = {}
        token_merge_entries: dict[int, dict | None] = {}
        for rel_idx in torch.where(transfer_index)[0].tolist():
            token = int(x0[rel_idx].item())
            if token != int(block_tokens[rel_idx].item()):
                block_writes[int(rel_idx)] = token
            token_merge_entries[int(rel_idx)] = self._build_token_merge_descriptor(
                probs=probs[rel_idx],
                token=token,
                mask_id=mask_id,
            )

        return block_writes, token_merge_entries

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
        for req_idx, (req, req_logits) in enumerate(zip(reqs, split_logits)):
            req_id_str = str(req.req_id)
            req_edit_writes: dict[str, dict[int, int]] = {}
            req_token_merge: dict[int, dict | None] = {}
            for block_id, block in enumerate(req.dllm_blocks):
                if not block.is_active:
                    continue
                block_logits = self._extract_block_logits(req, req_logits, block, attn_metadata.is_prefill[req_idx])
                if block_logits is None or block_logits.shape[0] != int(block.block_size):
                    continue
                for rel_idx in range(int(block.block_size)):
                    req_token_merge.setdefault(int(block.start + rel_idx), None)
                block_tokens = torch.tensor(block.token_ids, dtype=torch.long, device=block_logits.device)
                block_writes, token_merge_entries = self._build_dmax_block_outputs(
                    block=block,
                    block_tokens=block_tokens,
                    block_logits=block_logits,
                    temperature=float(temperatures[req_idx].item()),
                )
                if block_writes:
                    req_edit_writes[str(block_id)] = block_writes
                for rel_idx, descriptor in token_merge_entries.items():
                    req_token_merge[int(block.start + rel_idx)] = descriptor
            edit_writes_map[req_id_str] = req_edit_writes
            self._set_token_merge_entries(req_id_str, req_token_merge)
        return edit_writes_map


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
