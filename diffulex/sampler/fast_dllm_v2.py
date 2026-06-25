import torch

from diffulex.sampler.auto_sampler import AutoSampler
from diffulex.sampler.base import DllmSamplerNoShiftBase
from diffulex.sampler.base import DllmSamplerShiftBase

_FDV2_SUB_BLOCK_REFINE = 1
_FDV2_FINAL_COMMIT = 2


@AutoSampler.register("fast_dllm_v2")
class FastdLLMV2Sampler(DllmSamplerShiftBase):
    def _shift_logits(self, logits, last_logit=None):
        del last_logit
        if logits.shape[0] == 0:
            return logits
        shifted_logits = torch.empty_like(logits)
        shifted_logits[0, ...] = logits[0, ...]
        shifted_logits[1:, ...] = logits[:-1, ...]
        return shifted_logits

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
        attn_metadata = self.fetch_attn_metadata()
        split_logits = DllmSamplerNoShiftBase._split_logits_per_req(attn_metadata, reqs, logits)

        accepted_ids_map = {}
        sampled_tokens_map = {}
        true_local_ids_map = {}
        mask_token_rel_ids_map = {}
        confidence_map = {}
        initial_confidence_map = {}

        for idx, (temperature, req, req_logits) in enumerate(zip(temperatures, reqs, split_logits)):
            temperature_value = float(temperature.item()) if torch.is_tensor(temperature) else float(temperature)
            req_id_str = str(req.req_id)
            true_local_ids_sub_map = {}
            accepted_ids_sub_map = {}
            sampled_tokens_sub_map = {}
            mask_token_rel_ids_sub_map = {}
            confidence_sub_map = {}
            initial_confidence_sub_map = {}

            if req_logits.shape[0] == 0:
                true_local_ids_map[req_id_str] = true_local_ids_sub_map
                accepted_ids_map[req_id_str] = accepted_ids_sub_map
                sampled_tokens_map[req_id_str] = sampled_tokens_sub_map
                mask_token_rel_ids_map[req_id_str] = mask_token_rel_ids_sub_map
                confidence_map[req_id_str] = confidence_sub_map
                initial_confidence_map[req_id_str] = initial_confidence_sub_map
                continue

            if int(getattr(req, "fdv2_mode", -1)) == _FDV2_FINAL_COMMIT:
                req.fdv2_pending_next_token_id = int(torch.argmax(req_logits[-1, ...]).item())
                true_local_ids_map[req_id_str] = true_local_ids_sub_map
                accepted_ids_map[req_id_str] = accepted_ids_sub_map
                sampled_tokens_map[req_id_str] = sampled_tokens_sub_map
                mask_token_rel_ids_map[req_id_str] = mask_token_rel_ids_sub_map
                confidence_map[req_id_str] = confidence_sub_map
                initial_confidence_map[req_id_str] = initial_confidence_sub_map
                continue

            is_fdv2_native_req = hasattr(req, "fdv2_current_sub_block") or int(
                getattr(req, "fdv2_mode", -1)
            ) in (_FDV2_SUB_BLOCK_REFINE, _FDV2_FINAL_COMMIT)
            if is_fdv2_native_req:
                shifted_logits = self._shift_logits(req_logits)
            else:
                last_logits = self._fetch_last_logits(req_logits, req)
                shifted_logits = DllmSamplerShiftBase._shift_logits(self, req_logits, last_logits)

            if attn_metadata.is_prefill[idx]:
                candidate_blocks = []
            elif not hasattr(req, "fdv2_current_sub_block"):
                candidate_blocks = [block for block in req.dllm_blocks if block.is_active]
            else:
                candidate_blocks = [req.fdv2_current_sub_block]

            for block in candidate_blocks:
                block_mask_relative_ids = (
                    req.fdv2_block_mask_token_relative_ids(block)
                    if hasattr(req, "fdv2_block_mask_token_relative_ids")
                    else list(block.mask_token_relative_ids)
                )
                if not block.is_active or not block_mask_relative_ids:
                    continue

                if attn_metadata.is_prefill[idx]:
                    local_ids = DllmSamplerNoShiftBase._prefill_mask_token_local_ids(req, block, shifted_logits)
                    mask_token_logits = shifted_logits[local_ids, ...]
                elif int(getattr(req, "fdv2_mode", -1)) == _FDV2_SUB_BLOCK_REFINE:
                    if bool(getattr(req, "fdv2_use_block_cache", True)):
                        buf_ids = list(block_mask_relative_ids)
                    else:
                        buf_offset = int(block.start - req.dllm_block_buffer.first_running_block.start)
                        buf_ids = [buf_offset + i for i in block_mask_relative_ids]
                    mask_token_logits = shifted_logits[buf_ids, ...]
                else:
                    buf_offset = int(block.start - req.dllm_block_buffer.first_running_block.start)
                    buf_ids = [buf_offset + i for i in block_mask_relative_ids]
                    mask_token_logits = shifted_logits[buf_ids, ...]

                confidence, sampled_tokens, initial_confidence = self.sample_tokens(
                    mask_token_logits,
                    temperature_value,
                    top_p=top_p,
                    top_k=top_k,
                    neg_entropy=(neg_entropy == "neg_entropy"),
                    margin_confidence=(margin_confidence == "margin_confidence"),
                    forbidden_token_ids=[int(block.mask_token_id)],
                )
                block_id_str = str(block.block_id)
                (
                    accepted_ids_list,
                    sampled_tokens_list,
                    confidence_list,
                    initial_confidence_list,
                ) = self._materialize_sampled_block(
                    block,
                    confidence,
                    sampled_tokens,
                    initial_confidence,
                    **kwargs,
                )
                true_local_ids_sub_map[block_id_str] = [block_mask_relative_ids[i] for i in accepted_ids_list]
                accepted_ids_sub_map[block_id_str] = accepted_ids_list
                sampled_tokens_sub_map[block_id_str] = sampled_tokens_list
                mask_token_rel_ids_sub_map[block_id_str] = list(block_mask_relative_ids)
                confidence_sub_map[block_id_str] = confidence_list
                initial_confidence_sub_map[block_id_str] = initial_confidence_list

            true_local_ids_map[req_id_str] = true_local_ids_sub_map
            accepted_ids_map[req_id_str] = accepted_ids_sub_map
            sampled_tokens_map[req_id_str] = sampled_tokens_sub_map
            mask_token_rel_ids_map[req_id_str] = mask_token_rel_ids_sub_map
            confidence_map[req_id_str] = confidence_sub_map
            initial_confidence_map[req_id_str] = initial_confidence_sub_map

        return self.output_cls(
            true_local_ids_map=true_local_ids_map,
            accepted_ids_map=accepted_ids_map,
            sampled_tokens_map=sampled_tokens_map,
            mask_token_rel_ids_map=mask_token_rel_ids_map,
            confidence_map=confidence_map,
            initial_confidence_map=initial_confidence_map,
        )

    def _compute_accepted_ids(
        self,
        block,
        confidence: torch.Tensor,
        initial_confidence: torch.Tensor,
        sampled_tokens: torch.Tensor,
        *,
        threshold: float = 0.95,
        **kwargs,
    ) -> torch.Tensor:
        accept_threshold = block.thresholds.accept_threshold
        high_conf_indices = torch.where(initial_confidence > accept_threshold)[0]
        topk_idx = (
            torch.topk(confidence, 1)[1]
            if len(high_conf_indices) == 0
            else torch.tensor([], device=confidence.device, dtype=torch.long)
        )

        req = getattr(block, "req", None)
        is_fdv2_native_step = getattr(req, "fdv2_current_sub_block", None) is block
        if is_fdv2_native_step:
            return torch.unique(torch.cat([topk_idx, high_conf_indices]))

        pre_block_complete = block.prev_block.is_semi_complete if block.prev_block else True
        if pre_block_complete:
            return torch.unique(torch.cat([topk_idx, high_conf_indices]))
        return high_conf_indices
