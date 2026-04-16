from __future__ import annotations

import torch

from diffulex.engine.request import DllmReq
from diffulex.logger import get_logger

from .core import SamplerBase
from .no_shift import DllmSamplerNoShiftBase
from .output import SampleOutputBase

logger = get_logger(__name__)


class SamplerShiftLogits(SamplerBase):
    def __init__(self):
        super().__init__()
        self.req_last_logits_map: dict[str, torch.Tensor] = {}

    def _fetch_last_logits(self, logits: torch.Tensor, req: DllmReq) -> torch.Tensor:
        req_id_str = str(req.req_id)
        if req.has_to_cache_block:
            last_logits = logits[req.to_cache_last_token_id]
            self.req_last_logits_map[req_id_str] = last_logits
            return last_logits

        if req_id_str in self.req_last_logits_map:
            return self.req_last_logits_map[req_id_str]

        last_logits = logits[-1] if logits.shape[0] > 0 else None
        if last_logits is not None:
            self.req_last_logits_map[req_id_str] = last_logits
            return last_logits

        raise ValueError(f"Cannot fetch last logits for req {req.req_id}: empty logits tensor")

    def _shift_logits(self, logits, last_logit=None):
        if logits.shape[1] == 0:
            logger.warning("Logits sequence length is 0, returning empty logits")
            raise Exception("logits sequence length is 0")

        shifted_logits = torch.zeros_like(logits)
        shifted_logits[1:, ...] = logits[:-1, ...]

        if last_logit is not None:
            shifted_logits[0, ...] = last_logit
            return shifted_logits

        shifted_logits[0, ...] = 1.0
        return shifted_logits


class DllmSamplerShiftBase(SamplerShiftLogits):
    output_cls = SampleOutputBase

    def forward(
        self,
        reqs: list[DllmReq],
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
            true_local_ids_sub_map = {}
            accepted_ids_sub_map = {}
            sampled_tokens_sub_map = {}
            mask_token_rel_ids_sub_map = {}
            confidence_sub_map = {}
            initial_confidence_sub_map = {}
            if req_logits.shape[0] == 0:
                req_id_str = str(req.req_id)
                true_local_ids_map[req_id_str] = true_local_ids_sub_map
                accepted_ids_map[req_id_str] = accepted_ids_sub_map
                sampled_tokens_map[req_id_str] = sampled_tokens_sub_map
                mask_token_rel_ids_map[req_id_str] = mask_token_rel_ids_sub_map
                confidence_map[req_id_str] = confidence_sub_map
                initial_confidence_map[req_id_str] = initial_confidence_sub_map
                continue
            last_logits = self._fetch_last_logits(req_logits, req)
            shifted_logits = self._shift_logits(req_logits, last_logits)

            for block_id, block in enumerate(req.dllm_blocks):
                if not block.is_active or (block.num_mask_tokens == 0):
                    continue

                if len(block.mask_token_global_ids) == 0:
                    continue

                if attn_metadata.is_prefill[idx]:
                    if shifted_logits.shape[0] == 0:
                        continue
                    local_ids = DllmSamplerNoShiftBase._prefill_mask_token_local_ids(req, block, shifted_logits)
                    mask_token_logits = shifted_logits[local_ids, ...]
                else:
                    buf_offset = block.start - req.dllm_block_buffer.first_running_block.start
                    buf_ids = [buf_offset + i for i in block.mask_token_relative_ids]
                    mask_token_logits = shifted_logits[buf_ids, ...]

                confidence, sampled_tokens, initial_confidence = self.sample_tokens(
                    mask_token_logits,
                    temperature,
                    top_p=top_p,
                    top_k=top_k,
                    neg_entropy=(neg_entropy == "neg_entropy"),
                    margin_confidence=(margin_confidence == "margin_confidence"),
                    forbidden_token_ids=[int(block.mask_token_id)],
                )
                accepted_ids = self._compute_accepted_ids(
                    block, confidence, initial_confidence, sampled_tokens, **kwargs
                )
                block_id_str = str(block_id)
                accepted_ids_list = accepted_ids.to(device="cpu").tolist()
                true_local_ids_sub_map[block_id_str] = [block.mask_token_relative_ids[i] for i in accepted_ids_list]
                accepted_ids_sub_map[block_id_str] = accepted_ids_list
                sampled_tokens_sub_map[block_id_str] = sampled_tokens.to(device="cpu").tolist()
                mask_token_rel_ids_sub_map[block_id_str] = list(block.mask_token_relative_ids)
                confidence_sub_map[block_id_str] = confidence.to(device="cpu").tolist()
                initial_confidence_sub_map[block_id_str] = initial_confidence.to(device="cpu").tolist()

            req_id_str = str(req.req_id)
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
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError
