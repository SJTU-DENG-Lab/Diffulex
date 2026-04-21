from __future__ import annotations

import torch

from diffulex.engine.request import DllmReq

from .core import SamplerBase
from .output import SampleOutputBase


class SamplerNoShiftLogits(SamplerBase):
    pass


class DllmSamplerNoShiftBase(SamplerNoShiftLogits):
    output_cls = SampleOutputBase

    def _maybe_log_prefill_alignment(
        self,
        req: DllmReq,
        req_logits: torch.Tensor,
        block_summaries: list[dict],
    ) -> None:
        logged = getattr(self, "_debug_prefill_alignment_logged", None)
        if logged is None:
            logged = set()
            self._debug_prefill_alignment_logged = logged
        req_id = int(getattr(req, "req_id", -1))
        if req_id in logged:
            return
        logged.add(req_id)

    @staticmethod
    def _split_logits_per_req(attn_metadata, reqs: list[DllmReq], logits: torch.Tensor) -> tuple[torch.Tensor, ...]:
        cu = attn_metadata.cu_seqlens_q
        if cu is not None and len(cu) == len(reqs) + 1:
            split_sizes = [(int(cu[i + 1]) - int(cu[i])) for i in range(len(reqs))]
        else:
            split_sizes = [
                len(req.running_sequence) if attn_metadata.is_prefill[idx] else req.chunk_size
                for idx, req in enumerate(reqs)
            ]
        return torch.split(logits, split_sizes, dim=0)

    @staticmethod
    def _prefill_mask_token_local_ids(req: DllmReq, block, req_logits: torch.Tensor) -> list[int]:
        # Use contiguous cached prefix length for prefill-logits alignment.
        # `in_cache_len` may include non-prefix cached blocks and can overshoot.
        prefix_offset = int(req.contiguous_in_cache_prefix_len)
        local_ids = [idx - prefix_offset for idx in block.mask_token_global_ids]
        if not local_ids:
            return local_ids

        if min(local_ids) < 0 or max(local_ids) >= req_logits.shape[0]:
            # Mixed prefill batches can yield partial q_len for a req in one step.
            # Skip this block this step and retry when its logits slice is present.
            return []
        return local_ids

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

        split_logits = self._split_logits_per_req(attn_metadata, reqs, logits)

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
            prefill_block_summaries: list[dict] = []

            for block_id, block in enumerate(req.dllm_blocks):
                if not block.is_active or (block.num_mask_tokens == 0):
                    continue

                if len(block.mask_token_global_ids) == 0:
                    continue

                if attn_metadata.is_prefill[idx]:
                    if getattr(req, "_resume_prefill_until", 0) > 0 and getattr(block, "start", 0) >= req.running_len:
                        continue
                    # Prefix-cache prefill can produce q_len=0 for some requests in mixed batches.
                    # In that case there are no logits to sample for this req in this step.
                    if req_logits.shape[0] == 0:
                        continue
                    local_ids = self._prefill_mask_token_local_ids(req, block, req_logits)
                    if not local_ids:
                        continue
                    prefill_block_summaries.append(
                        {
                            "block_id": int(block_id),
                            "start": int(getattr(block, "start", -1)),
                            "end": int(getattr(block, "end", -1)),
                            "num_mask_tokens": int(getattr(block, "num_mask_tokens", 0)),
                            "mask_token_global_ids_head": [int(x) for x in block.mask_token_global_ids[:8]],
                            "local_ids_head": [int(x) for x in local_ids[:8]],
                            "local_ids_min": int(min(local_ids)) if local_ids else None,
                            "local_ids_max": int(max(local_ids)) if local_ids else None,
                        }
                    )
                    mask_token_logits = req_logits[local_ids, ...]
                else:
                    buf_offset = block.start - req.dllm_block_buffer.first_running_block.start
                    buf_ids = [buf_offset + i for i in block.mask_token_relative_ids]
                    mask_token_logits = req_logits[buf_ids, ...]

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
            if attn_metadata.is_prefill[idx]:
                self._maybe_log_prefill_alignment(req, req_logits, prefill_block_summaries[:4])
            true_local_ids_map[req_id_str] = true_local_ids_sub_map
            accepted_ids_map[req_id_str] = accepted_ids_sub_map
            sampled_tokens_map[req_id_str] = sampled_tokens_sub_map
            mask_token_rel_ids_map[req_id_str] = mask_token_rel_ids_sub_map
            confidence_map[req_id_str] = confidence_sub_map
            initial_confidence_map[req_id_str] = initial_confidence_sub_map

        sample_output = self.output_cls(
            true_local_ids_map=true_local_ids_map,
            accepted_ids_map=accepted_ids_map,
            sampled_tokens_map=sampled_tokens_map,
            mask_token_rel_ids_map=mask_token_rel_ids_map,
            confidence_map=confidence_map,
            initial_confidence_map=initial_confidence_map,
        )
        return self._postprocess_sample_output(
            reqs=reqs,
            split_logits=split_logits,
            temperatures=temperatures,
            sample_output=sample_output,
            attn_metadata=attn_metadata,
            **kwargs,
        )

    def _postprocess_sample_output(
        self,
        reqs: list[DllmReq],
        split_logits: tuple[torch.Tensor, ...],
        temperatures: torch.Tensor,
        sample_output: SampleOutputBase,
        attn_metadata,
        **kwargs,
    ) -> SampleOutputBase:
        del reqs, split_logits, temperatures, attn_metadata, kwargs
        return sample_output

    def _compute_accepted_ids(
        self,
        block,
        confidence: torch.Tensor,
        initial_confidence: torch.Tensor,
        sampled_tokens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError
