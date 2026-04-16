from __future__ import annotations

import torch

from diffulex.sampler.base import SampleOutputBase


class EditSamplerMixin:
    def _build_default_edit_writes_map(
        self,
        sample_output: SampleOutputBase,
    ) -> dict[str, dict[str, dict[int, int]]]:
        edit_writes_map: dict[str, dict[str, dict[int, int]]] = {}
        for req_id_str, accepted_blocks in sample_output.accepted_ids_map.items():
            req_edit_writes: dict[str, dict[int, int]] = {}
            sampled_blocks = sample_output.sampled_tokens_map[req_id_str]
            true_local_blocks = sample_output.true_local_ids_map[req_id_str]
            for block_id_str, accepted_ids in accepted_blocks.items():
                if not accepted_ids:
                    continue
                sampled_tokens = sampled_blocks[block_id_str]
                true_local_ids = true_local_blocks[block_id_str]
                block_writes: dict[int, int] = {}
                for true_local_id, accepted_id in zip(true_local_ids, accepted_ids):
                    if accepted_id >= len(sampled_tokens):
                        continue
                    block_writes[int(true_local_id)] = int(sampled_tokens[accepted_id])
                if block_writes:
                    req_edit_writes[block_id_str] = block_writes
            edit_writes_map[req_id_str] = req_edit_writes
        return edit_writes_map

    def _build_edit_writes_map(
        self,
        reqs,
        split_logits: tuple[torch.Tensor, ...],
        temperatures: torch.Tensor,
        sample_output: SampleOutputBase,
        attn_metadata,
        **kwargs,
    ) -> dict[str, dict[str, dict[int, int]]]:
        del reqs, split_logits, temperatures, attn_metadata, kwargs
        return self._build_default_edit_writes_map(sample_output)

    def _postprocess_sample_output(
        self,
        reqs,
        split_logits: tuple[torch.Tensor, ...],
        temperatures: torch.Tensor,
        sample_output: SampleOutputBase,
        attn_metadata,
        **kwargs,
    ) -> SampleOutputBase:
        sample_output = super()._postprocess_sample_output(
            reqs=reqs,
            split_logits=split_logits,
            temperatures=temperatures,
            sample_output=sample_output,
            attn_metadata=attn_metadata,
            **kwargs,
        )
        sample_output.edit_writes_map = self._build_edit_writes_map(
            reqs=reqs,
            split_logits=split_logits,
            temperatures=temperatures,
            sample_output=sample_output,
            attn_metadata=attn_metadata,
            **kwargs,
        )
        return sample_output
