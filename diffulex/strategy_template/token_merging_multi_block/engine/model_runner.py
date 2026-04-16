from __future__ import annotations

import torch

from diffulex.strategy_template.token_merging_multi_block.attention.metadata import (
    TokenMergingMultiBlockAttnMetaDataTemplate,
)
from diffulex.strategy_template.multi_block.engine.model_runner import MultiBlockModelRunnerTemplate
from diffulex.strategy_template.token_merging_multi_block.engine.request import TokenMergingMultiBlockReqTemplate


class TokenMergingMultiBlockModelRunnerTemplate(MultiBlockModelRunnerTemplate):
    token_merge_renormalize = True

    def prepare_chunked_prefill_token_merging_multi_block(
        self: TokenMergingMultiBlockModelRunnerTemplate,
        reqs: list[TokenMergingMultiBlockReqTemplate],
    ):
        input_ids, positions = self.prepare_chunked_prefill_multi_block(reqs)
        self._init_token_merge_metadata(reqs, positions)
        return input_ids, positions

    def _init_token_merge_metadata(
        self: TokenMergingMultiBlockModelRunnerTemplate,
        reqs: list[TokenMergingMultiBlockReqTemplate],
        positions: torch.Tensor,
    ) -> None:
        attn_metadata: TokenMergingMultiBlockAttnMetaDataTemplate = self.fetch_attn_metadata()

        num_tokens = int(positions.numel())
        if num_tokens == 0:
            attn_metadata.init_token_merging(mask_token_id=self.config.mask_token_id)
            return

        descriptors = []
        max_top_k = 1
        has_merge = False
        for req in reqs:
            for position in req.running_position_ids:
                descriptor = req.token_merge_descriptor_for_position(position)
                descriptors.append(descriptor)
                if descriptor is not None:
                    has_merge = True
                    max_top_k = max(max_top_k, len(descriptor.topk_ids))

        if len(descriptors) != num_tokens:
            raise RuntimeError(
                "Token-merge descriptor alignment failed: "
                f"descriptors={len(descriptors)}, num_tokens={num_tokens}"
            )

        if not has_merge:
            attn_metadata.init_token_merging(mask_token_id=self.config.mask_token_id)
            return

        device = positions.device
        merge_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        topk_ids = torch.full(
            (num_tokens, max_top_k),
            int(self.config.mask_token_id),
            dtype=torch.int64,
            device=device,
        )
        topk_probs = torch.zeros((num_tokens, max_top_k), dtype=torch.float32, device=device)
        residual_probs = torch.zeros((num_tokens, 1), dtype=torch.float32, device=device)

        for idx, descriptor in enumerate(descriptors):
            if descriptor is None:
                continue
            k = len(descriptor.topk_ids)
            merge_mask[idx] = True
            topk_ids[idx, :k] = torch.tensor(descriptor.topk_ids, dtype=torch.int64, device=device)
            topk_probs[idx, :k] = torch.tensor(descriptor.topk_probs, dtype=torch.float32, device=device)
            residual_probs[idx, 0] = float(descriptor.residual_prob)

        attn_metadata.init_token_merging(
            merge_mask=merge_mask,
            topk_ids=topk_ids,
            topk_probs=topk_probs,
            residual_probs=residual_probs,
            mask_token_id=self.config.mask_token_id,
            renormalize=bool(self.config.token_merge_renormalize),
            mode=self.config.token_merge_mode,
            weight=float(self.config.token_merge_weight),
        )

    def run_multi_block(
        self: TokenMergingMultiBlockModelRunnerTemplate,
        reqs: list[TokenMergingMultiBlockReqTemplate],
    ) -> list[int]:
        return self._run_token_merging_multi_block_subgroup(reqs)

    def _run_token_merging_multi_block_subgroup(
        self: TokenMergingMultiBlockModelRunnerTemplate,
        reqs: list[TokenMergingMultiBlockReqTemplate],
    ):
        input_ids, positions = self.prepare_chunked_prefill_token_merging_multi_block(reqs)
        temperatures = self.prepare_sample(reqs) if self.rank == 0 else None
        logits = self.run_model_multi_block(input_ids, positions)
        sample_output = self.sampler(reqs, logits, temperatures) if self.rank == 0 else None
        self.reset_attn_metadata()
        return sample_output
