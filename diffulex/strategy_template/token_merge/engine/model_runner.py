from __future__ import annotations

import torch

from diffulex.strategy_template.token_merge.attention.metadata import (
    TokenMergeAttnMetaDataTemplate,
)
from diffulex.strategy_template.multi_block.engine.model_runner import MultiBlockModelRunnerTemplate
from diffulex.strategy_template.token_merge.engine.cudagraph import TokenMergeCudaGraphMixin
from diffulex.strategy_template.token_merge.engine.request import TokenMergeReqTemplate


class TokenMergeModelRunnerTemplate(TokenMergeCudaGraphMixin, MultiBlockModelRunnerTemplate):
    token_merge_renormalize = True

    def _ensure_runtime_token_merge_buffers(
        self,
        *,
        token_capacity: int,
        top_k_capacity: int,
    ) -> dict[str, torch.Tensor]:
        device = self.cuda_graph_device
        token_capacity = max(1, int(token_capacity), int(self.config.max_num_batched_tokens))
        top_k_capacity = max(1, int(top_k_capacity), int(self.config.token_merge_top_k))
        buffers = getattr(self, "_runtime_token_merge_buffers", None)
        if buffers is not None:
            if (
                int(buffers["merge_mask"].numel()) >= token_capacity
                and int(buffers["topk_ids"].size(1)) >= top_k_capacity
            ):
                return buffers

        buffers = {
            "merge_mask": torch.empty(token_capacity, dtype=torch.bool, device=device),
            "topk_ids": torch.empty(token_capacity, top_k_capacity, dtype=torch.int64, device=device),
            "topk_probs": torch.empty(token_capacity, top_k_capacity, dtype=torch.float32, device=device),
            "residual_probs": torch.empty(token_capacity, 1, dtype=torch.float32, device=device),
        }
        self._runtime_token_merge_buffers = buffers
        return buffers

    def prepare_chunked_prefill_multi_block(
        self: TokenMergeModelRunnerTemplate,
        reqs: list[TokenMergeReqTemplate],
    ):
        input_ids, positions = super().prepare_chunked_prefill_multi_block(reqs)
        self._init_token_merge_metadata(reqs, positions)
        return input_ids, positions

    def prepare_chunked_prefill_token_merge(
        self: TokenMergeModelRunnerTemplate,
        reqs: list[TokenMergeReqTemplate],
    ):
        return self.prepare_chunked_prefill_multi_block(reqs)

    def _init_token_merge_metadata(
        self: TokenMergeModelRunnerTemplate,
        reqs: list[TokenMergeReqTemplate],
        positions: torch.Tensor,
    ) -> None:
        attn_metadata: TokenMergeAttnMetaDataTemplate = self.fetch_attn_metadata()

        num_tokens = int(positions.numel())
        if num_tokens == 0:
            attn_metadata.init_token_merge(mask_token_id=self.config.mask_token_id)
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
            attn_metadata.init_token_merge(mask_token_id=self.config.mask_token_id)
            return

        buffers = self._ensure_runtime_token_merge_buffers(token_capacity=num_tokens, top_k_capacity=max_top_k)
        merge_mask = buffers["merge_mask"][:num_tokens]
        topk_ids = buffers["topk_ids"][:num_tokens, :max_top_k]
        topk_probs = buffers["topk_probs"][:num_tokens, :max_top_k]
        residual_probs = buffers["residual_probs"][:num_tokens]

        mask_token_id = int(self.config.mask_token_id)
        merge_mask_rows = [False] * num_tokens
        topk_id_rows = [[mask_token_id] * max_top_k for _ in range(num_tokens)]
        topk_prob_rows = [[0.0] * max_top_k for _ in range(num_tokens)]
        residual_prob_rows = [[0.0] for _ in range(num_tokens)]

        for idx, descriptor in enumerate(descriptors):
            if descriptor is None:
                continue
            k = len(descriptor.topk_ids)
            merge_mask_rows[idx] = True
            topk_id_rows[idx][:k] = descriptor.topk_ids
            topk_prob_rows[idx][:k] = descriptor.topk_probs
            residual_prob_rows[idx][0] = float(descriptor.residual_prob)

        merge_mask.copy_(self._cpu_tensor(merge_mask_rows, torch.bool), non_blocking=True)
        topk_ids.copy_(self._cpu_tensor(topk_id_rows, torch.int64), non_blocking=True)
        topk_probs.copy_(self._cpu_tensor(topk_prob_rows, torch.float32), non_blocking=True)
        residual_probs.copy_(self._cpu_tensor(residual_prob_rows, torch.float32), non_blocking=True)

        attn_metadata.init_token_merge(
            merge_mask=merge_mask,
            topk_ids=topk_ids,
            topk_probs=topk_probs,
            residual_probs=residual_probs,
            mask_token_id=self.config.mask_token_id,
            renormalize=bool(self.config.token_merge_renormalize),
            mode=self.config.token_merge_mode,
            weight=float(self.config.token_merge_weight),
        )
