from __future__ import annotations

import torch

from diffulex.strategy_template.token_merge.attention.metadata import (
    TokenMergeAttnMetaDataTemplate,
)


class TokenMergeCudaGraphMixin:
    @property
    def graph_token_merge_top_k(self) -> int:
        return max(1, int(self.config.token_merge_top_k))

    def _can_use_decode_graph_extra(
        self,
        attn_metadata: TokenMergeAttnMetaDataTemplate,
        num_tokens: int,
        *,
        captured_num_tokens: int | None = None,
    ) -> bool:
        return self._token_merge_graph_capacity_ok(attn_metadata)

    def _token_merge_graph_capacity_ok(
        self,
        attn_metadata: TokenMergeAttnMetaDataTemplate,
    ) -> bool:
        if not bool(attn_metadata.token_merge_enabled):
            return True
        topk_ids = attn_metadata.token_merge_topk_ids
        if topk_ids is None:
            return True
        return int(topk_ids.shape[1]) <= self.graph_token_merge_top_k

    def _token_merge_graph_vars(self, token_capacity: int, device: torch.device) -> dict[str, torch.Tensor]:
        token_merge_top_k = self.graph_token_merge_top_k
        return {
            "token_merge_mask": torch.zeros(token_capacity, dtype=torch.bool, device=device),
            "token_merge_topk_ids": torch.full(
                (token_capacity, token_merge_top_k),
                int(self.config.mask_token_id),
                dtype=torch.int64,
                device=device,
            ),
            "token_merge_topk_probs": torch.zeros(
                (token_capacity, token_merge_top_k),
                dtype=torch.float32,
                device=device,
            ),
            "token_merge_residual_probs": torch.zeros((token_capacity, 1), dtype=torch.float32, device=device),
        }

    def _init_graph_capture_token_merge_metadata(
        self,
        attn_metadata: TokenMergeAttnMetaDataTemplate,
        graph_vars: dict[str, torch.Tensor],
        num_tokens: int,
    ) -> None:
        merge_mask = graph_vars["token_merge_mask"][:num_tokens]
        topk_ids = graph_vars["token_merge_topk_ids"][:num_tokens]
        topk_probs = graph_vars["token_merge_topk_probs"][:num_tokens]
        residual_probs = graph_vars["token_merge_residual_probs"][:num_tokens]

        merge_mask.zero_()
        topk_ids.fill_(int(self.config.mask_token_id))
        topk_probs.zero_()
        residual_probs.zero_()
        if num_tokens > 0:
            # Capture the token-merge branch once; replay-time merge_mask controls no-op behavior.
            merge_mask[0] = True
            residual_probs[0, 0] = 1.0

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

    def _bind_graph_token_merge_metadata(
        self,
        attn_metadata: TokenMergeAttnMetaDataTemplate,
        graph_vars: dict[str, torch.Tensor],
        num_tokens: int,
        source_attn_metadata: TokenMergeAttnMetaDataTemplate | None = None,
    ) -> None:
        graph_merge_mask = graph_vars["token_merge_mask"][:num_tokens]
        graph_topk_ids = graph_vars["token_merge_topk_ids"][:num_tokens]
        graph_topk_probs = graph_vars["token_merge_topk_probs"][:num_tokens]
        graph_residual_probs = graph_vars["token_merge_residual_probs"][:num_tokens]

        graph_merge_mask.zero_()
        graph_topk_ids.fill_(int(self.config.mask_token_id))
        graph_topk_probs.zero_()
        graph_residual_probs.zero_()

        src_metadata = source_attn_metadata if source_attn_metadata is not None else attn_metadata
        src_merge_mask = src_metadata.token_merge_mask
        src_topk_ids = src_metadata.token_merge_topk_ids
        src_topk_probs = src_metadata.token_merge_topk_probs
        src_residual_probs = src_metadata.token_merge_residual_probs
        has_merge = (
            bool(src_metadata.token_merge_enabled)
            and src_merge_mask is not None
            and src_topk_ids is not None
            and src_topk_probs is not None
            and src_residual_probs is not None
        )

        if has_merge:
            if int(src_merge_mask.numel()) != num_tokens:
                raise RuntimeError(
                    "Token-merge metadata length does not match CUDA graph replay size: "
                    f"merge_mask={src_merge_mask.numel()}, num_tokens={num_tokens}"
                )
            src_top_k = int(src_topk_ids.shape[1])
            graph_top_k = int(graph_topk_ids.shape[1])
            if src_top_k > graph_top_k:
                raise RuntimeError(
                    "Token-merge top-k exceeds CUDA graph capture capacity: "
                    f"src_top_k={src_top_k}, graph_top_k={graph_top_k}"
                )

            graph_merge_mask.copy_(src_merge_mask.to(device=graph_merge_mask.device, dtype=torch.bool))
            graph_topk_ids[:, :src_top_k].copy_(src_topk_ids.to(device=graph_topk_ids.device, dtype=torch.int64))
            graph_topk_probs[:, :src_top_k].copy_(
                src_topk_probs.to(device=graph_topk_probs.device, dtype=torch.float32)
            )
            graph_residual_probs.copy_(
                src_residual_probs.to(device=graph_residual_probs.device, dtype=torch.float32)
            )

        attn_metadata.init_token_merge(
            merge_mask=graph_merge_mask,
            topk_ids=graph_topk_ids,
            topk_probs=graph_topk_probs,
            residual_probs=graph_residual_probs,
            mask_token_id=self.config.mask_token_id,
            renormalize=bool(self.config.token_merge_renormalize),
            mode=self.config.token_merge_mode,
            weight=float(self.config.token_merge_weight),
        )

    def _bind_decode_graph_extra_metadata(
        self,
        attn_metadata: TokenMergeAttnMetaDataTemplate,
        graph_vars: dict[str, torch.Tensor],
        num_tokens: int,
        source_attn_metadata: TokenMergeAttnMetaDataTemplate | None = None,
    ) -> None:
        self._bind_graph_token_merge_metadata(
            attn_metadata,
            graph_vars,
            num_tokens,
            source_attn_metadata=source_attn_metadata,
        )

    def _decode_graph_extra_vars(
        self,
        max_num_tokens: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        return self._token_merge_graph_vars(max_num_tokens, device)

    def _init_decode_graph_extra_metadata(
        self,
        attn_metadata: TokenMergeAttnMetaDataTemplate,
        graph_vars: dict[str, torch.Tensor],
        num_tokens: int,
    ) -> None:
        self._init_graph_capture_token_merge_metadata(attn_metadata, graph_vars, num_tokens)
