from __future__ import annotations

import torch


class TokenMergeSamplerMixin:
    def __init__(self, *args, config=None, token_merge_top_k: int | None = None, **kwargs):
        if token_merge_top_k is None:
            token_merge_top_k = int(getattr(config, "token_merge_top_k", 1)) if config is not None else 1
        self.token_merge_top_k = max(1, int(token_merge_top_k))
        self._last_token_merge_map: dict[str, dict[int, dict | None]] = {}
        super().__init__(*args, config=config, **kwargs)

    def _reset_token_merge_map(self) -> None:
        self._last_token_merge_map = {}

    def _set_token_merge_entries(self, req_id_str: str, entries: dict[int, dict | None]) -> None:
        self._last_token_merge_map[req_id_str] = entries

    def _build_token_merge_descriptor(
        self,
        probs: torch.Tensor,
        token: int,
        mask_id: int,
    ) -> dict | None:
        if token == mask_id:
            return None

        top_k = min(self.token_merge_top_k, probs.shape[-1])
        topk_probs, topk_ids = torch.topk(probs, top_k, dim=-1)
        residual_prob = max(0.0, 1.0 - float(topk_probs.sum().item()))
        return {
            "topk_ids": topk_ids.to(dtype=torch.int64).tolist(),
            "topk_probs": topk_probs.tolist(),
            "residual_prob": residual_prob,
        }

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
        sample_output.token_merge_map = self._last_token_merge_map
        return sample_output
