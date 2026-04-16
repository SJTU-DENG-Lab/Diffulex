from __future__ import annotations

import torch
import torch.distributions as dists
import torch.nn as nn
import torch.nn.functional as F


class SamplerBase(nn.Module):
    def __init__(self):
        super().__init__()
        from diffulex.attention import fetch_attn_metadata

        self.fetch_attn_metadata = fetch_attn_metadata
        self.tokenizer_vocab_size: int | None = None

    def top_p_logits(self, logits, top_p):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
        mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)

        return logits

    def top_k_logits(self, logits, top_k):
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)

        return logits

    def sample_tokens(
        self,
        logits,
        temperature=0.0,
        top_p=None,
        top_k=None,
        margin_confidence=False,
        neg_entropy=False,
        forbidden_token_ids: list[int] | None = None,
    ):
        logits_min = torch.finfo(logits.dtype).min
        logits_neg_inf = torch.tensor(float("-inf"), dtype=logits.dtype, device=logits.device)

        if not torch.isfinite(logits).all():
            logits = logits.clone()
            logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, logits_min))

        if self.tokenizer_vocab_size is not None and 0 <= self.tokenizer_vocab_size < logits.size(-1):
            logits = logits.clone()
            logits[..., self.tokenizer_vocab_size :] = logits_neg_inf

        if forbidden_token_ids:
            logits = logits.clone()
            for token_id in forbidden_token_ids:
                if 0 <= token_id < logits.size(-1):
                    logits[..., token_id] = logits_neg_inf

        if temperature > 0:
            logits = logits / temperature

        if top_p is not None and top_p < 1:
            logits = self.top_p_logits(logits, top_p)

        if top_k is not None:
            logits = self.top_k_logits(logits, top_k)

        probs = torch.softmax(logits, dim=-1)

        if temperature > 0:
            try:
                x0 = dists.Categorical(probs=probs).sample()
                initial_confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
            except Exception:
                initial_confidence, x0 = probs.max(dim=-1)
        else:
            max_logits = logits.max(dim=-1).values
            tie_mask = logits == max_logits.unsqueeze(-1)
            # Greedy decode should be deterministic even when bf16 quantization makes
            # multiple vocab entries exactly tie for top-1. Prefer the highest token id
            # among the surviving logits after sanitization and masking.
            x0 = logits.size(-1) - 1 - tie_mask.flip(dims=[-1]).to(torch.int32).argmax(dim=-1)
            initial_confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

        confidence = initial_confidence.clone()

        if margin_confidence:
            sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
            top1_probs = sorted_probs[:, 0]
            top2_probs = sorted_probs[:, 1]
            confidence = top1_probs - top2_probs

        if neg_entropy:
            epsilon = 1e-10
            log_probs = torch.log(probs + epsilon)
            confidence = torch.sum(probs * log_probs, dim=-1)

        return confidence, x0, initial_confidence
