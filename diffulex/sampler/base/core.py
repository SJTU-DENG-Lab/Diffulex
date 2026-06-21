from __future__ import annotations

import os

import torch
import torch.distributions as dists
import torch.nn as nn
import torch.nn.functional as F

from diffulex.profiling import record_function


class SamplerBase(nn.Module):
    def __init__(self):
        super().__init__()
        from diffulex.attention import fetch_attn_metadata

        self.fetch_attn_metadata = fetch_attn_metadata
        self.tokenizer_vocab_size: int | None = None

    def _compute_accepted_ids_cpu(
        self,
        block,
        confidence: list[float],
        initial_confidence: list[float],
        sampled_tokens: list[int],
        **kwargs,
    ) -> list[int] | None:
        del block, confidence, initial_confidence, sampled_tokens, kwargs
        return None

    def _can_compute_accepted_ids_cpu(self, block) -> bool:
        del block
        return False

    def _materialize_sampled_block(
        self,
        block,
        confidence: torch.Tensor,
        sampled_tokens: torch.Tensor,
        initial_confidence: torch.Tensor,
        **kwargs,
    ) -> tuple[list[int], list[int], list[float], list[float]]:
        with record_function("diffulex.sampler.materialize.can_cpu"):
            can_compute_cpu = self._can_compute_accepted_ids_cpu(block)
        if not can_compute_cpu:
            with record_function("diffulex.sampler.materialize.accepted_gpu"):
                accepted_ids = self._compute_accepted_ids(
                    block, confidence, initial_confidence, sampled_tokens, **kwargs
                )
            with record_function("diffulex.sampler.materialize.to_cpu_lists"):
                accepted_ids_list = [int(idx) for idx in accepted_ids.to(device="cpu").tolist()]
                sampled_tokens_list = [int(token) for token in sampled_tokens.to(device="cpu").tolist()]
                confidence_list = [float(value) for value in confidence.to(device="cpu").tolist()]
                initial_confidence_list = [
                    float(value) for value in initial_confidence.to(device="cpu").tolist()
                ]
            return accepted_ids_list, sampled_tokens_list, confidence_list, initial_confidence_list

        with record_function("diffulex.sampler.materialize.pack"):
            packed = torch.stack(
                (
                    sampled_tokens.to(dtype=torch.float32),
                    confidence.to(dtype=torch.float32),
                    initial_confidence.to(dtype=torch.float32),
                ),
                dim=0,
            )
        with record_function("diffulex.sampler.materialize.to_cpu_lists"):
            sampled_tokens_raw, confidence_raw, initial_confidence_raw = packed.to(device="cpu").tolist()
            sampled_tokens_list = [int(token) for token in sampled_tokens_raw]
            confidence_list = [float(value) for value in confidence_raw]
            initial_confidence_list = [float(value) for value in initial_confidence_raw]
        with record_function("diffulex.sampler.materialize.accepted_cpu"):
            accepted_ids_list = self._compute_accepted_ids_cpu(
                block,
                confidence=confidence_list,
                initial_confidence=initial_confidence_list,
                sampled_tokens=sampled_tokens_list,
                **kwargs,
            )
        if accepted_ids_list is None:
            accepted_ids_list = []
        return [int(idx) for idx in accepted_ids_list], sampled_tokens_list, confidence_list, initial_confidence_list

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
        with record_function("diffulex.sampler.sample_tokens.temperature"):
            if torch.is_tensor(temperature):
                temperature = float(temperature.item())
            else:
                temperature = float(temperature)

        logits_min = torch.finfo(logits.dtype).min
        logits_neg_inf = torch.tensor(float("-inf"), dtype=logits.dtype, device=logits.device)

        with record_function("diffulex.sampler.sample_tokens.sanitize"):
            if os.getenv("DIFFULEX_SANITIZE_LOGITS", "0") == "1" and not torch.isfinite(logits).all():
                logits = logits.clone()
                logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, logits_min))

        with record_function("diffulex.sampler.sample_tokens.apply_masks"):
            needs_vocab_mask = (
                self.tokenizer_vocab_size is not None
                and 0 <= self.tokenizer_vocab_size < logits.size(-1)
            )
            valid_forbidden_token_ids = [
                int(token_id)
                for token_id in (forbidden_token_ids or [])
                if 0 <= int(token_id) < logits.size(-1)
            ]
            if needs_vocab_mask or valid_forbidden_token_ids:
                logits = logits.clone()
                if needs_vocab_mask:
                    logits[..., self.tokenizer_vocab_size :] = logits_neg_inf
                for token_id in valid_forbidden_token_ids:
                    logits[..., token_id] = logits_neg_inf

        with record_function("diffulex.sampler.sample_tokens.temperature_scale"):
            if temperature > 0:
                logits = logits / temperature

        with record_function("diffulex.sampler.sample_tokens.top_p"):
            if top_p is not None and top_p < 1:
                logits = self.top_p_logits(logits, top_p)

        with record_function("diffulex.sampler.sample_tokens.top_k"):
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

        with record_function("diffulex.sampler.sample_tokens.softmax"):
            probs = torch.softmax(logits, dim=-1)

        with record_function("diffulex.sampler.sample_tokens.select"):
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

        with record_function("diffulex.sampler.sample_tokens.confidence_clone"):
            confidence = initial_confidence.clone()

        with record_function("diffulex.sampler.sample_tokens.margin_confidence"):
            if margin_confidence:
                sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
                top1_probs = sorted_probs[:, 0]
                top2_probs = sorted_probs[:, 1]
                confidence = top1_probs - top2_probs

        with record_function("diffulex.sampler.sample_tokens.neg_entropy"):
            if neg_entropy:
                epsilon = 1e-10
                log_probs = torch.log(probs + epsilon)
                confidence = torch.sum(probs * log_probs, dim=-1)

        return confidence, x0, initial_confidence
