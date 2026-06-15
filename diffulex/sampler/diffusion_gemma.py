from __future__ import annotations

import torch
import torch.nn.functional as F

from diffulex.layer.linear import tp_all_reduce
from diffulex.logger import get_logger
from diffulex.sampler.auto_sampler import AutoSampler
from diffulex.sampler.base import DllmSamplerNoShiftBase, SampleOutputBase

logger = get_logger(__name__)


@AutoSampler.register("diffusion_gemma", use_full_config=True)
class DiffusionGemmaSampler(DllmSamplerNoShiftBase):
    def __init__(self, config) -> None:
        super().__init__()
        self.max_denoising_steps = int(getattr(config, "diffusion_gemma_max_denoising_steps", 32))
        self.stability_threshold = int(getattr(config, "diffusion_gemma_stability_threshold", 2))
        self.t_min = float(getattr(config, "diffusion_gemma_t_min", 0.0))
        self.t_max = float(getattr(config, "diffusion_gemma_t_max", 1.0))
        self.confidence_threshold = float(getattr(config, "diffusion_gemma_confidence_threshold", 0.1))
        self.entropy_bound = float(getattr(config, "diffusion_gemma_entropy_bound", 1.0))
        self.vocab_size = int(
            getattr(config, "tokenizer_vocab_size", None)
            or getattr(config.hf_config, "vocab_size", 0)
            or getattr(getattr(config.hf_config, "text_config", None), "vocab_size", 0)
            or 0
        )
        self._states: dict[str, dict] = {}
        self._model = None
        self._warned_self_conditioning_unavailable = False

    def evict_req_states(self, req_ids: list[int] | list[str]) -> None:
        for req_id in req_ids:
            self._states.pop(str(req_id), None)

    def bind_model(self, model) -> None:
        self._model = model

    def _temperature_for_step(self, step: int) -> float:
        max_steps = max(1, self.max_denoising_steps)
        remaining_ratio = max(0.0, (max_steps - step) / max_steps)
        return self.t_min + (self.t_max - self.t_min) * remaining_ratio

    @staticmethod
    def _sample_argmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)
        noise = torch.rand_like(logits, dtype=torch.float32).clamp_(1e-6, 1 - 1e-6)
        gumbel = -torch.log(-torch.log(noise))
        return torch.argmax(logits.to(torch.float32) + gumbel * temperature, dim=-1)

    @staticmethod
    def _entropy_bound_mask(entropy: torch.Tensor, entropy_bound: float) -> torch.Tensor:
        if entropy.numel() == 0:
            return torch.zeros_like(entropy, dtype=torch.bool)
        if entropy_bound < 0:
            return torch.ones_like(entropy, dtype=torch.bool)

        order = torch.argsort(entropy, dim=0)
        sorted_entropy = entropy[order]
        cumulative = torch.cumsum(sorted_entropy, dim=0)
        cumulative_max = torch.cummax(sorted_entropy, dim=0).values
        sorted_mask = (cumulative - cumulative_max) <= entropy_bound
        mask = torch.zeros_like(sorted_mask, dtype=torch.bool)
        return mask.scatter(0, order, sorted_mask)

    @staticmethod
    def _extract_decode_block_logits(req, req_logits: torch.Tensor, block) -> torch.Tensor | None:
        if req_logits.shape[0] == 0:
            return None
        buf_offset = int(block.start - req.dllm_block_buffer.first_running_block.start)
        local_start = buf_offset
        local_end = local_start + int(block.block_size)
        if local_start < 0 or local_end > req_logits.shape[0]:
            return None
        return req_logits[local_start:local_end, ...]

    @staticmethod
    def _stable_argmax(history: list[torch.Tensor], threshold: int) -> bool:
        threshold = max(1, int(threshold))
        if len(history) < threshold:
            return False
        latest = history[-1]
        return all(torch.equal(latest, previous) for previous in history[-threshold:])

    def _random_tokens_like(self, tokens: torch.Tensor, vocab_limit: int, mask_token_id: int) -> torch.Tensor:
        random_tokens = torch.randint(0, vocab_limit, tokens.shape, device=tokens.device)
        if 0 <= mask_token_id < vocab_limit and vocab_limit > 1:
            replacement = torch.randint(0, vocab_limit - 1, tokens.shape, device=tokens.device)
            replacement = replacement + (replacement >= mask_token_id).to(replacement.dtype)
            random_tokens = torch.where(random_tokens == mask_token_id, replacement, random_tokens)
        return random_tokens

    def _sanitize_logits(self, logits: torch.Tensor, vocab_limit: int, mask_token_id: int) -> torch.Tensor:
        logits = logits.to(torch.float32)
        logits_min = torch.finfo(logits.dtype).min
        if not torch.isfinite(logits).all():
            logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, logits_min))
        if 0 < vocab_limit < logits.size(-1):
            logits = logits.clone()
            logits[..., vocab_limit:] = logits_min
        if 0 <= mask_token_id < logits.size(-1):
            logits = logits.clone()
            logits[..., mask_token_id] = logits_min
        return logits

    def get_self_conditioning_embeds(self, req_id, block_id) -> torch.Tensor | None:
        state = self._states.get(str(req_id))
        if not state:
            return None
        if bool(state.get("commit_next", False)):
            return None
        embeds_by_block = state.get("self_conditioning_embeds") or {}
        return embeds_by_block.get(str(block_id))

    def _compute_self_conditioning_embeds(
        self,
        probs: torch.Tensor,
        valid_commit_len: int,
        block_size: int,
    ) -> torch.Tensor | None:
        model = self._model
        text_model = getattr(model, "model", None)
        embed_tokens = getattr(text_model, "embed_tokens", None)
        embed_weight = getattr(embed_tokens, "weight", None)
        normalizer = getattr(text_model, "normalizer", None)
        if embed_weight is None or normalizer is None:
            return None
        if int(embed_weight.shape[0]) != int(probs.shape[-1]):
            tp_size = int(getattr(embed_tokens, "tp_size", 1))
            vocab_start = int(getattr(embed_tokens, "vocab_start_idx", 0))
            vocab_end = int(getattr(embed_tokens, "vocab_end_idx", vocab_start + int(embed_weight.shape[0])))
            tp_group = getattr(embed_tokens, "tp_group", None)
            if tp_size > 1 and vocab_end <= int(probs.shape[-1]):
                probs_shard = probs[..., vocab_start:vocab_end]
                soft_embeds = torch.matmul(probs_shard.to(embed_weight.dtype), embed_weight)
                soft_embeds = tp_all_reduce(soft_embeds, tp_group)
                soft_embeds = soft_embeds * normalizer.to(device=soft_embeds.device, dtype=soft_embeds.dtype)
                if valid_commit_len < block_size:
                    soft_embeds = soft_embeds.clone()
                    soft_embeds[valid_commit_len:] = 0
                return soft_embeds.detach()

            if not self._warned_self_conditioning_unavailable:
                logger.warning(
                    "DiffusionGemma self-conditioning disabled because embedding "
                    "vocab partition (%s) does not match logits vocab (%s).",
                    int(embed_weight.shape[0]),
                    int(probs.shape[-1]),
                )
                self._warned_self_conditioning_unavailable = True
            return None

        soft_embeds = torch.matmul(probs.to(embed_weight.dtype), embed_weight)
        soft_embeds = soft_embeds * normalizer.to(device=soft_embeds.device, dtype=soft_embeds.dtype)
        if valid_commit_len < block_size:
            soft_embeds = soft_embeds.clone()
            soft_embeds[valid_commit_len:] = 0
        return soft_embeds.detach()

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
    ) -> SampleOutputBase:
        del temperatures, top_p, top_k, margin_confidence, neg_entropy, kwargs

        attn_metadata = self.fetch_attn_metadata()
        split_logits = self._split_logits_per_req(attn_metadata, reqs, logits)

        true_local_ids_map: dict[str, dict[str, list[int]]] = {}
        accepted_ids_map: dict[str, dict[str, list[int]]] = {}
        sampled_tokens_map: dict[str, dict[str, list[int]]] = {}
        edit_writes_map: dict[str, dict[str, dict[int, int]]] = {}
        block_state_map: dict[str, dict[str, dict]] = {}
        confidence_map: dict[str, dict[str, list[float]]] = {}

        for req_idx, (req, req_logits) in enumerate(zip(reqs, split_logits)):
            req_id_str = str(req.req_id)
            true_local_ids_sub: dict[str, list[int]] = {}
            accepted_ids_sub: dict[str, list[int]] = {}
            sampled_tokens_sub: dict[str, list[int]] = {}
            edit_writes_sub: dict[str, dict[int, int]] = {}
            block_state_sub: dict[str, dict] = {}
            confidence_sub: dict[str, list[float]] = {}

            for block in req.dllm_block_buffer.active_blocks:
                block_id_str = str(block.block_id)
                valid_commit_len = int(getattr(block, "valid_commit_len", block.block_size))
                valid_commit_len = max(0, min(valid_commit_len, int(block.block_size)))

                if attn_metadata.is_prefill[req_idx]:
                    self._states.pop(req_id_str, None)
                    block_state_sub[block_id_str] = {
                        "committable": False,
                        "same_as_previous": False,
                        "same_token_ratio": 0.0,
                        "all_confident": False,
                        "valid_commit_len": valid_commit_len,
                    }
                    continue

                block_logits = self._extract_decode_block_logits(req, req_logits, block)
                if block_logits is None or block_logits.shape[0] != int(block.block_size):
                    block_state_sub[block_id_str] = {
                        "committable": False,
                        "same_as_previous": False,
                        "same_token_ratio": 0.0,
                        "all_confident": False,
                        "valid_commit_len": valid_commit_len,
                    }
                    continue

                vocab_limit = self.vocab_size if self.vocab_size > 0 else block_logits.size(-1)
                vocab_limit = min(vocab_limit, block_logits.size(-1))
                logits_fp32 = self._sanitize_logits(block_logits, vocab_limit, int(block.mask_token_id))
                if valid_commit_len < int(block.block_size):
                    logits_fp32 = logits_fp32.clone()
                    logits_fp32[valid_commit_len:, :] = 0

                state = self._states.setdefault(
                    req_id_str,
                    {"step": 0, "history": [], "commit_next": False},
                )
                if bool(state.get("commit_next", False)):
                    argmax_canvas = state.get("argmax_canvas")
                    if argmax_canvas is None:
                        argmax_tokens = torch.argmax(logits_fp32, dim=-1)
                        block_tokens = argmax_tokens.detach().to("cpu").tolist()
                    else:
                        block_tokens = list(argmax_canvas)
                    true_local_ids_sub[block_id_str] = []
                    accepted_ids_sub[block_id_str] = []
                    sampled_tokens_sub[block_id_str] = block_tokens
                    edit_writes_sub[block_id_str] = {
                        rel_idx: int(token) for rel_idx, token in enumerate(block_tokens)
                    }
                    confidence_sub[block_id_str] = []
                    block_state_sub[block_id_str] = {
                        "committable": True,
                        "same_as_previous": True,
                        "same_token_ratio": 1.0,
                        "all_confident": True,
                        "valid_commit_len": valid_commit_len,
                    }
                    self._states.pop(req_id_str, None)
                    continue

                step = int(state.get("step", 0))
                temperature = self._temperature_for_step(step)
                sampled_tokens = self._sample_argmax(logits_fp32, temperature)
                argmax_tokens = torch.argmax(logits_fp32, dim=-1)

                probs = F.softmax(logits_fp32, dim=-1)
                log_probs = F.log_softmax(logits_fp32, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1)
                entropy_mask = self._entropy_bound_mask(entropy, self.entropy_bound)
                random_tokens = self._random_tokens_like(sampled_tokens, vocab_limit, int(block.mask_token_id))
                denoised_tokens = torch.where(entropy_mask, sampled_tokens, random_tokens)

                history = list(state.get("history", []))
                history.append(argmax_tokens.detach().to("cpu"))
                max_history = max(1, self.stability_threshold)
                history = history[-max_history:]

                next_step = step + 1
                stable = self._stable_argmax(history, self.stability_threshold)
                mean_entropy = float(entropy.mean().item()) if entropy.numel() else 0.0
                confident = mean_entropy <= self.confidence_threshold
                timed_out = next_step >= max(1, self.max_denoising_steps)
                converged = timed_out or (stable and confident)

                if converged:
                    block_tokens = argmax_tokens.detach().to("cpu").tolist()
                    true_local_ids_sub[block_id_str] = []
                    accepted_ids_sub[block_id_str] = []
                    sampled_tokens_sub[block_id_str] = block_tokens
                    edit_writes_sub[block_id_str] = {
                        rel_idx: int(token) for rel_idx, token in enumerate(block_tokens)
                    }
                    state["step"] = next_step
                    state["history"] = history
                    state["argmax_canvas"] = block_tokens
                    state["commit_next"] = True
                    embeds_by_block = state.get("self_conditioning_embeds")
                    if embeds_by_block is not None:
                        embeds_by_block.pop(block_id_str, None)
                else:
                    block_tokens = denoised_tokens.detach().to("cpu").tolist()
                    true_local_ids_sub[block_id_str] = []
                    accepted_ids_sub[block_id_str] = []
                    sampled_tokens_sub[block_id_str] = block_tokens
                    edit_writes_sub[block_id_str] = {
                        rel_idx: int(token) for rel_idx, token in enumerate(block_tokens)
                    }
                    state["step"] = next_step
                    state["history"] = history
                    embeds = self._compute_self_conditioning_embeds(
                        probs,
                        valid_commit_len=valid_commit_len,
                        block_size=int(block.block_size),
                    )
                    if embeds is not None:
                        embeds_by_block = state.setdefault("self_conditioning_embeds", {})
                        embeds_by_block[block_id_str] = embeds
                    else:
                        embeds_by_block = state.get("self_conditioning_embeds")
                        if embeds_by_block is not None:
                            embeds_by_block.pop(block_id_str, None)

                confidence_sub[block_id_str] = (-entropy).detach().to("cpu").tolist()
                block_state_sub[block_id_str] = {
                    "committable": False,
                    "same_as_previous": stable,
                    "same_token_ratio": 1.0 if stable else 0.0,
                    "all_confident": confident,
                    "valid_commit_len": valid_commit_len,
                }

            true_local_ids_map[req_id_str] = true_local_ids_sub
            accepted_ids_map[req_id_str] = accepted_ids_sub
            sampled_tokens_map[req_id_str] = sampled_tokens_sub
            edit_writes_map[req_id_str] = edit_writes_sub
            block_state_map[req_id_str] = block_state_sub
            confidence_map[req_id_str] = confidence_sub

        return SampleOutputBase(
            true_local_ids_map=true_local_ids_map,
            accepted_ids_map=accepted_ids_map,
            sampled_tokens_map=sampled_tokens_map,
            confidence_map=confidence_map,
            edit_writes_map=edit_writes_map,
            block_state_map=block_state_map,
        )
