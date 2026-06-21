from __future__ import annotations

import os
import torch
import torch.nn.functional as F

from diffulex.mixin import BlockRewriteSamplerMixin, TokenMergeSamplerMixin
from diffulex.profiling import record_function
from diffulex.sampler.auto_sampler import AutoSampler
from diffulex.sampler.base import DllmSamplerNoShiftBase


def _llada2_greedy_sample_eager(
    logits: torch.Tensor,
    tokenizer_vocab_size: int,
    mask_token_id: int,
    sanitize_logits: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_min = torch.finfo(logits.dtype).min
    logits_neg_inf = torch.tensor(float("-inf"), dtype=logits.dtype, device=logits.device)

    if sanitize_logits and not torch.isfinite(logits).all():
        logits = logits.clone()
        logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, logits_min))

    needs_vocab_mask = 0 <= int(tokenizer_vocab_size) < logits.size(-1)
    needs_mask_token_mask = 0 <= int(mask_token_id) < logits.size(-1)
    if needs_vocab_mask or needs_mask_token_mask:
        logits = logits.clone()
        if needs_vocab_mask:
            logits[..., int(tokenizer_vocab_size) :] = logits_neg_inf
        if needs_mask_token_mask:
            logits[..., int(mask_token_id)] = logits_neg_inf

    probs = torch.softmax(logits, dim=-1)
    max_logits = logits.max(dim=-1).values
    tie_mask = logits == max_logits.unsqueeze(-1)
    sampled_tokens = logits.size(-1) - 1 - tie_mask.flip(dims=[-1]).to(torch.int32).argmax(dim=-1)
    initial_confidence = torch.gather(probs, -1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
    confidence = initial_confidence.clone()
    return confidence, sampled_tokens, initial_confidence


def _temperature_value(temperatures, req_idx: int) -> float:
    temperature = temperatures[req_idx]
    if torch.is_tensor(temperature):
        return float(temperature.item())
    return float(temperature)


class LLaDA2AcceptedIdsMixin:
    def _can_compute_accepted_ids_cpu(self, block) -> bool:
        del block
        return True

    def _compute_accepted_ids_cpu(
        self,
        block,
        confidence: list[float],
        initial_confidence: list[float],
        sampled_tokens: list[int],
        **kwargs,
    ) -> list[int] | None:
        del sampled_tokens, kwargs
        accept_threshold = float(block.thresholds.accept_threshold)
        pre_block_complete = block.prev_block.is_semi_complete if block.prev_block else True

        high_conf_indices = [
            idx for idx, value in enumerate(initial_confidence) if float(value) > accept_threshold
        ]
        if pre_block_complete:
            if not high_conf_indices:
                if not confidence:
                    return []
                best_idx = max(range(len(confidence)), key=lambda idx: confidence[idx])
                return [int(best_idx)]
            return sorted(set(int(idx) for idx in high_conf_indices))
        return [int(idx) for idx in high_conf_indices]

    def _compute_accepted_ids(
        self,
        block,
        confidence: torch.Tensor,
        initial_confidence: torch.Tensor,
        sampled_tokens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        accept_threshold = block.thresholds.accept_threshold
        pre_block_complete = block.prev_block.is_semi_complete if block.prev_block else True

        high_conf_indices = torch.where(initial_confidence > accept_threshold)[0]
        if pre_block_complete:
            if len(high_conf_indices) == 0:
                _, transfer_index = torch.topk(confidence, 1)
                return transfer_index
            transfer_index = torch.tensor([], device=sampled_tokens.device, dtype=torch.long)
            return torch.unique(torch.cat([transfer_index, high_conf_indices]))
        return high_conf_indices


class LLaDA2Sampler(LLaDA2AcceptedIdsMixin, DllmSamplerNoShiftBase):
    def __init__(self, config=None):
        del config
        super().__init__()


class LLaDA2VectorizedSampler(LLaDA2Sampler):
    def __init__(self, config=None):
        super().__init__(config=config)
        self._enable_vectorized_compile = bool(getattr(config, "enable_vectorized_sampler_compile", False))
        self._tensor_parallel_size = int(getattr(config, "tensor_parallel_size", 1))
        self._compiled_greedy_sample = None
        self._compile_failed = False
        self._fused_greedy_unavailable = False

    def _can_vectorize_request(
        self,
        temperatures: torch.Tensor | None,
        top_p,
        top_k,
        margin_confidence,
        neg_entropy,
    ) -> bool:
        if self._tensor_parallel_size != 1:
            return False
        if top_p is not None or top_k is not None:
            return False
        if margin_confidence not in (False, None):
            return False
        if neg_entropy not in (False, None):
            return False
        if temperatures is None:
            return False
        if torch.is_tensor(temperatures):
            if temperatures.numel() == 0:
                return True
            return not bool((temperatures != 0).any().item())
        return all(float(value) == 0.0 for value in temperatures)

    def _greedy_sample(
        self,
        logits: torch.Tensor,
        mask_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokenizer_vocab_size = self.tokenizer_vocab_size
        vocab_limit = -1 if tokenizer_vocab_size is None else int(tokenizer_vocab_size)
        sanitize_logits = os.getenv("DIFFULEX_SANITIZE_LOGITS", "0") == "1"

        use_experimental_fused_greedy = (
            os.getenv("DIFFULEX_ENABLE_EXPERIMENTAL_FUSED_GREEDY_CONFIDENCE", "0") == "1"
        )
        if (
            use_experimental_fused_greedy
            and logits.is_cuda
            and not sanitize_logits
            and not self._fused_greedy_unavailable
        ):
            try:
                from diffulex_kernel import greedy_confidence

                return greedy_confidence(
                    logits,
                    vocab_limit=vocab_limit,
                    forbidden_token_id=int(mask_token_id),
                )
            except Exception:
                self._fused_greedy_unavailable = True

        if not self._enable_vectorized_compile or self._compile_failed:
            return _llada2_greedy_sample_eager(logits, vocab_limit, int(mask_token_id), sanitize_logits)

        if self._compiled_greedy_sample is None:
            try:
                self._compiled_greedy_sample = torch.compile(
                    _llada2_greedy_sample_eager,
                    dynamic=True,
                    fullgraph=False,
                )
            except Exception:
                self._compile_failed = True
                return _llada2_greedy_sample_eager(logits, vocab_limit, int(mask_token_id), sanitize_logits)

        try:
            return self._compiled_greedy_sample(logits, vocab_limit, int(mask_token_id), sanitize_logits)
        except Exception:
            self._compile_failed = True
            return _llada2_greedy_sample_eager(logits, vocab_limit, int(mask_token_id), sanitize_logits)

    @staticmethod
    def _decode_mask_token_local_ids(req, block) -> list[int]:
        buf_offset = int(block.start - req.dllm_block_buffer.first_running_block.start)
        return [buf_offset + int(rel_id) for rel_id in block.mask_token_relative_ids]

    @staticmethod
    def _sample_output_plain(output) -> dict:
        return {
            "true_local_ids_map": output.true_local_ids_map,
            "accepted_ids_map": output.accepted_ids_map,
            "sampled_tokens_map": output.sampled_tokens_map,
            "mask_token_rel_ids_map": output.mask_token_rel_ids_map,
            "confidence_map": output.confidence_map,
            "initial_confidence_map": output.initial_confidence_map,
        }

    @staticmethod
    def _confidence_maps_close(lhs: dict, rhs: dict) -> bool:
        if lhs.keys() != rhs.keys():
            return False
        for req_id, lhs_blocks in lhs.items():
            rhs_blocks = rhs[req_id]
            if lhs_blocks.keys() != rhs_blocks.keys():
                return False
            for block_id, lhs_values in lhs_blocks.items():
                rhs_values = rhs_blocks[block_id]
                if len(lhs_values) != len(rhs_values):
                    return False
                for lhs_value, rhs_value in zip(lhs_values, rhs_values):
                    if abs(float(lhs_value) - float(rhs_value)) > 2e-3:
                        return False
        return True

    def _validate_against_legacy(
        self,
        vectorized_output,
        reqs,
        logits: torch.Tensor,
        temperatures,
        top_p,
        top_k,
        margin_confidence,
        neg_entropy,
        **kwargs,
    ) -> None:
        if os.getenv("DIFFULEX_VALIDATE_VECTORIZED_SAMPLER", "0") != "1":
            return
        with record_function("diffulex.sampler.vectorized.validate_legacy"):
            legacy_output = super().forward(
                reqs,
                logits,
                temperatures,
                top_p=top_p,
                top_k=top_k,
                margin_confidence=margin_confidence,
                neg_entropy=neg_entropy,
                **kwargs,
            )
        vectorized_plain = self._sample_output_plain(vectorized_output)
        legacy_plain = self._sample_output_plain(legacy_output)
        for key in (
            "true_local_ids_map",
            "accepted_ids_map",
            "sampled_tokens_map",
            "mask_token_rel_ids_map",
        ):
            if vectorized_plain[key] != legacy_plain[key]:
                raise AssertionError("LLaDA2 vectorized sampler output does not match legacy sampler")
        for key in ("confidence_map", "initial_confidence_map"):
            if not self._confidence_maps_close(vectorized_plain[key], legacy_plain[key]):
                raise AssertionError("LLaDA2 vectorized sampler confidence does not match legacy sampler")

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
    ):
        if not self._can_vectorize_request(temperatures, top_p, top_k, margin_confidence, neg_entropy):
            return super().forward(
                reqs,
                logits,
                temperatures,
                top_p=top_p,
                top_k=top_k,
                margin_confidence=margin_confidence,
                neg_entropy=neg_entropy,
                **kwargs,
            )

        with record_function("diffulex.sampler.vectorized.fetch_attn_metadata"):
            attn_metadata = self.fetch_attn_metadata()
        with record_function("diffulex.sampler.vectorized.split_logits"):
            split_logits = self._split_logits_per_req(attn_metadata, reqs, logits)

        true_local_ids_map: dict[str, dict[str, list[int]]] = {}
        accepted_ids_map: dict[str, dict[str, list[int]]] = {}
        sampled_tokens_map: dict[str, dict[str, list[int]]] = {}
        mask_token_rel_ids_map: dict[str, dict[str, list[int]]] = {}
        confidence_map: dict[str, dict[str, list[float]]] = {}
        initial_confidence_map: dict[str, dict[str, list[float]]] = {}

        records: list[dict] = []
        row_indices: list[int] = []
        offset = 0
        mask_token_id: int | None = None

        with record_function("diffulex.sampler.vectorized.collect_rows"):
            for req_idx, (req, req_logits) in enumerate(zip(reqs, split_logits)):
                req_id_str = str(req.req_id)
                true_local_ids_map[req_id_str] = {}
                accepted_ids_map[req_id_str] = {}
                sampled_tokens_map[req_id_str] = {}
                mask_token_rel_ids_map[req_id_str] = {}
                confidence_map[req_id_str] = {}
                initial_confidence_map[req_id_str] = {}

                for block_id, block in enumerate(req.dllm_blocks):
                    if not block.is_active or block.num_mask_tokens == 0:
                        continue
                    if len(block.mask_token_global_ids) == 0:
                        continue

                    current_mask_token_id = int(block.mask_token_id)
                    if mask_token_id is None:
                        mask_token_id = current_mask_token_id
                    elif current_mask_token_id != mask_token_id:
                        return super().forward(
                            reqs,
                            logits,
                            temperatures,
                            top_p=top_p,
                            top_k=top_k,
                            margin_confidence=margin_confidence,
                            neg_entropy=neg_entropy,
                            **kwargs,
                        )

                    if attn_metadata.is_prefill[req_idx]:
                        if req_logits.shape[0] == 0:
                            continue
                        local_ids = self._prefill_mask_token_local_ids(req, block, req_logits)
                    else:
                        local_ids = self._decode_mask_token_local_ids(req, block)

                    if not local_ids:
                        continue
                    if min(local_ids) < 0 or max(local_ids) >= req_logits.shape[0]:
                        return super().forward(
                            reqs,
                            logits,
                            temperatures,
                            top_p=top_p,
                            top_k=top_k,
                            margin_confidence=margin_confidence,
                            neg_entropy=neg_entropy,
                            **kwargs,
                        )

                    row_start = len(row_indices)
                    row_indices.extend(offset + int(local_id) for local_id in local_ids)
                    records.append(
                        {
                            "req_id_str": req_id_str,
                            "block_id_str": str(block_id),
                            "block": block,
                            "row_start": row_start,
                            "row_count": len(local_ids),
                        }
                    )
                offset += int(req_logits.shape[0])

        if not records:
            sample_output = self.output_cls(
                true_local_ids_map=true_local_ids_map,
                accepted_ids_map=accepted_ids_map,
                sampled_tokens_map=sampled_tokens_map,
                mask_token_rel_ids_map=mask_token_rel_ids_map,
                confidence_map=confidence_map,
                initial_confidence_map=initial_confidence_map,
            )
            self._validate_against_legacy(
                sample_output,
                reqs,
                logits,
                temperatures,
                top_p,
                top_k,
                margin_confidence,
                neg_entropy,
                **kwargs,
            )
            return sample_output

        with record_function("diffulex.sampler.vectorized.gather_logits"):
            index = torch.tensor(row_indices, dtype=torch.long, device=logits.device)
            flat_mask_logits = logits.index_select(0, index)

        with record_function("diffulex.sampler.vectorized.greedy_sample"):
            confidence, sampled_tokens, initial_confidence = self._greedy_sample(
                flat_mask_logits,
                int(mask_token_id),
            )

        with record_function("diffulex.sampler.vectorized.to_cpu"):
            packed = torch.stack(
                (
                    sampled_tokens.to(dtype=torch.float32),
                    confidence.to(dtype=torch.float32),
                    initial_confidence.to(dtype=torch.float32),
                ),
                dim=0,
            )
            sampled_tokens_raw, confidence_raw, initial_confidence_raw = packed.to(device="cpu").tolist()

        with record_function("diffulex.sampler.vectorized.build_maps"):
            for record in records:
                start = int(record["row_start"])
                end = start + int(record["row_count"])
                block = record["block"]
                sampled_tokens_list = [int(token) for token in sampled_tokens_raw[start:end]]
                confidence_list = [float(value) for value in confidence_raw[start:end]]
                initial_confidence_list = [float(value) for value in initial_confidence_raw[start:end]]
                accepted_ids_list = self._compute_accepted_ids_cpu(
                    block,
                    confidence=confidence_list,
                    initial_confidence=initial_confidence_list,
                    sampled_tokens=sampled_tokens_list,
                    **kwargs,
                )
                if accepted_ids_list is None:
                    accepted_ids_list = []

                req_id_str = record["req_id_str"]
                block_id_str = record["block_id_str"]
                accepted_ids_list = [int(idx) for idx in accepted_ids_list]
                true_local_ids_map[req_id_str][block_id_str] = [
                    block.mask_token_relative_ids[i] for i in accepted_ids_list
                ]
                accepted_ids_map[req_id_str][block_id_str] = accepted_ids_list
                sampled_tokens_map[req_id_str][block_id_str] = sampled_tokens_list
                mask_token_rel_ids_map[req_id_str][block_id_str] = list(block.mask_token_relative_ids)
                confidence_map[req_id_str][block_id_str] = confidence_list
                initial_confidence_map[req_id_str][block_id_str] = initial_confidence_list

        sample_output = self.output_cls(
            true_local_ids_map=true_local_ids_map,
            accepted_ids_map=accepted_ids_map,
            sampled_tokens_map=sampled_tokens_map,
            mask_token_rel_ids_map=mask_token_rel_ids_map,
            confidence_map=confidence_map,
            initial_confidence_map=initial_confidence_map,
        )
        self._validate_against_legacy(
            sample_output,
            reqs,
            logits,
            temperatures,
            top_p,
            top_k,
            margin_confidence,
            neg_entropy,
            **kwargs,
        )
        return sample_output


class LLaDA2dot1Sampler(BlockRewriteSamplerMixin, LLaDA2Sampler):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.edit_threshold = float(getattr(config, "edit_threshold", 0.0))
        self.max_post_edit_steps = int(getattr(config, "max_post_edit_steps", 16))
        self.penalty_lambda = float(getattr(config, "penalty_lambda", 0.0))
        self._last_block_state_map: dict[str, dict[str, dict]] = {}

    def _reset_block_state_map(self) -> None:
        self._last_block_state_map = {}

    @staticmethod
    def _sample_argmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature > 0:
            gumbel = -torch.log(-torch.log(torch.rand_like(logits, dtype=torch.float32).clamp_(1e-6, 1 - 1e-6)))
            logits = logits.to(torch.float32) + gumbel * temperature
        return torch.argmax(logits, dim=-1)

    @staticmethod
    def _extract_block_logits(req, req_logits: torch.Tensor, block, is_prefill: bool) -> torch.Tensor | None:
        if req_logits.shape[0] == 0:
            return None
        if is_prefill:
            prefix_offset = int(req.contiguous_in_cache_prefix_len)
            local_start = int(block.start - prefix_offset)
            local_end = int(block.end - prefix_offset)
            if local_start < 0 or local_end > req_logits.shape[0]:
                return None
            return req_logits[local_start:local_end, ...]

        buf_offset = int(block.start - req.dllm_block_buffer.first_running_block.start)
        local_start = buf_offset
        local_end = buf_offset + int(block.block_size)
        if local_start < 0 or local_end > req_logits.shape[0]:
            return None
        return req_logits[local_start:local_end, ...]

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
    ):
        del top_p, top_k, margin_confidence, neg_entropy
        attn_metadata = self.fetch_attn_metadata()
        split_logits = self._split_logits_per_req(attn_metadata, reqs, logits)

        empty_per_req = {str(req.req_id): {} for req in reqs}
        sample_output = self.output_cls(
            true_local_ids_map=dict(empty_per_req),
            accepted_ids_map=dict(empty_per_req),
            sampled_tokens_map=dict(empty_per_req),
            mask_token_rel_ids_map=dict(empty_per_req),
            confidence_map=dict(empty_per_req),
            initial_confidence_map=dict(empty_per_req),
        )
        return self._postprocess_sample_output(
            reqs=reqs,
            split_logits=split_logits,
            temperatures=temperatures,
            sample_output=sample_output,
            attn_metadata=attn_metadata,
            **kwargs,
        )

    def _build_block_writes_map(
        self,
        reqs,
        split_logits,
        temperatures: torch.Tensor,
        sample_output,
        attn_metadata,
        **kwargs,
    ) -> dict[str, dict[str, dict[int, int]]]:
        del sample_output, kwargs
        block_writes_map: dict[str, dict[str, dict[int, int]]] = {}
        self._reset_block_state_map()

        for req_idx, (req, req_logits) in enumerate(zip(reqs, split_logits)):
            req_id_str = str(req.req_id)
            req_block_writes: dict[str, dict[int, int]] = {}
            req_block_states: dict[str, dict] = {}

            for block_id, block in enumerate(req.dllm_blocks):
                if not block.is_active:
                    continue

                post_edit_steps = int(getattr(block, "post_edit_steps", 0))
                total_steps = int(getattr(block, "total_steps", 0))
                editable_start = int(getattr(block, "editable_start", 0) or 0)
                accept_threshold = float(block.thresholds.accept_threshold)
                mask_id = int(block.mask_token_id)

                block_logits = self._extract_block_logits(
                    req, req_logits, block, attn_metadata.is_prefill[req_idx]
                )
                if block_logits is None or block_logits.shape[0] != int(block.block_size):
                    continue

                block_tokens = torch.tensor(block.token_ids, dtype=torch.long, device=block_logits.device)
                block_size = int(block.block_size)

                # Argmax + confidence over full block
                temperature = _temperature_value(temperatures, req_idx)
                x = self._sample_argmax(block_logits, temperature)
                logits_fp32 = block_logits.to(torch.float32)

                # Penalty lambda: penalize predicting the adjacent previous token
                if self.penalty_lambda > 0 and block_size > 1:
                    prev_ids = block_tokens[:-1]
                    logits_fp32[1:, :].scatter_(
                        1, prev_ids.unsqueeze(-1), -self.penalty_lambda, reduce="add"
                    )
                    x = torch.argmax(logits_fp32, dim=-1)

                top1_logits = logits_fp32.gather(dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
                logsumexp = torch.logsumexp(logits_fp32, dim=-1)
                p = torch.exp(top1_logits - logsumexp)

                mask_index = block_tokens.eq(mask_id)
                has_mask = bool(mask_index.any().item())

                # M2T: mask-to-token transfers
                mask_transfer_index = torch.zeros(block_size, dtype=torch.bool, device=block_tokens.device)
                if has_mask:
                    eligible_mask = mask_index & (
                        torch.arange(block_size, device=block_tokens.device) >= editable_start
                    )
                    confidence_at_mask = torch.where(
                        eligible_mask, p,
                        torch.tensor(-float("inf"), device=p.device, dtype=p.dtype),
                    )
                    mask_transfer_index = confidence_at_mask > accept_threshold
                    if not mask_transfer_index.any():
                        _, select_index = torch.topk(confidence_at_mask, k=1)
                        mask_transfer_index = torch.zeros(block_size, dtype=torch.bool, device=block_tokens.device)
                        mask_transfer_index[select_index] = True
                else:
                    post_edit_steps += 1

                # T2T: token-to-token edits on editable, non-mask positions
                editable_positions = torch.arange(block_size, device=block_tokens.device) >= editable_start
                edit_positions = ~mask_index & editable_positions
                edit_transfer_index = (
                    (p > self.edit_threshold) & (block_tokens != x) & edit_positions
                )

                transfer_index = mask_transfer_index | edit_transfer_index

                # Hard upper bound: block_size + max_post_edit_steps
                max_steps = block_size + self.max_post_edit_steps
                timed_out = total_steps >= max_steps

                # Determine finished state
                finished = False
                if timed_out:
                    finished = True
                elif not transfer_index.any():
                    finished = True
                elif not has_mask and post_edit_steps > self.max_post_edit_steps:
                    finished = True

                # Build block_writes
                block_writes: dict[int, int] = {}
                if timed_out:
                    # Force-fill remaining editable mask positions with argmax
                    eligible_mask = mask_index & (
                        torch.arange(block_size, device=block_tokens.device) >= editable_start
                    )
                    for rel_idx in torch.nonzero(eligible_mask, as_tuple=False).flatten().tolist():
                        block_writes[int(rel_idx)] = int(x[int(rel_idx)].item())
                    # Also include any T2T edits that would have happened
                    for rel_idx in torch.nonzero(edit_transfer_index, as_tuple=False).flatten().tolist():
                        rel_idx_int = int(rel_idx)
                        if rel_idx_int >= editable_start:
                            block_writes[rel_idx_int] = int(x[rel_idx_int].item())
                elif not finished:
                    for rel_idx in torch.nonzero(transfer_index, as_tuple=False).flatten().tolist():
                        rel_idx_int = int(rel_idx)
                        if rel_idx_int >= editable_start:
                            block_writes[rel_idx_int] = int(x[rel_idx_int].item())

                # Persist counters on block
                block.post_edit_steps = post_edit_steps

                # Block state for scheduler
                same_as_previous = not bool(transfer_index.any().item()) and not timed_out
                comparable = ~mask_index & editable_positions
                if comparable.any() and not timed_out:
                    same_token_ratio = float(
                        x[comparable].eq(block_tokens[comparable]).to(torch.float32).mean().item()
                    )
                else:
                    same_token_ratio = 1.0
                all_confident = bool((p >= accept_threshold).all().item()) if p.numel() > 0 else True

                req_block_states[str(block.block_id)] = {
                    "committable": finished or (same_as_previous and not has_mask),
                    "same_as_previous": same_as_previous,
                    "same_token_ratio": same_token_ratio,
                    "all_confident": all_confident,
                }

                if block_writes:
                    req_block_writes[str(block.block_id)] = block_writes

            block_writes_map[req_id_str] = req_block_writes
            self._last_block_state_map[req_id_str] = req_block_states

        return block_writes_map

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
        sample_output.block_state_map = self._last_block_state_map
        return sample_output


class LLaDA2DMaxSampler(TokenMergeSamplerMixin, LLaDA2dot1Sampler):
    def __init__(self, config=None):
        super().__init__(config=config)
        self._token_merge_mode = str(getattr(config, "token_merge_mode", "dmax_topk"))
        self._enable_token_merge = bool(
            self._token_merge_mode in {"dmax_topk", "iter_smooth_topk"}
            and float(getattr(config, "token_merge_weight", 1.0)) > 0.0
        )
        self._last_block_state_map: dict[str, dict[str, dict]] = {}
        self._fast_prob_path = bool(getattr(config, "dmax_sampler_fast_path", True))
        del config

    def _compute_accepted_ids(
        self,
        block,
        confidence: torch.Tensor,
        initial_confidence: torch.Tensor,
        sampled_tokens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del block, confidence, initial_confidence, kwargs
        return torch.empty(0, dtype=torch.long, device=sampled_tokens.device)

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
    ):
        # DMax derives writes/token-merge state from full block logits in
        # _build_block_writes_map. Running the generic mask-token sampler first
        # duplicates argmax/softmax work and its accepted-id output is unused.
        del top_p, top_k, margin_confidence, neg_entropy
        with record_function("diffulex.sampler.dmax.fetch_attn_metadata"):
            attn_metadata = self.fetch_attn_metadata()
        with record_function("diffulex.sampler.dmax.split_logits"):
            split_logits = self._split_logits_per_req(attn_metadata, reqs, logits)

        with record_function("diffulex.sampler.dmax.init_output"):
            empty_per_req = {str(req.req_id): {} for req in reqs}
            sample_output = self.output_cls(
                true_local_ids_map=dict(empty_per_req),
                accepted_ids_map=dict(empty_per_req),
                sampled_tokens_map=dict(empty_per_req),
                mask_token_rel_ids_map=dict(empty_per_req),
                confidence_map=dict(empty_per_req),
                initial_confidence_map=dict(empty_per_req),
            )
        with record_function("diffulex.sampler.dmax.postprocess"):
            return self._postprocess_sample_output(
                reqs=reqs,
                split_logits=split_logits,
                temperatures=temperatures,
                sample_output=sample_output,
                attn_metadata=attn_metadata,
                **kwargs,
            )

    @staticmethod
    def _sample_argmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature > 0:
            gumbel = -torch.log(-torch.log(torch.rand_like(logits, dtype=torch.float32).clamp_(1e-6, 1 - 1e-6)))
            logits = logits.to(torch.float32) + gumbel * temperature
        return torch.argmax(logits, dim=-1)

    def _extract_block_logits(self, req, req_logits: torch.Tensor, block, is_prefill: bool) -> torch.Tensor | None:
        if req_logits.shape[0] == 0:
            return None
        if is_prefill:
            prefix_offset = int(req.contiguous_in_cache_prefix_len)
            local_start = int(block.start - prefix_offset)
            local_end = int(block.end - prefix_offset)
            if local_start < 0 or local_end > req_logits.shape[0]:
                return None
            return req_logits[local_start:local_end, ...]

        buf_offset = int(block.start - req.dllm_block_buffer.first_running_block.start)
        local_start = buf_offset
        local_end = buf_offset + int(block.block_size)
        if local_start < 0 or local_end > req_logits.shape[0]:
            return None
        return req_logits[local_start:local_end, ...]

    def _build_dmax_block_outputs(
        self,
        block,
        block_tokens: torch.Tensor,
        block_logits: torch.Tensor,
        temperature: float,
    ) -> tuple[dict[int, int], dict[int, dict | None], dict]:
        editable_start = int(getattr(block, "editable_start", 0) or 0)
        if editable_start >= int(block.block_size):
            return {}, {}, {
                "committable": True,
                "same_as_previous": True,
                "same_token_ratio": 1.0,
                "all_confident": True,
            }

        mask_id = int(block.mask_token_id)
        accept_threshold = float(block.thresholds.accept_threshold)
        with record_function("diffulex.sampler.dmax.block.setup_cpu_state"):
            full_block_before_cpu = [int(token) for token in block.token_ids]
            mask_positions_cpu = [idx for idx, token in enumerate(full_block_before_cpu) if token == mask_id]
        with record_function("diffulex.sampler.dmax.block.argmax"):
            top1_tokens = self._sample_argmax(block_logits, temperature)
        if self._fast_prob_path:
            # Fast path: only compute exact confidence for mask positions.
            with record_function("diffulex.sampler.dmax.block.confidence_fast_init"):
                top1_confidence = torch.ones(top1_tokens.shape, dtype=torch.float32, device=top1_tokens.device)
            if mask_positions_cpu:
                with record_function("diffulex.sampler.dmax.block.confidence_fast_mask"):
                    mask_positions = torch.tensor(mask_positions_cpu, dtype=torch.long, device=top1_tokens.device)
                    mask_logits_fp32 = block_logits.index_select(0, mask_positions).to(torch.float32)
                    mask_top1 = top1_tokens.index_select(0, mask_positions).unsqueeze(-1)
                    mask_top1_logits = mask_logits_fp32.gather(dim=-1, index=mask_top1).squeeze(-1)
                    mask_lse = torch.logsumexp(mask_logits_fp32, dim=-1)
                    top1_confidence[mask_positions] = torch.exp(mask_top1_logits - mask_lse)
        else:
            with record_function("diffulex.sampler.dmax.block.confidence_full"):
                logits_fp32 = block_logits.to(torch.float32)
                top1_logits = logits_fp32.gather(dim=-1, index=top1_tokens.unsqueeze(-1)).squeeze(-1)
                logsumexp = torch.logsumexp(logits_fp32, dim=-1)
                top1_confidence = torch.exp(top1_logits - logsumexp)

        with record_function("diffulex.sampler.dmax.block.cpu_materialize_top1"):
            top1_tokens_cpu = [int(token) for token in top1_tokens.detach().to("cpu").tolist()]
            top1_confidence_cpu = [float(conf) for conf in top1_confidence.detach().to("cpu").tolist()]

        with record_function("diffulex.sampler.dmax.block.build_writes_cpu"):
            target_tokens_cpu = list(full_block_before_cpu)
            for rel_idx, token in enumerate(full_block_before_cpu):
                if token != mask_id:
                    target_tokens_cpu[rel_idx] = top1_tokens_cpu[rel_idx]

            if mask_positions_cpu:
                decode_upto = len(mask_positions_cpu)
                for mask_offset, rel_idx in enumerate(mask_positions_cpu):
                    if top1_confidence_cpu[rel_idx] < accept_threshold:
                        decode_upto = 1 if mask_offset == 0 else mask_offset
                        break
                for rel_idx in mask_positions_cpu[:decode_upto]:
                    target_tokens_cpu[rel_idx] = top1_tokens_cpu[rel_idx]

            block_writes: dict[int, int] = {}
            token_merge_entries: dict[int, dict | None] = {}
            changed_positions = [
                rel_idx
                for rel_idx, (target_token, old_token) in enumerate(zip(target_tokens_cpu, full_block_before_cpu))
                if target_token != old_token
            ]
            same_as_previous = not changed_positions
            comparable_positions = [
                rel_idx
                for rel_idx, (old_token, target_token) in enumerate(zip(full_block_before_cpu, target_tokens_cpu))
                if old_token != mask_id and target_token != mask_id
            ]
            if comparable_positions:
                same_count = sum(
                    1 for rel_idx in comparable_positions if target_tokens_cpu[rel_idx] == full_block_before_cpu[rel_idx]
                )
                same_token_ratio = float(same_count / len(comparable_positions))
            else:
                same_token_ratio = 1.0
            all_confident = all(confidence >= 0.9 for confidence in top1_confidence_cpu)
            for rel_idx in changed_positions:
                if rel_idx < editable_start:
                    continue
                token = target_tokens_cpu[rel_idx]
                if token != full_block_before_cpu[rel_idx]:
                    block_writes[rel_idx] = token

        if self._enable_token_merge:
            with record_function("diffulex.sampler.dmax.block.token_merge_descriptors"):
                non_mask_positions = [
                    rel_idx for rel_idx, token in enumerate(target_tokens_cpu) if token != mask_id
                ]
                for rel_idx in non_mask_positions:
                    if rel_idx < editable_start:
                        continue
                    token = target_tokens_cpu[rel_idx]
                    if self._token_merge_mode == "dmax_topk":
                        descriptor = self._build_manual_token_merge_descriptor(
                            token=token,
                            confidence=top1_confidence_cpu[rel_idx],
                            mask_id=mask_id,
                        )
                    else:
                        row_probs = F.softmax(block_logits[rel_idx].to(torch.float32), dim=-1)
                        descriptor = self._build_token_merge_descriptor(
                            probs=row_probs,
                            token=token,
                            mask_id=mask_id,
                        )
                    token_merge_entries[rel_idx] = descriptor

        return block_writes, token_merge_entries, {
            "committable": bool(same_as_previous or all_confident),
            "same_as_previous": bool(same_as_previous),
            "same_token_ratio": same_token_ratio,
            "all_confident": bool(all_confident),
        }

    def _reset_block_state_map(self) -> None:
        self._last_block_state_map = {}

    def _build_block_writes_map(
        self,
        reqs,
        split_logits,
        temperatures: torch.Tensor,
        sample_output,
        attn_metadata,
        **kwargs,
    ) -> dict[str, dict[str, dict[int, int]]]:
        del sample_output, kwargs
        block_writes_map: dict[str, dict[str, dict[int, int]]] = {}
        with record_function("diffulex.sampler.dmax.reset_maps"):
            self._reset_token_merge_map()
            self._reset_block_state_map()
        with record_function("diffulex.sampler.dmax.build_block_writes_map.loop"):
            for req_idx, (req, req_logits) in enumerate(zip(reqs, split_logits)):
                req_id_str = str(req.req_id)
                req_block_writes: dict[str, dict[int, int]] = {}
                req_token_merge: dict[int, dict | None] = {}
                req_block_states: dict[str, dict] = {}
                for block_id, block in enumerate(req.dllm_blocks):
                    if not block.is_active:
                        continue
                    req_block_states[str(block_id)] = {
                        "committable": False,
                        "same_as_previous": False,
                        "same_token_ratio": 0.0,
                        "all_confident": False,
                    }
                    with record_function("diffulex.sampler.dmax.extract_block_logits"):
                        block_logits = self._extract_block_logits(req, req_logits, block, attn_metadata.is_prefill[req_idx])
                    if block_logits is None or block_logits.shape[0] != int(block.block_size):
                        continue
                    with record_function("diffulex.sampler.dmax.init_token_merge_slots"):
                        for rel_idx in range(int(block.block_size)):
                            req_token_merge.setdefault(int(block.start + rel_idx), None)
                    with record_function("diffulex.sampler.dmax.block_tokens_tensor"):
                        block_tokens = torch.tensor(block.token_ids, dtype=torch.long, device=block_logits.device)
                    with record_function("diffulex.sampler.dmax.build_block_outputs"):
                        block_writes, token_merge_entries, block_state = self._build_dmax_block_outputs(
                            block=block,
                            block_tokens=block_tokens,
                            block_logits=block_logits,
                            temperature=_temperature_value(temperatures, req_idx),
                        )
                    req_block_states[str(block_id)] = block_state
                    if block_writes:
                        req_block_writes[str(block_id)] = block_writes
                    with record_function("diffulex.sampler.dmax.merge_token_entries"):
                        for rel_idx, descriptor in token_merge_entries.items():
                            req_token_merge[int(block.start + rel_idx)] = descriptor
                block_writes_map[req_id_str] = req_block_writes
                self._set_token_merge_entries(req_id_str, req_token_merge)
                self._last_block_state_map[req_id_str] = req_block_states
        return block_writes_map

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
        sample_output.block_state_map = self._last_block_state_map
        return sample_output


def build_llada2_sampler(config=None):
    sampling_mode = str(getattr(config, "sampling_mode", "naive"))
    if getattr(config, "decoding_strategy", None) == "dmax":
        return LLaDA2DMaxSampler(config)
    if sampling_mode == "edit" and getattr(config, "model_name", None) in [
        "llada2",
        "llada2_moe",
        "llada2_mini",
        "llada2dot1_mini",
    ]:
        return LLaDA2dot1Sampler(config)
    if sampling_mode == "naive" and getattr(config, "model_name", None) in ["llada2", "llada2_moe", "llada2_mini"]:
        if (
            bool(getattr(config, "enable_vectorized_sampler", False))
            and getattr(config, "decoding_strategy", None) == "multi_bd"
        ):
            return LLaDA2VectorizedSampler(config)
        return LLaDA2Sampler(config)


AutoSampler.register("llada2", build_llada2_sampler, use_full_config=True)
AutoSampler.register("llada2_moe", build_llada2_sampler, use_full_config=True)
AutoSampler.register("llada2_mini", build_llada2_sampler, use_full_config=True)
AutoSampler.register("llada2dot1_mini", build_llada2_sampler, use_full_config=True)
AutoSampler.register("llada2_mini_dmax", build_llada2_sampler, use_full_config=True)
