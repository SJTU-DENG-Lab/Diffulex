import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists

from dataclasses import dataclass
from easydict import EasyDict as edict

from diffulex.engine.request import DllmReq
from diffulex.logger import get_logger

logger = get_logger(__name__)


class SamplerBase(nn.Module):
    def __init__(self):
        super().__init__()
        from diffulex.attention import fetch_attn_metadata

        self.fetch_attn_metadata = fetch_attn_metadata

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
    ):
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
            initial_confidence, x0 = probs.max(dim=-1)

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


@dataclass
class SampleOutputBase:
    true_local_ids_map: dict[str, dict[str, list[int]]]
    accepted_ids_map: dict[str, dict[str, list[int]]]
    sampled_tokens_map: dict[str, dict[str, list[int]]]

    def __post_init__(self):
        self.accepted_ids_map = edict(self.accepted_ids_map)
        self.sampled_tokens_map = edict(self.sampled_tokens_map)
        self.true_local_ids_map = edict(self.true_local_ids_map)


class SamplerShiftLogits(SamplerBase):
    def __init__(self):
        super().__init__()
        self.req_last_logits_map: dict[str, torch.Tensor] = {}

    def _fetch_last_logits(self, logits: torch.Tensor, req: DllmReq) -> torch.Tensor:
        req_id_str = str(req.req_id)
        if req.has_to_cache_block:
            last_logits = logits[req.to_cache_last_token_id]
            self.req_last_logits_map[req_id_str] = last_logits
            return last_logits

        if req_id_str in self.req_last_logits_map:
            return self.req_last_logits_map[req_id_str]

        last_logits = logits[-1] if logits.shape[0] > 0 else None
        if last_logits is not None:
            self.req_last_logits_map[req_id_str] = last_logits
            return last_logits

        raise ValueError(f"Cannot fetch last logits for req {req.req_id}: empty logits tensor")

    def _shift_logits(self, logits, last_logit=None):
        if logits.shape[1] == 0:
            logger.warning("Logits sequence length is 0, returning empty logits")
            raise Exception("logits sequence length is 0")

        shifted_logits = torch.zeros_like(logits)
        shifted_logits[1:, ...] = logits[:-1, ...]

        if last_logit is not None:
            shifted_logits[0, ...] = last_logit
            return shifted_logits

        shifted_logits[0, ...] = 1.0
        return shifted_logits


class SamplerNoShiftLogits(SamplerBase):
    pass


class DllmSamplerNoShiftBase(SamplerNoShiftLogits):
    output_cls = SampleOutputBase

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

        split_logits = torch.split(
            logits,
            [
                len(req.running_sequence) if attn_metadata.is_prefill[idx] else req.chunk_size
                for idx, req in enumerate(reqs)
            ],
            dim=0,
        )

        accepted_ids_map = {}
        sampled_tokens_map = {}
        true_local_ids_map = {}

        for idx, (temperature, req, req_logits) in enumerate(zip(temperatures, reqs, split_logits)):
            true_local_ids_sub_map = {}
            accepted_ids_sub_map = {}
            sampled_tokens_sub_map = {}

            for block_id, block in enumerate(req.dllm_blocks):
                if not block.is_active or (block.num_mask_tokens == 0):
                    continue

                if len(block.mask_token_global_ids) == 0:
                    continue

                if attn_metadata.is_prefill[idx]:
                    mask_token_logits = req_logits[block.mask_token_global_ids, ...]
                else:
                    mask_token_logits = req_logits[block.mask_token_relative_ids, ...]

                confidence, sampled_tokens, initial_confidence = self.sample_tokens(
                    mask_token_logits,
                    temperature,
                    top_p=top_p,
                    top_k=top_k,
                    neg_entropy=(neg_entropy == "neg_entropy"),
                    margin_confidence=(margin_confidence == "margin_confidence"),
                )
                accepted_ids = self._compute_accepted_ids(
                    block, confidence, initial_confidence, sampled_tokens, **kwargs
                )
                accepted_ids_list = accepted_ids.to(device="cpu").tolist()
                true_local_ids_sub_map[str(block_id)] = [block.mask_token_relative_ids[i] for i in accepted_ids_list]
                accepted_ids_sub_map[str(block_id)] = accepted_ids_list
                sampled_tokens_sub_map[str(block_id)] = sampled_tokens.to(device="cpu").tolist()

            req_id_str = str(req.req_id)
            true_local_ids_map[req_id_str] = true_local_ids_sub_map
            accepted_ids_map[req_id_str] = accepted_ids_sub_map
            sampled_tokens_map[req_id_str] = sampled_tokens_sub_map

        return self.output_cls(
            true_local_ids_map=true_local_ids_map,
            accepted_ids_map=accepted_ids_map,
            sampled_tokens_map=sampled_tokens_map,
        )

    def _compute_accepted_ids(
        self,
        block,
        confidence: torch.Tensor,
        initial_confidence: torch.Tensor,
        sampled_tokens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError


class DllmSamplerShiftBase(SamplerShiftLogits):
    output_cls = SampleOutputBase

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

        split_logits = torch.split(
            logits,
            [
                len(req.running_sequence) if attn_metadata.is_prefill[idx] else req.chunk_size
                for idx, req in enumerate(reqs)
            ],
            dim=0,
        )

        accepted_ids_map = {}
        sampled_tokens_map = {}
        true_local_ids_map = {}

        for idx, (temperature, req, req_logits) in enumerate(zip(temperatures, reqs, split_logits)):
            true_local_ids_sub_map = {}
            accepted_ids_sub_map = {}
            sampled_tokens_sub_map = {}
            last_logits = self._fetch_last_logits(req_logits, req)
            shifted_logits = self._shift_logits(req_logits, last_logits)

            for block_id, block in enumerate(req.dllm_blocks):
                if not block.is_active or (block.num_mask_tokens == 0):
                    continue

                if len(block.mask_token_global_ids) == 0:
                    continue

                if attn_metadata.is_prefill[idx]:
                    mask_token_logits = shifted_logits[block.mask_token_global_ids, ...]
                else:
                    mask_token_logits = shifted_logits[block.mask_token_relative_ids, ...]

                confidence, sampled_tokens, initial_confidence = self.sample_tokens(
                    mask_token_logits,
                    temperature,
                    top_p=top_p,
                    top_k=top_k,
                    neg_entropy=(neg_entropy == "neg_entropy"),
                    margin_confidence=(margin_confidence == "margin_confidence"),
                )
                accepted_ids = self._compute_accepted_ids(
                    block, confidence, initial_confidence, sampled_tokens, **kwargs
                )
                accepted_ids_list = accepted_ids.to(device="cpu").tolist()
                true_local_ids_sub_map[str(block_id)] = [block.mask_token_relative_ids[i] for i in accepted_ids_list]
                accepted_ids_sub_map[str(block_id)] = accepted_ids_list
                sampled_tokens_sub_map[str(block_id)] = sampled_tokens.to(device="cpu").tolist()

            req_id_str = str(req.req_id)
            true_local_ids_map[req_id_str] = true_local_ids_sub_map
            accepted_ids_map[req_id_str] = accepted_ids_sub_map
            sampled_tokens_map[req_id_str] = sampled_tokens_sub_map

        return self.output_cls(
            true_local_ids_map=true_local_ids_map,
            accepted_ids_map=accepted_ids_map,
            sampled_tokens_map=sampled_tokens_map,
        )

    def _compute_accepted_ids(
        self,
        block,
        confidence: torch.Tensor,
        initial_confidence: torch.Tensor,
        sampled_tokens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError
