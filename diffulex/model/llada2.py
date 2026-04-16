from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffulex.attention import Attention
from diffulex.layer.activation import SiluAndMul
from diffulex.layer.embed_head import ParallelLMHead, VocabParallelEmbedding
from diffulex.layer.layernorm import RMSNorm
from diffulex.layer.linear import ColumnParallelLinear, RowParallelLinear, divide
from diffulex.model.auto_model import AutoModelForDiffusionLM
from diffulex.moe.config import get_norm_topk_prob, is_moe_layer
from diffulex.moe.layer.base import FusedMoE
from diffulex.moe.layer.ep_impl import EPFusedMoE
from diffulex.moe.layer.tp_impl import TPFusedMoE
from diffulex.moe.layer.trivial_impl import TrivialFusedMoE
from diffulex.moe.topk.datatype import TopKOutput
from diffulex.utils.parallelism import get_tp_rank, get_tp_world_size, is_ep_enabled, is_tp_enabled


class LLaDA2PartialRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        if rotary_dim <= 0 or rotary_dim > head_size or rotary_dim % 2 != 0:
            raise ValueError(f"Invalid rotary_dim={rotary_dim} for head_size={head_size}.")
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        positions = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        self.register_buffer("cos_cache", torch.cat((freqs.cos(), freqs.cos()), dim=-1), persistent=False)
        self.register_buffer("sin_cache", torch.cat((freqs.sin(), freqs.sin()), dim=-1), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.view(positions.shape[0], -1, self.head_size)
        x_rot, x_pass = x[..., : self.rotary_dim], x[..., self.rotary_dim :]
        cos = self.cos_cache[positions].unsqueeze(1).to(dtype=x.dtype)
        sin = self.sin_cache[positions].unsqueeze(1).to(dtype=x.dtype)
        x_rot = (x_rot * cos) + (self._rotate_half(x_rot) * sin)
        return torch.cat((x_rot, x_pass), dim=-1).view(original_shape)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._apply(query, positions), self._apply(key, positions)


class LLaDA2QKVParallelLinear(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        *,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.tp_size = get_tp_world_size()
        self.tp_rank = get_tp_rank()
        self.num_heads = divide(total_num_heads, self.tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, self.tp_size)
        self.q_size = self.num_heads * head_size
        self.kv_size = self.num_kv_heads * head_size
        self.weight = nn.Parameter(torch.empty(self.q_size + 2 * self.kv_size, hidden_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.q_size + 2 * self.kv_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def _local_qkv(self, loaded_weight: torch.Tensor) -> torch.Tensor:
        q_total = self.total_num_heads * self.head_size
        kv_total = self.total_num_kv_heads * self.head_size
        q, k, v = loaded_weight.split((q_total, kv_total, kv_total), dim=0)
        q = q.chunk(self.tp_size, dim=0)[self.tp_rank]
        k = k.chunk(self.tp_size, dim=0)[self.tp_rank]
        v = v.chunk(self.tp_size, dim=0)[self.tp_rank]
        return torch.cat((q, k, v), dim=0).contiguous()

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(self._local_qkv(loaded_weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class LLaDA2Attention(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        del layer_idx
        tp_size = get_tp_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.num_heads = divide(self.total_num_heads, tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.query_key_value = LLaDA2QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bool(getattr(config, "use_qkv_bias", False)),
        )
        self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=bool(getattr(config, "use_bias", False)),
        )
        partial_rotary_factor = float(getattr(config, "partial_rotary_factor", 1.0))
        rotary_dim = int(self.head_dim * partial_rotary_factor)
        rotary_dim = int(getattr(config, "rotary_dim", rotary_dim) or rotary_dim)
        self.rotary_emb = LLaDA2PartialRotaryEmbedding(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 10000),
        )
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qkv = self.query_key_value(hidden_states)
        q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        q = self.query_layernorm(q.view(-1, self.num_heads, self.head_dim)).view(q.shape)
        k = self.key_layernorm(k.view(-1, self.num_kv_heads, self.head_dim)).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        return self.dense(self.attn(q, k, v, mask))


class LLaDA2DenseMLP(nn.Module):
    def __init__(self, config, intermediate_size: int | None = None) -> None:
        super().__init__()
        intermediate_size = int(intermediate_size or config.intermediate_size)
        self.gate_proj = ColumnParallelLinear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = ColumnParallelLinear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, config.hidden_size, bias=False)
        if getattr(config, "hidden_act", "silu") != "silu":
            raise NotImplementedError("LLaDA2 dense MLP currently supports only silu.")
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(torch.cat((self.gate_proj(x), self.up_proj(x)), dim=-1)))


class LLaDA2GroupLimitedRouter(nn.Module):
    def __init__(
        self,
        *,
        top_k: int,
        num_experts: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float,
        renormalize: bool,
        expert_bias_getter,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.renormalize = renormalize
        self._expert_bias_getter = expert_bias_getter

    def _group_limited_topk(self, scores: torch.Tensor) -> torch.Tensor:
        if (
            self.n_group <= 0
            or self.topk_group <= 0
            or self.n_group > self.num_experts
            or self.num_experts % self.n_group != 0
        ):
            return torch.topk(scores, k=self.top_k, dim=-1, sorted=False).indices
        experts_per_group = self.num_experts // self.n_group
        group_scores = scores.view(scores.shape[0], self.n_group, experts_per_group).topk(
            min(2, experts_per_group),
            dim=-1,
        ).values.sum(dim=-1)
        group_idx = torch.topk(group_scores, k=min(self.topk_group, self.n_group), dim=-1, sorted=False).indices
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, group_idx, True)
        score_mask = group_mask.unsqueeze(-1).expand(-1, -1, experts_per_group).reshape(scores.shape)
        masked_scores = scores.masked_fill(~score_mask, float("-inf"))
        return torch.topk(masked_scores, k=self.top_k, dim=-1, sorted=False).indices

    def forward(self, router_logits: torch.Tensor) -> TopKOutput:
        scores = torch.sigmoid(router_logits.float()).to(router_logits.dtype)
        expert_bias = self._expert_bias_getter().to(scores.device, dtype=scores.dtype)
        topk_ids = self._group_limited_topk(scores + expert_bias)
        topk_weights = torch.gather(scores, dim=-1, index=topk_ids)
        if self.renormalize and self.top_k > 1:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return TopKOutput(weights=topk_weights, ids=topk_ids, router_logits=router_logits)


class LLaDA2MoEMixin:
    def _init_llada2_moe(self, config) -> None:
        self.gate.register_buffer("expert_bias", torch.zeros((self.num_experts,), dtype=torch.float32))
        self.router = LLaDA2GroupLimitedRouter(
            top_k=self.top_k,
            num_experts=self.num_experts,
            n_group=int(getattr(config, "n_group", 0) or 0),
            topk_group=int(getattr(config, "topk_group", 0) or 0),
            routed_scaling_factor=float(getattr(config, "routed_scaling_factor", 1.0)),
            renormalize=get_norm_topk_prob(config),
            expert_bias_getter=lambda: self.gate.expert_bias,
        )
        num_shared_experts = int(getattr(config, "num_shared_experts", 0) or 0)
        self.shared_experts = (
            LLaDA2DenseMLP(config, intermediate_size=self.intermediate_size * num_shared_experts)
            if num_shared_experts > 0
            else None
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        routed_states, router_logits = super().forward(hidden_states)
        if self.shared_experts is not None:
            routed_states = routed_states + self.shared_experts(hidden_states)
        return routed_states, router_logits


class LLaDA2TrivialMoE(LLaDA2MoEMixin, TrivialFusedMoE):
    @classmethod
    def from_config(cls, config) -> "LLaDA2TrivialMoE":
        module = cls(
            hidden_size=config.hidden_size,
            intermediate_size=int(config.moe_intermediate_size),
            num_experts=int(config.num_experts),
            top_k=int(config.num_experts_per_tok),
            hidden_act=getattr(config, "hidden_act", "silu"),
            norm_topk_prob=get_norm_topk_prob(config),
        )
        module._init_llada2_moe(config)
        return module


class LLaDA2TPMoE(LLaDA2MoEMixin, TPFusedMoE):
    @classmethod
    def from_config(cls, config) -> "LLaDA2TPMoE":
        module = cls(
            hidden_size=config.hidden_size,
            intermediate_size=int(config.moe_intermediate_size),
            num_experts=int(config.num_experts),
            top_k=int(config.num_experts_per_tok),
            hidden_act=getattr(config, "hidden_act", "silu"),
            norm_topk_prob=get_norm_topk_prob(config),
        )
        module._init_llada2_moe(config)
        return module


class LLaDA2EPMoE(LLaDA2MoEMixin, EPFusedMoE):
    @classmethod
    def from_config(cls, config) -> "LLaDA2EPMoE":
        module = cls(
            hidden_size=config.hidden_size,
            intermediate_size=int(config.moe_intermediate_size),
            num_experts=int(config.num_experts),
            top_k=int(config.num_experts_per_tok),
            hidden_act=getattr(config, "hidden_act", "silu"),
            norm_topk_prob=get_norm_topk_prob(config),
        )
        module._init_llada2_moe(config)
        return module


def build_llada2_mlp(config, layer_idx: int) -> nn.Module:
    if not is_moe_layer(config, layer_idx):
        return LLaDA2DenseMLP(config)
    if is_ep_enabled():
        return LLaDA2EPMoE.from_config(config)
    if is_tp_enabled():
        return LLaDA2TPMoE.from_config(config)
    return LLaDA2TrivialMoE.from_config(config)


class LLaDA2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.attention = LLaDA2Attention(config, layer_idx)
        self.mlp = build_llada2_mlp(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.attention(positions, hidden_states, mask)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = mlp_output[0] if isinstance(mlp_output, tuple) else mlp_output
        return residual + hidden_states


class LLaDA2Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.word_embeddings = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LLaDA2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.word_embeddings(input_ids)
        hidden_states = self._maybe_apply_token_merging(hidden_states)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, mask)
        return self.norm(hidden_states)

    def _maybe_apply_token_merging(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from diffulex.attention import fetch_attn_metadata

        attn_metadata = fetch_attn_metadata()

        if not attn_metadata.token_merge_enabled:
            return hidden_states

        merge_mask = attn_metadata.token_merge_mask
        topk_ids = attn_metadata.token_merge_topk_ids
        topk_probs = attn_metadata.token_merge_topk_probs
        residual_probs = attn_metadata.token_merge_residual_probs
        mask_token_id = attn_metadata.token_merge_mask_token_id
        merge_mode = attn_metadata.token_merge_mode
        if merge_mask is None or topk_ids is None or topk_probs is None or residual_probs is None or mask_token_id is None:
            raise RuntimeError("token_merge_enabled=True but token-merge metadata is incomplete.")
        if int(merge_mask.numel()) != int(hidden_states.shape[0]):
            raise RuntimeError(
                "Token-merge metadata length does not match hidden states: "
                f"merge_mask={merge_mask.numel()}, hidden_states={hidden_states.shape[0]}"
            )

        device = hidden_states.device
        dtype = hidden_states.dtype
        merge_mask = merge_mask.to(device=device, dtype=torch.bool)
        if not bool(merge_mask.any().item()):
            return hidden_states

        topk_ids = topk_ids.to(device=device, dtype=torch.int64)
        topk_probs = topk_probs.to(device=device, dtype=torch.float32)
        residual_probs = residual_probs.to(device=device, dtype=torch.float32)

        flat_topk_ids = topk_ids.reshape(-1)
        topk_embeds = self.word_embeddings(flat_topk_ids).view(*topk_ids.shape, hidden_states.shape[-1])
        topk_weighted = (topk_embeds.to(torch.float32) * topk_probs.unsqueeze(-1)).sum(dim=1)

        if merge_mode == "dmax_topk":
            mask_id = torch.tensor([int(mask_token_id)], dtype=torch.int64, device=device)
            mask_embed = self.word_embeddings(mask_id).to(torch.float32)
            soft_embeds = topk_weighted + mask_embed * residual_probs

            if attn_metadata.token_merge_renormalize:
                current_norm = torch.norm(soft_embeds, p=2, dim=-1, keepdim=True)
                topk_norms = torch.norm(topk_embeds.to(torch.float32), p=2, dim=-1)
                expected_topk_norm = (topk_norms * topk_probs).sum(dim=-1, keepdim=True)
                expected_mask_norm = torch.norm(mask_embed, p=2, dim=-1, keepdim=True) * residual_probs
                target_norm = expected_topk_norm + expected_mask_norm
                soft_embeds = soft_embeds * (target_norm / (current_norm + 1e-6))
        elif merge_mode == "iter_smooth_topk":
            merge_weight = float(attn_metadata.token_merge_weight)
            soft_embeds = hidden_states.to(torch.float32) + merge_weight * topk_weighted
        else:
            raise ValueError(f"Unsupported token_merge_mode: {merge_mode}")

        output = hidden_states.clone()
        output[merge_mask] = soft_embeds.to(dtype=dtype)[merge_mask]
        return output


@AutoModelForDiffusionLM.register("llada2")
@AutoModelForDiffusionLM.register("llada2_moe")
@AutoModelForDiffusionLM.register("llada2_mini")
class LLaDA2ForDiffusionLM(nn.Module):
    packed_modules_mapping = {}

    def __init__(self, config) -> None:
        super().__init__()
        self.model = LLaDA2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight.data = self.model.word_embeddings.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, mask)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


SparseMoEBlock = FusedMoE


__all__ = [
    "LLaDA2Attention",
    "LLaDA2DecoderLayer",
    "LLaDA2DenseMLP",
    "LLaDA2ForDiffusionLM",
    "LLaDA2Model",
    "LLaDA2TrivialMoE",
    "LLaDA2TPMoE",
    "LLaDA2EPMoE",
]
