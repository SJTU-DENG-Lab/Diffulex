from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

from diffulex.attention import Attention
from diffulex.layer.activation import SiluAndMul
from diffulex.layer.embed_head import ParallelLMHead, VocabParallelEmbedding
from diffulex.layer.layernorm import RMSNorm
from diffulex.layer.linear import ColumnParallelLinear, RowParallelLinear, divide
from diffulex.layer.rotary_embedding import get_rope
from diffulex.model.auto_model import AutoModelForDiffusionLM
from diffulex.moe.config import get_norm_topk_prob, is_moe_layer
from diffulex.moe.layer.base import FusedMoE
from diffulex.moe.layer.ep_impl import EPFusedMoE
from diffulex.moe.layer.tp_impl import TPFusedMoE
from diffulex.moe.layer.naive_impl import NaiveFusedMoE
from diffulex.moe.topk import GroupLimitedTopKRouter
from diffulex.distributed.parallel_state import fetch_parallel_state
from diffulex.utils.checkpoint import LoadContext, ResolvedWeight


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
        parallel_state = fetch_parallel_state()
        self.tp_size = parallel_state.get_tp_world_size()
        self.tp_rank = parallel_state.get_tp_rank()
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
        self.layer_idx = layer_idx
        parallel_state = fetch_parallel_state()
        tp_size = parallel_state.get_tp_world_size()
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
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
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
        q = rearrange(
            self.query_layernorm(
                rearrange(q, "token (head head_dim) -> token head head_dim", head=self.num_heads)
            ),
            "token head head_dim -> token (head head_dim)",
        )
        k = rearrange(
            self.key_layernorm(
                rearrange(k, "token (head head_dim) -> token head head_dim", head=self.num_kv_heads)
            ),
            "token head head_dim -> token (head head_dim)",
        )
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


class LLaDA2MoEMixin:
    def _init_llada2_moe(self, config) -> None:
        self.gate.register_buffer("expert_bias", torch.zeros((self.num_experts,), dtype=torch.float32))
        self.gate.forward = self._forward_llada2_gate
        self.router = GroupLimitedTopKRouter(
            top_k=self.top_k,
            num_experts=self.num_experts,
            n_group=int(getattr(config, "n_group", 0) or 0),
            topk_group=int(getattr(config, "topk_group", 0) or 0),
            routed_scaling_factor=float(getattr(config, "routed_scaling_factor", 1.0)),
            renormalize=get_norm_topk_prob(config),
            expert_bias_getter=lambda: self.gate.expert_bias,
        )

    def _forward_llada2_gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states.to(torch.float32), self.gate.weight.to(torch.float32))

    def resolve_checkpoint_weight(self, suffix: str, ctx: LoadContext) -> ResolvedWeight | None:
        if suffix == "gate.e_score_correction_bias":
            return ResolvedWeight(buffer=self.gate.expert_bias)

        return super().resolve_checkpoint_weight(suffix, ctx)


class LLaDA2NaiveMoE(LLaDA2MoEMixin, NaiveFusedMoE):
    @classmethod
    def from_config(cls, config) -> "LLaDA2NaiveMoE":
        num_shared_experts = int(getattr(config, "num_shared_experts", 0) or 0)
        module = cls(
            hidden_size=config.hidden_size,
            intermediate_size=int(config.moe_intermediate_size),
            num_experts=int(config.num_experts),
            top_k=int(config.num_experts_per_tok),
            hidden_act=getattr(config, "hidden_act", "silu"),
            norm_topk_prob=get_norm_topk_prob(config),
            num_shared_experts=num_shared_experts,
            shared_expert_intermediate_size=int(config.moe_intermediate_size) * num_shared_experts,
        )
        module._init_llada2_moe(config)
        return module


class LLaDA2TPMoE(LLaDA2MoEMixin, TPFusedMoE):
    @classmethod
    def from_config(cls, config) -> "LLaDA2TPMoE":
        num_shared_experts = int(getattr(config, "num_shared_experts", 0) or 0)
        module = cls(
            hidden_size=config.hidden_size,
            intermediate_size=int(config.moe_intermediate_size),
            num_experts=int(config.num_experts),
            top_k=int(config.num_experts_per_tok),
            hidden_act=getattr(config, "hidden_act", "silu"),
            norm_topk_prob=get_norm_topk_prob(config),
            num_shared_experts=num_shared_experts,
            shared_expert_intermediate_size=int(config.moe_intermediate_size) * num_shared_experts,
        )
        module._init_llada2_moe(config)
        return module


class LLaDA2EPMoE(LLaDA2MoEMixin, EPFusedMoE):
    @classmethod
    def from_config(cls, config) -> "LLaDA2EPMoE":
        num_shared_experts = int(getattr(config, "num_shared_experts", 0) or 0)
        module = cls(
            hidden_size=config.hidden_size,
            intermediate_size=int(config.moe_intermediate_size),
            num_experts=int(config.num_experts),
            top_k=int(config.num_experts_per_tok),
            hidden_act=getattr(config, "hidden_act", "silu"),
            norm_topk_prob=get_norm_topk_prob(config),
            dispatcher_backend=getattr(config, "moe_dispatcher_backend", "naive"),
            deepep_mode=getattr(config, "deepep_mode", "auto"),
            deepep_num_max_dispatch_tokens_per_rank=getattr(
                config,
                "deepep_num_max_dispatch_tokens_per_rank",
                256,
            ),
            num_shared_experts=num_shared_experts,
            shared_expert_intermediate_size=int(config.moe_intermediate_size) * num_shared_experts,
        )
        module._init_llada2_moe(config)
        return module


def build_llada2_mlp(config, layer_idx: int) -> nn.Module:
    if not is_moe_layer(config, layer_idx):
        return LLaDA2DenseMLP(config)
    parallel_state = fetch_parallel_state()
    if parallel_state.is_ep_enabled():
        module = LLaDA2EPMoE.from_config(config)
    elif parallel_state.is_tp_enabled():
        module = LLaDA2TPMoE.from_config(config)
    else:
        module = LLaDA2NaiveMoE.from_config(config)
    module._llada2_layer_idx = layer_idx
    return module


class LLaDA2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.attention = LLaDA2Attention(config, layer_idx)
        self.mlp = build_llada2_mlp(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

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

        flat_topk_ids = rearrange(topk_ids, "token topk -> (token topk)")
        topk_embeds = rearrange(
            self.word_embeddings(flat_topk_ids),
            "(token topk) hidden -> token topk hidden",
            token=topk_ids.shape[0],
            topk=topk_ids.shape[1],
        )
        topk_weighted = reduce(
            topk_embeds.to(torch.float32) * rearrange(topk_probs, "token topk -> token topk 1"),
            "token topk hidden -> token hidden",
            "sum",
        )

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


def build_llada2_runtime_config(config):
    runtime_config = copy.copy(config.hf_config)
    for name in (
        "moe_dispatcher_backend",
        "deepep_mode",
        "deepep_num_max_dispatch_tokens_per_rank",
        "expert_parallel_size",
        "tensor_parallel_size",
        "data_parallel_size",
    ):
        setattr(runtime_config, name, getattr(config, name))
    return runtime_config


@AutoModelForDiffusionLM.register("llada2", use_full_config=True)
@AutoModelForDiffusionLM.register("llada2_moe", use_full_config=True)
@AutoModelForDiffusionLM.register("llada2_mini", use_full_config=True)
class LLaDA2ForDiffusionLM(nn.Module):
    packed_modules_mapping = {}

    def __init__(self, config) -> None:
        super().__init__()
        runtime_config = build_llada2_runtime_config(config)
        self.model = LLaDA2Model(runtime_config)
        self.lm_head = ParallelLMHead(runtime_config.vocab_size, runtime_config.hidden_size)
        if getattr(runtime_config, "tie_word_embeddings", False):
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
    "LLaDA2NaiveMoE",
    "LLaDA2TPMoE",
    "LLaDA2EPMoE",
]
