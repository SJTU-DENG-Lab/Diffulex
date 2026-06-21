from __future__ import annotations

import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

from diffulex.attention import Attention
from diffulex.layer.activation import SiluAndMul
from diffulex.layer.embed_head import ParallelLMHead, VocabParallelEmbedding
from diffulex.layer.layernorm import RMSNorm
from diffulex.layer.linear import ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear, divide
from diffulex.layer.rotary_embedding import get_rope
from diffulex.model.auto_model import AutoModelForDiffusionLM
from diffulex.moe.config import get_norm_topk_prob, is_moe_layer
from diffulex.moe.layer.base import FusedMoE
from diffulex.moe.layer.ep_impl import EPFusedMoE
from diffulex.moe.layer.tp_impl import TPFusedMoE
from diffulex.moe.layer.naive_impl import NaiveFusedMoE
from diffulex.moe.topk import GroupLimitedTopKRouter
from diffulex.distributed.parallel_state import fetch_parallel_state
from diffulex.profiling import record_function
from diffulex.utils.checkpoint import LoadContext, ResolvedWeight


def _llada2_use_reference_view_path() -> bool:
    return os.getenv("DIFFULEX_LLADA2_REFERENCE_VIEW_PATH", "0") == "1"


def _llada2_use_legacy_qkv_path() -> bool:
    return os.getenv("DIFFULEX_LLADA2_LEGACY_QKV_PATH", "0") == "1"


def _llada2_gate_use_fp32() -> bool:
    # Keep fp32 gate as default for alignment; allow explicit opt-out for perf probing.
    return os.getenv("DIFFULEX_LLADA2_GATE_FP32", "1") != "0"


def _llada2_enable_fused_qk_norm_rope() -> bool:
    explicit = os.getenv("DIFFULEX_ENABLE_FUSED_QK_NORM_ROPE")
    if explicit is not None:
        return explicit != "0"
    return os.getenv("DIFFULEX_DISABLE_FUSED_QK_NORM_ROPE", "0") != "1"


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
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            attn_impl=getattr(config, "attn_impl", "triton"),
        )
        self._fused_qk_norm_rope_unavailable = False

    def _split_qkv_reference(
        self,
        qkv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = qkv.size(0)
        qkv = qkv.view(tokens, self.num_heads + 2 * self.num_kv_heads, self.head_dim)
        q = qkv[:, : self.num_heads, :]
        k = qkv[:, self.num_heads : self.num_heads + self.num_kv_heads, :]
        v = qkv[:, self.num_heads + self.num_kv_heads :, :]
        return q, k, v

    def _split_qkv_default(
        self,
        qkv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        q = rearrange(q, "token (head head_dim) -> token head head_dim", head=self.num_heads)
        k = rearrange(k, "token (head head_dim) -> token head head_dim", head=self.num_kv_heads)
        v = rearrange(v, "token (head head_dim) -> token head head_dim", head=self.num_kv_heads)
        return q, k, v

    def _try_fused_qk_norm_rope(self, qkv: torch.Tensor, positions: torch.Tensor) -> bool:
        if (
            self._fused_qk_norm_rope_unavailable
            or not _llada2_enable_fused_qk_norm_rope()
            or not qkv.is_cuda
            or not qkv.is_contiguous()
            or qkv.dim() != 2
            or positions.dim() != 1
        ):
            return False
        if (
            self.query_layernorm.weight is None
            or self.key_layernorm.weight is None
            or self.query_layernorm.eps != self.key_layernorm.eps
            or self.query_layernorm.weight.dtype != qkv.dtype
            or self.key_layernorm.weight.dtype != qkv.dtype
        ):
            return False
        inner_rope = getattr(self.rotary_emb, "rotary_emb", None)
        cos_sin_cache = None
        for rope in (self.rotary_emb, inner_rope):
            if rope is None:
                continue
            match_cache_dtype = getattr(rope, "_match_cos_sin_cache_dtype", None)
            if callable(match_cache_dtype):
                cos_sin_cache = match_cache_dtype(qkv)
                break
            cos_sin_cache = getattr(rope, "cos_sin_cache", None)
            if cos_sin_cache is not None:
                break
        if cos_sin_cache is None or not torch.is_tensor(cos_sin_cache):
            return False
        rotary_dim = int(cos_sin_cache.shape[-1])
        if cos_sin_cache.dim() != 2 or rotary_dim <= 0 or rotary_dim > self.head_dim or rotary_dim % 2 != 0:
            return False
        if cos_sin_cache.device != qkv.device:
            return False
        try:
            from vllm import _custom_ops as ops

            ops.fused_qk_norm_rope(
                qkv,
                self.num_heads,
                self.num_kv_heads,
                self.num_kv_heads,
                self.head_dim,
                self.query_layernorm.eps,
                self.query_layernorm.weight,
                self.key_layernorm.weight,
                cos_sin_cache,
                True,
                positions,
            )
            return True
        except Exception:
            self._fused_qk_norm_rope_unavailable = True
            return False

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layer_prefix = f"diffulex.llada2.layer{self.layer_idx}.attention"
        with record_function(f"{layer_prefix}.qkv_proj"):
            qkv = self.query_key_value(hidden_states)
        if _llada2_use_legacy_qkv_path():
            with record_function(f"{layer_prefix}.split_qkv"):
                q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
            with record_function(f"{layer_prefix}.qk_norm"):
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
            with record_function(f"{layer_prefix}.rope"):
                q, k = self.rotary_emb(positions, q, k)
            with record_function(f"{layer_prefix}.attn_core"):
                attn_out = self.attn(q, k, v, mask)
            with record_function(f"{layer_prefix}.out_proj"):
                return self.dense(attn_out)
        with record_function(f"{layer_prefix}.split_qkv"):
            if _llada2_use_reference_view_path():
                q, k, v = self._split_qkv_reference(qkv)
            else:
                q, k, v = self._split_qkv_default(qkv)
        with record_function(f"{layer_prefix}.qk_norm"):
            fused_qk_norm_rope = self._try_fused_qk_norm_rope(qkv, positions)
            if fused_qk_norm_rope:
                q, k, v = self._split_qkv_default(qkv)
            else:
                q = self.query_layernorm(q)
                k = self.key_layernorm(k)
        if not fused_qk_norm_rope:
            with record_function(f"{layer_prefix}.rope"):
                q, k = self.rotary_emb(positions, q, k)
        with record_function(f"{layer_prefix}.attn_core"):
            attn_out = self.attn(q, k, v, mask)
        with record_function(f"{layer_prefix}.out_proj"):
            attn_out = self.dense(attn_out)
        return attn_out


class LLaDA2DenseMLP(nn.Module):
    def __init__(self, config, intermediate_size: int | None = None) -> None:
        super().__init__()
        intermediate_size = int(intermediate_size or config.intermediate_size)
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
        )
        self.down_proj = RowParallelLinear(intermediate_size, config.hidden_size, bias=False)
        if getattr(config, "hidden_act", "silu") != "silu":
            raise NotImplementedError("LLaDA2 dense MLP currently supports only silu.")
        self.act_fn = SiluAndMul()
        self._register_state_dict_hook(self._add_state_dict_aliases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))

    def _add_state_dict_aliases(self, module, state_dict, prefix, local_metadata) -> None:
        gate_weight, up_weight = self.gate_up_proj.weight.split(self.gate_up_proj.weight.shape[0] // 2, dim=0)
        state_dict[prefix + "gate_proj.weight"] = gate_weight
        state_dict[prefix + "up_proj.weight"] = up_weight

    def resolve_checkpoint_weight(self, suffix: str, ctx: LoadContext) -> ResolvedWeight | None:
        if suffix == "gate_proj.weight":
            return ResolvedWeight(param=self.gate_up_proj.weight, shard_id=0)
        if suffix == "up_proj.weight":
            return ResolvedWeight(param=self.gate_up_proj.weight, shard_id=1)
        return None


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
        if _llada2_gate_use_fp32():
            logits = F.linear(hidden_states.to(torch.float32), self.gate.weight.to(torch.float32))
        else:
            logits = F.linear(hidden_states, self.gate.weight)
        if os.getenv("DIFFULEX_DISABLE_EXPERT_BIAS", "0") == "1":
            # Debug toggle: keep checkpoint loading intact but ignore expert bias at runtime.
            self.gate.expert_bias.zero_()
        return logits

    def resolve_checkpoint_weight(self, suffix: str, ctx: LoadContext) -> ResolvedWeight | None:
        if suffix in {"gate.e_score_correction_bias", "gate.expert_bias"}:
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
            moe_gemm_impl=getattr(config, "moe_gemm_impl", "triton"),
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
            moe_gemm_impl=getattr(config, "moe_gemm_impl", "triton"),
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
            moe_gemm_impl=getattr(config, "moe_gemm_impl", "triton"),
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
        residual: torch.Tensor | None = None,
        *,
        return_residual: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        layer_prefix = f"diffulex.llada2.layer{self.layer_idx}"
        with record_function(f"{layer_prefix}.input_norm"):
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

        with record_function(f"{layer_prefix}.attention_total"):
            hidden_states = self.attention(positions, hidden_states, mask)
        with record_function(f"{layer_prefix}.post_attention_norm"):
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        with record_function(f"{layer_prefix}.mlp"):
            mlp_output = self.mlp(hidden_states)
        if isinstance(mlp_output, tuple):
            hidden_states, _router_logits = mlp_output
        else:
            hidden_states = mlp_output
        if return_residual:
            return hidden_states, residual
        return residual + hidden_states


class LLaDA2Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.word_embeddings = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.mask_token_id = int(getattr(config, "mask_token_id", -1))
        if self.mask_token_id >= 0:
            self.register_buffer(
                "mask_token_id_tensor",
                torch.tensor([self.mask_token_id], dtype=torch.int64),
                persistent=False,
            )
        else:
            self.register_buffer("mask_token_id_tensor", None, persistent=False)
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
        hidden_states = self._maybe_apply_token_merge(hidden_states)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                mask,
                residual,
                return_residual=True,
            )
        if residual is None:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _residual = self.norm(hidden_states, residual)
        return hidden_states

    def _maybe_apply_token_merge(self, hidden_states: torch.Tensor) -> torch.Tensor:
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
        merge_dtype = hidden_states.dtype
        topk_embeds_merge = topk_embeds.to(dtype=merge_dtype)
        topk_probs_merge = topk_probs.to(dtype=merge_dtype)
        residual_probs_merge = residual_probs.to(dtype=merge_dtype)
        topk_weighted = reduce(
            topk_embeds_merge * rearrange(topk_probs_merge, "token topk -> token topk 1"),
            "token topk hidden -> token hidden",
            "sum",
        )

        if merge_mode == "dmax_topk":
            if self.mask_token_id_tensor is not None and self.mask_token_id == int(mask_token_id):
                mask_embed = self.word_embeddings(self.mask_token_id_tensor).to(dtype=merge_dtype)
            else:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "CUDA graph capture requires LLaDA2Model.mask_token_id_tensor to match "
                        f"attn_metadata mask token id (model={self.mask_token_id}, metadata={mask_token_id})."
                    )
                mask_id = torch.tensor([int(mask_token_id)], dtype=torch.int64, device=device)
                mask_embed = self.word_embeddings(mask_id).to(dtype=merge_dtype)
            # Match native generate_spd's embedding-dtype blend and only lift norm
            # calculations to float temporarily.
            soft_embeds = topk_weighted + mask_embed * residual_probs_merge
            
            if attn_metadata.token_merge_renormalize:
                current_norm = torch.linalg.vector_norm(
                    soft_embeds.float(), dim=-1, keepdim=True
                ).clamp_min(1e-12).to(dtype=merge_dtype)
                topk_norms = torch.linalg.vector_norm(
                    topk_embeds.float(), dim=-1
                ).to(dtype=merge_dtype)
                expected_topk_norm = (topk_norms * topk_probs_merge).sum(dim=-1, keepdim=True)
                expected_mask_norm = torch.linalg.vector_norm(
                    mask_embed.float(), dim=-1, keepdim=True
                ).to(dtype=merge_dtype) * residual_probs_merge
                target_norm = expected_topk_norm + expected_mask_norm
                soft_embeds = soft_embeds * (target_norm / current_norm)
        elif merge_mode == "iter_smooth_topk":
            merge_weight = float(attn_metadata.token_merge_weight)
            soft_embeds = hidden_states.to(torch.float32) + merge_weight * topk_weighted
        else:
            raise ValueError(f"Unsupported token_merge_mode: {merge_mode}")

        merge_mask_expanded = rearrange(merge_mask, "token -> token 1")
        return torch.where(merge_mask_expanded, soft_embeds.to(dtype=dtype), hidden_states)


def build_llada2_runtime_config(config):
    runtime_config = copy.copy(getattr(config, "hf_config", config))
    for name in (
        "moe_dispatcher_backend",
        "moe_gemm_impl",
        "deepep_mode",
        "deepep_num_max_dispatch_tokens_per_rank",
        "expert_parallel_size",
        "tensor_parallel_size",
        "data_parallel_size",
        "mask_token_id",
        "attn_impl",
    ):
        if hasattr(config, name):
            setattr(runtime_config, name, getattr(config, name))
    return runtime_config


@AutoModelForDiffusionLM.register("llada2", use_full_config=True)
@AutoModelForDiffusionLM.register("llada2_moe", use_full_config=True)
@AutoModelForDiffusionLM.register("llada2_mini", use_full_config=True)
@AutoModelForDiffusionLM.register("llada2dot1_mini", use_full_config=True)
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
