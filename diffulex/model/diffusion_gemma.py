from __future__ import annotations

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffulex.attention import Attention
from diffulex.distributed.parallel_state import fetch_parallel_state
from diffulex.layer.activation import GeluAndMul, SiluAndMul
from diffulex.layer.embed_head import ParallelLMHead, VocabParallelEmbedding
from diffulex.layer.layernorm import RMSNorm
from diffulex.layer.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from diffulex.layer.rotary_embedding import get_gemma4_proportional_rope, get_rope
from diffulex.model.auto_model import AutoModelForDiffusionLM
from diffulex.layer.linear import divide, tp_all_reduce
from diffulex.utils.checkpoint import LoadContext, ResolvedWeight
from diffulex_kernel import fused_moe


class RMSNormNoWeight(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        return (x_fp32 * torch.rsqrt(var + self.eps)).to(orig_dtype)


def _text_config(config):
    return getattr(config, "text_config", config)


def _cfg(config, *names, default=None):
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def _layer_types(config) -> list[str]:
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        return list(layer_types)
    return ["full_attention"] * int(config.num_hidden_layers)


def _hidden_activation(config) -> str:
    return str(_cfg(config, "hidden_activation", "hidden_act", default="gelu_pytorch_tanh"))


class DiffusionGemmaMLP(nn.Module):
    def __init__(self, config, intermediate_size: int | None = None) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(intermediate_size or config.intermediate_size)
        self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        activation = _hidden_activation(config)
        if activation in {"silu", "swiglu"}:
            self.act_fn = SiluAndMul()
        elif activation in {"gelu", "gelu_pytorch_tanh", "gelu_tanh"}:
            self.act_fn = GeluAndMul()
        else:
            raise NotImplementedError(f"Unsupported DiffusionGemma activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(torch.cat((self.gate_proj(x), self.up_proj(x)), dim=-1)))


class DiffusionGemmaRouter(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        num_experts = int(getattr(config, "num_experts"))
        self.norm = RMSNormNoWeight(hidden_size, eps=float(config.rms_norm_eps))
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(num_experts))
        self.register_buffer("root_size", torch.tensor(hidden_size**-0.5), persistent=False)
        self.proj = ReplicatedLinear(hidden_size, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = x * self.root_size.to(x.dtype)
        x = x * self.scale.to(x.dtype)
        return self.proj(x)


class DiffusionGemmaMoE(nn.Module):
    def __init__(self, config, intermediate_size: int) -> None:
        super().__init__()
        parallel_state = fetch_parallel_state()
        self.tp_rank = parallel_state.get_tp_rank()
        self.tp_size = parallel_state.get_tp_world_size()
        self.tp_group = parallel_state.get_tp_group()
        self.hidden_size = int(config.hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_experts = int(getattr(config, "num_experts"))
        self.top_k = int(_cfg(config, "top_k_experts", "num_experts_per_tok", "moe_top_k", default=2))
        self.num_local_experts = divide(self.num_experts, self.tp_size)
        self.local_expert_start = self.tp_rank * self.num_local_experts
        self.local_expert_end = self.local_expert_start + self.num_local_experts
        self.hidden_act = _hidden_activation(config)
        self.moe_gemm_impl = str(getattr(config, "moe_gemm_impl", "triton"))
        self.w13 = nn.Parameter(torch.empty(self.num_local_experts, self.intermediate_size * 2, self.hidden_size))
        self.w2 = nn.Parameter(torch.empty(self.num_local_experts, self.hidden_size, self.intermediate_size))

    def forward(self, x: torch.Tensor, router_logits: torch.Tensor, per_expert_scale: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, original_shape[-1])
        logits = router_logits.reshape(-1, router_logits.shape[-1]).to(torch.float32)
        _, topk_ids = torch.topk(logits, k=self.top_k, dim=-1)
        scores = torch.softmax(logits, dim=-1)
        topk_weights = scores.gather(1, topk_ids)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        topk_weights = topk_weights * per_expert_scale.to(device=topk_weights.device, dtype=topk_weights.dtype)[topk_ids]
        out = fused_moe(
            hidden_states=x_flat,
            w13=self.w13,
            w2=self.w2,
            topk_ids=topk_ids.to(torch.int32),
            topk_weights=topk_weights.to(x_flat.dtype),
            local_expert_start=self.local_expert_start,
            hidden_act=self.hidden_act,
        )
        if self.tp_size > 1:
            out = tp_all_reduce(out, self.tp_group)
        return out.reshape(original_shape)

    def resolve_checkpoint_weight(self, suffix: str, ctx: LoadContext) -> ResolvedWeight | None:
        del ctx
        if suffix == "experts.gate_up_proj":
            return ResolvedWeight(loader=self._load_stacked_gate_up)
        if suffix == "experts.down_proj":
            return ResolvedWeight(loader=self._load_stacked_down)

        match = re.fullmatch(r"experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", suffix)
        if match is None:
            return None
        expert_idx = int(match.group(1))
        proj_name = match.group(2)
        local_expert_idx = self._local_expert_idx(expert_idx)
        if local_expert_idx is None:
            return ResolvedWeight(skip=True)
        if proj_name == "gate_proj":
            return ResolvedWeight(
                loader=lambda w, e=local_expert_idx: self.w13.data[e, : self.intermediate_size].copy_(w)
            )
        if proj_name == "up_proj":
            return ResolvedWeight(
                loader=lambda w, e=local_expert_idx: self.w13.data[e, self.intermediate_size :].copy_(w)
            )
        return ResolvedWeight(loader=lambda w, e=local_expert_idx: self._load_w2(e, w))

    def _local_expert_idx(self, expert_idx: int) -> int | None:
        if expert_idx < self.local_expert_start or expert_idx >= self.local_expert_end:
            return None
        return expert_idx - self.local_expert_start

    def _load_stacked_gate_up(self, weight: torch.Tensor) -> None:
        local = weight[self.local_expert_start : self.local_expert_end]
        if local.shape != self.w13.data.shape:
            raise ValueError(f"Unexpected DiffusionGemma gate_up_proj shape: {local.shape}, target={self.w13.shape}")
        self.w13.data.copy_(local)

    def _load_stacked_down(self, weight: torch.Tensor) -> None:
        local = weight[self.local_expert_start : self.local_expert_end]
        if local.shape == self.w2.data.shape:
            self.w2.data.copy_(local)
        else:
            self.w2.data.copy_(local.transpose(1, 2).contiguous())

    def _load_w2(self, expert_idx: int, weight: torch.Tensor) -> None:
        target = self.w2.data[expert_idx]
        if weight.shape == target.shape:
            target.copy_(weight)
        else:
            target.copy_(weight.transpose(0, 1).contiguous())


class DiffusionGemmaAttention(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        parallel_state = fetch_parallel_state()
        tp_size = parallel_state.get_tp_world_size()
        layer_type = _layer_types(config)[layer_idx]
        self.is_full_attention = layer_type == "full_attention"
        self.use_k_eq_v = self.is_full_attention and bool(getattr(config, "attention_k_eq_v", True))

        self.total_num_heads = int(config.num_attention_heads)
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = int(
            getattr(config, "num_global_key_value_heads", config.num_key_value_heads)
            if self.use_k_eq_v
            else config.num_key_value_heads
        )
        self.replicate_kv_heads = self.total_num_kv_heads < tp_size
        if self.replicate_kv_heads:
            self.num_kv_heads = self.total_num_kv_heads
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
        self.head_dim = int(
            getattr(config, "global_head_dim", getattr(config, "head_dim", config.hidden_size // self.total_num_heads))
            if self.is_full_attention
            else getattr(config, "head_dim", config.hidden_size // self.total_num_heads)
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        bias = bool(getattr(config, "attention_bias", False))
        kv_linear_cls = ReplicatedLinear if self.replicate_kv_heads else ColumnParallelLinear
        self.q_proj = ColumnParallelLinear(config.hidden_size, self.total_num_heads * self.head_dim, bias=bias)
        self.k_proj = kv_linear_cls(config.hidden_size, self.total_num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = None if self.use_k_eq_v else kv_linear_cls(
            config.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=bias,
        )
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, config.hidden_size, bias=bias)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNormNoWeight(self.head_dim, eps=config.rms_norm_eps)

        rope_theta = float(getattr(config, "rope_theta", 10000.0))
        rope_type = "default"
        partial_rotary_factor = 1.0
        rope_parameters = getattr(config, "rope_parameters", None)
        if isinstance(rope_parameters, dict):
            params = rope_parameters.get(layer_type, rope_parameters)
            rope_theta = float(params.get("rope_theta", params.get("base", rope_theta)))
            rope_type = str(params.get("rope_type", params.get("type", rope_type)))
            partial_rotary_factor = float(params.get("partial_rotary_factor", partial_rotary_factor))
        if layer_type == "sliding_attention":
            rope_theta = float(getattr(config, "rope_local_base_freq", rope_theta))
        rotary_dim = int(self.head_dim * partial_rotary_factor)
        if rope_type == "proportional":
            self.rotary_emb = get_gemma4_proportional_rope(
                self.head_dim,
                rotary_dim=rotary_dim,
                max_position=int(config.max_position_embeddings),
                base=rope_theta,
            )
        else:
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=rotary_dim,
                max_position=int(config.max_position_embeddings),
                base=rope_theta,
            )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            1.0,
            self.num_kv_heads,
            attn_impl=getattr(config, "attn_impl", "triton"),
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        raw_k = self.k_proj(hidden_states)
        raw_v = raw_k if self.use_k_eq_v else self.v_proj(hidden_states)

        q = self.q_norm(q.view(q.shape[0], self.num_heads, self.head_dim))
        k = self.k_norm(raw_k.view(raw_k.shape[0], self.num_kv_heads, self.head_dim))
        v = self.v_norm(raw_v.view(raw_v.shape[0], self.num_kv_heads, self.head_dim))
        q, k = self.rotary_emb(positions, q, k)
        return self.o_proj(self.attn(q, k, v, mask))


class DiffusionGemmaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.self_attn = DiffusionGemmaAttention(config, layer_idx)
        first_kv_shared_layer_idx = int(config.num_hidden_layers) - int(getattr(config, "num_kv_shared_layers", 0) or 0)
        use_double = bool(getattr(config, "use_double_wide_mlp", False)) and layer_idx >= first_kv_shared_layer_idx > 0
        intermediate_size = int(config.intermediate_size) * (2 if use_double else 1)
        self.mlp = DiffusionGemmaMLP(config, intermediate_size=intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.enable_moe_block = bool(getattr(config, "enable_moe_block", False) or getattr(config, "use_second_mlp_block", False))
        self.router_uses_prenormed_input = bool(getattr(config, "router_uses_prenormed_input", False))
        if self.enable_moe_block:
            moe_intermediate = int(_cfg(config, "moe_intermediate_size", "expert_intermediate_size", default=intermediate_size))
            self.router = DiffusionGemmaRouter(config)
            self.moe = DiffusionGemmaMoE(config, moe_intermediate)
            self.pre_feedforward_layernorm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_feedforward_layernorm_1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_feedforward_layernorm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.router = None
            self.moe = None
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(residual)
        hidden_states = self.self_attn(positions, hidden_states, mask)
        hidden_states = self.post_attention_layernorm(hidden_states) + residual
        residual = hidden_states

        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.enable_moe_block:
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)
            hidden_states_2 = self.pre_feedforward_layernorm_2(residual)
            router_input = hidden_states_2 if self.router_uses_prenormed_input else residual
            hidden_states_2 = self.moe(hidden_states_2, self.router(router_input), self.router.per_expert_scale)
            hidden_states = hidden_states_1 + self.post_feedforward_layernorm_2(hidden_states_2)
        hidden_states = self.post_feedforward_layernorm(hidden_states) + residual
        return hidden_states * self.layer_scalar


class DiffusionGemmaTextModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DiffusionGemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("normalizer", torch.tensor(math.sqrt(config.hidden_size)), persistent=False)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.normalizer.to(self.embed_tokens.weight.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, mask)
        return self.norm(hidden_states)


class DiffusionGemmaSelfConditioning(nn.Module):
    def __init__(self, hidden_size: int, self_conditioning_size: int, eps: float) -> None:
        super().__init__()
        self.pre_norm = RMSNorm(hidden_size, eps=eps)
        self.post_norm = RMSNormNoWeight(hidden_size, eps=eps)
        self.gate_proj = nn.Linear(hidden_size, self_conditioning_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, self_conditioning_size, bias=False)
        self.down_proj = nn.Linear(self_conditioning_size, hidden_size, bias=False)

    def forward(self, inputs_embeds: torch.Tensor, soft_embeds: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(soft_embeds)
        sc_signal = self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
        return self.post_norm(inputs_embeds + sc_signal)


@AutoModelForDiffusionLM.register("diffusion_gemma", use_full_config=True)
class DiffusionGemmaForDiffusionLM(nn.Module):
    packed_modules_mapping = {}

    def __init__(self, config) -> None:
        super().__init__()
        hf_config = config.hf_config
        text_config = _text_config(hf_config)
        text_config.router_uses_prenormed_input = False
        text_config.attention_k_eq_v = True
        if getattr(text_config, "num_experts", None):
            text_config.enable_moe_block = True
        self.model = DiffusionGemmaTextModel(text_config)
        self.lm_head = ParallelLMHead(text_config.vocab_size, text_config.hidden_size)
        if getattr(text_config, "tie_word_embeddings", False):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        self.final_logit_softcapping = getattr(text_config, "final_logit_softcapping", None)
        sc_size = int(getattr(hf_config, "self_conditioning_size", None) or text_config.intermediate_size)
        self.self_conditioning = DiffusionGemmaSelfConditioning(
            text_config.hidden_size,
            sc_size,
            eps=float(getattr(text_config, "rms_norm_eps", 1e-6)),
        )
        self._self_conditioning_context: list[dict] | None = None

    def set_self_conditioning_context(self, context: list[dict] | None) -> None:
        self._self_conditioning_context = context

    def clear_self_conditioning_context(self) -> None:
        self._self_conditioning_context = None

    def _apply_self_conditioning_context(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        context = self._self_conditioning_context
        if not context:
            return inputs_embeds
        inputs_embeds = inputs_embeds.clone()
        for item in context:
            start = int(item["start"])
            end = int(item["end"])
            if end <= start:
                continue
            soft_embeds = item.get("soft_embeds")
            if soft_embeds is None:
                continue
            length = min(end - start, int(soft_embeds.shape[0]), int(inputs_embeds.shape[0]) - start)
            if length <= 0:
                continue
            span = slice(start, start + length)
            inputs_embeds[span] = self.self_conditioning(
                inputs_embeds[span],
                soft_embeds[:length].to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
            )
        return inputs_embeds

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        inputs_embeds = self.model.embed_input_ids(input_ids)
        inputs_embeds = self._apply_self_conditioning_context(inputs_embeds)
        try:
            return self.model(input_ids, positions, mask, inputs_embeds=inputs_embeds)
        finally:
            self.clear_self_conditioning_context()

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states, gather_all=True)
        if logits is not None and self.final_logit_softcapping is not None:
            cap = float(self.final_logit_softcapping)
            logits = torch.tanh(logits.to(torch.float32) / cap) * cap
        return logits

    def resolve_checkpoint_weight(self, weight_name: str, ctx: LoadContext) -> ResolvedWeight | None:
        if weight_name.startswith(("model.encoder.vision_tower.", "model.encoder.embed_vision.")):
            return ResolvedWeight(skip=True)
        if "embed_vision.embedding." in weight_name:
            return ResolvedWeight(skip=True)
        if "self_conditioning" in weight_name:
            mapped = "self_conditioning." + weight_name.split("self_conditioning.", 1)[1]
            return self._resolve_direct(mapped)
        if weight_name.startswith("model.encoder.language_model."):
            weight_name = weight_name.replace("model.encoder.language_model.", "model.", 1)
        elif weight_name.startswith("model.decoder."):
            weight_name = weight_name.replace("model.decoder.", "model.", 1)
        if ".moe.experts." not in weight_name:
            weight_name = weight_name.replace(".experts.", ".moe.experts.")
        moe_match = re.fullmatch(r"model\.layers\.(\d+)\.moe\.(experts\..+)", weight_name)
        if moe_match is not None:
            layer_idx = int(moe_match.group(1))
            if 0 <= layer_idx < len(self.model.layers):
                moe = getattr(self.model.layers[layer_idx], "moe", None)
                if moe is not None:
                    return moe.resolve_checkpoint_weight(moe_match.group(2), ctx)
        if ".v_proj." in weight_name:
            layer_match = re.search(r"model\.layers\.(\d+)\.", weight_name)
            if layer_match is not None:
                layer = self.model.layers[int(layer_match.group(1))]
                if getattr(layer.self_attn, "v_proj", None) is None:
                    return ResolvedWeight(skip=True)
        return self._resolve_direct(weight_name)

    def _resolve_direct(self, name: str) -> ResolvedWeight | None:
        try:
            return ResolvedWeight(param=self.get_parameter(name))
        except (AttributeError, KeyError):
            pass
        try:
            return ResolvedWeight(buffer=self.get_buffer(name))
        except (AttributeError, KeyError):
            return None


__all__ = ["DiffusionGemmaForDiffusionLM"]
