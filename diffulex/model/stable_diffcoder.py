import os

import torch
import torch.nn as nn

from diffulex.attention import Attention
from diffulex.distributed.parallel_state import fetch_parallel_state
from diffulex.layer.activation import SiluAndMul
from diffulex.layer.embed_head import ParallelLMHead, VocabParallelEmbedding
from diffulex.layer.layernorm import RMSNorm
from diffulex.layer.linear import ColumnParallelLinear, RowParallelLinear
from diffulex.layer.rotary_embedding import get_rope
from diffulex.model.auto_model import AutoModelForDiffusionLM
from diffulex.model.config.stable_diffcoder import StableDiffCoderConfig


if os.environ.get("TRITON_INTERPRET", None) == "1":
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = True
    torch.backends.optimized_mode = False


class StableDiffCoderAttention(nn.Module):
    def __init__(self, config: StableDiffCoderConfig) -> None:
        super().__init__()
        parallel_state = fetch_parallel_state()
        tp_size = parallel_state.get_tp_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.total_num_heads
        self.scaling = self.head_dim**-0.5

        bias = bool(getattr(config, "attention_bias", False))
        self.q_proj = ColumnParallelLinear(config.hidden_size, self.total_num_heads * self.head_dim, bias=bias)
        self.k_proj = ColumnParallelLinear(config.hidden_size, self.total_num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = ColumnParallelLinear(config.hidden_size, self.total_num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, config.hidden_size, bias=bias)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            attn_impl=getattr(config, "attn_impl", "triton_grouped"),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q, k = self.rotary_emb(positions, q, k)
        return self.o_proj(self.attn(q, k, v, mask))


class StableDiffCoderMLP(nn.Module):
    def __init__(self, config: StableDiffCoderConfig) -> None:
        super().__init__()
        bias = bool(getattr(config, "mlp_bias", False))
        self.gate_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=bias)
        self.up_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=bias)
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=bias)
        assert getattr(config, "hidden_act", "silu") == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(self.act_fn(torch.cat([gate, up], dim=-1)))


class StableDiffCoderDecoderLayer(nn.Module):
    def __init__(self, config: StableDiffCoderConfig) -> None:
        super().__init__()
        self.self_attn = StableDiffCoderAttention(config)
        self.mlp = StableDiffCoderMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, mask)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class StableDiffCoderModel(nn.Module):
    def __init__(self, config: StableDiffCoderConfig) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([StableDiffCoderDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, mask)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@AutoModelForDiffusionLM.register("stable_diffcoder")
class StableDiffCoderForDiffusionLM(nn.Module):
    packed_modules_mapping = {}

    def __init__(self, config: StableDiffCoderConfig) -> None:
        super().__init__()
        self.model = StableDiffCoderModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, mask)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


__all__ = [
    "StableDiffCoderConfig",
    "StableDiffCoderAttention",
    "StableDiffCoderForDiffusionLM",
]
