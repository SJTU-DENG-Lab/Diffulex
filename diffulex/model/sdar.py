import os

import torch
import torch.nn as nn
from einops import rearrange

from diffulex.layer.layernorm import RMSNorm
from diffulex.layer.embed_head import VocabParallelEmbedding, ParallelLMHead
from diffulex.model.common import MergedQKVAttention, MergedSwiGLUMLP
from diffulex.model.auto_model import AutoModelForDiffusionLM
from diffulex.model.config.sdar.configuration_sdar import SDARConfig


if os.environ.get("TRITON_INTERPRET", None) == "1":
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = True
    torch.backends.optimized_mode = False


class SDARAttention(MergedQKVAttention):
    """SDAR attention (Diffulex native KV cache path).

    Compatible with Diffulex runner KV cache injection:
    runner sets `self.attn.k_cache` / `self.attn.v_cache` by assigning to modules
    that expose these attributes (see `diffulex/attention/attn_impl.py`).
    """

    def __init__(self, config: SDARConfig) -> None:
        bias = getattr(config, "attention_bias", False)
        super().__init__(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, "head_dim", None),
            qkv_bias=bias,
            out_bias=bias,
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
            attn_impl=getattr(config, "attn_impl", "triton"),
            qk_norm_eps=config.rms_norm_eps,
        )


class SDARMLP(MergedSwiGLUMLP):
    """SDAR MLP: SiLU(gate) * up -> down."""

    def __init__(self, config: SDARConfig) -> None:
        super().__init__(config.hidden_size, config.intermediate_size, hidden_act=getattr(config, "hidden_act", "silu"))


class SDARDecoderLayer(nn.Module):
    def __init__(self, config: SDARConfig) -> None:
        super().__init__()
        self.self_attn = SDARAttention(config)
        self.mlp = SDARMLP(config)
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


class SDARModel(nn.Module):
    def __init__(self, config: SDARConfig) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([SDARDecoderLayer(config) for _ in range(config.num_hidden_layers)])
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


@AutoModelForDiffusionLM.register("sdar")
class SDARForDiffusionLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("self_attn.qkv_proj", "q"),
        "k_proj": ("self_attn.qkv_proj", "k"),
        "v_proj": ("self_attn.qkv_proj", "v"),
        "gate_proj": ("mlp.gate_up_proj", 0),
        "up_proj": ("mlp.gate_up_proj", 1),
    }

    def __init__(self, config: SDARConfig) -> None:
        super().__init__()
        self.model = SDARModel(config)
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
    "SDARConfig",
    "SDARAttention",
    "SDARMLP",
    "SDARDecoderLayer",
    "SDARModel",
    "SDARForDiffusionLM",
]
