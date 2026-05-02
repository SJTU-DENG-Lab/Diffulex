import os
import torch
import torch.nn as nn

from diffulex.layer.layernorm import RMSNorm
from diffulex.model.auto_model import AutoModelForDiffusionLM
from diffulex.model.config.dream.configuration_dream import DreamConfig
from diffulex.layer.embed_head import VocabParallelEmbedding, ParallelLMHead
from diffulex.model.common import MergedQKVAttention, MergedSwiGLUMLP


if os.environ.get("TRITON_INTERPRET", None) == "1":
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = True
    torch.backends.optimized_mode = False


class DreamRMSNorm(RMSNorm):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)


class DreamAttention(MergedQKVAttention):
    """Dream attention mechanism."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 32768,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = True,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
        attn_impl: str = "triton",
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_position=max_position,
            head_dim=head_dim,
            qkv_bias=qkv_bias,
            out_bias=False,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attn_impl=attn_impl,
        )


class DreamMLP(MergedSwiGLUMLP):
    """Dream MLP with SiLU activation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__(hidden_size, intermediate_size, hidden_act=hidden_act)


class DreamDecoderLayer(nn.Module):
    """Dream transformer decoder layer."""

    def __init__(
        self,
        config: DreamConfig,
    ) -> None:
        super().__init__()
        self.self_attn = DreamAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=True,  # Dream uses bias in attention
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
            attn_impl=getattr(config, "attn_impl", "triton"),
        )
        self.mlp = DreamMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class DreamModel(nn.Module):
    """Dream model for diffusion language modeling."""

    def __init__(
        self,
        config: DreamConfig,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DreamDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for _, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual, mask)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@AutoModelForDiffusionLM.register("dream")
class DreamForDiffusionLM(nn.Module):
    """Dream model for diffusion language modeling with LM head."""

    packed_modules_mapping = {
        "q_proj": ("self_attn.qkv_proj", "q"),
        "k_proj": ("self_attn.qkv_proj", "k"),
        "v_proj": ("self_attn.qkv_proj", "v"),
        "gate_proj": ("mlp.gate_up_proj", 0),
        "up_proj": ("mlp.gate_up_proj", 1),
    }

    def __init__(
        self,
        config: DreamConfig,
    ) -> None:
        super().__init__()
        self.model = DreamModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, mask)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits
