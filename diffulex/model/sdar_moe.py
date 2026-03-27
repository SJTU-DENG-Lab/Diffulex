from __future__ import annotations

import torch
import torch.nn as nn

from diffulex.layer.embed_head import ParallelLMHead, VocabParallelEmbedding
from diffulex.layer.layernorm import RMSNorm
from diffulex.model.auto_model import AutoModelForDiffusionLM
from diffulex.model.sdar import SDARAttention, SDARMLP
from diffulex.moe import build_mlp_or_moe


class SDARMoEDecoderLayer(nn.Module):
    """SDAR decoder layer with optional sparse MoE MLP."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = SDARAttention(config)
        self.mlp = build_mlp_or_moe(config, layer_idx, lambda: SDARMLP(config))
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

        mlp_output = self.mlp(hidden_states)
        if isinstance(mlp_output, tuple):
            hidden_states, _router_logits = mlp_output
        else:
            hidden_states = mlp_output
        return hidden_states, residual


class SDARMoEModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [SDARMoEDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
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


@AutoModelForDiffusionLM.register("sdar_moe")
class SDARMoEForDiffusionLM(nn.Module):
    packed_modules_mapping = {}

    def __init__(self, config) -> None:
        super().__init__()
        self.model = SDARMoEModel(config)
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
    "SDARMoEDecoderLayer",
    "SDARMoEModel",
    "SDARMoEForDiffusionLM",
]
