import os
import torch
import torch.nn as nn

from diffulex.layer.layernorm import RMSNorm
from diffulex.model.auto_model import AutoModelForDiffusionLM
from diffulex.model.common import MergedQKVAttention, MergedSwiGLUMLP
from diffulex.model.config.llada.configuration_llada import LLaDAConfig
from diffulex.layer.embed_head import VocabParallelEmbedding, ParallelLMHead


if os.environ.get("TRITON_INTERPRET", None) == "1":
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = True
    torch.backends.optimized_mode = False


class LLaDARMSNorm(RMSNorm):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)


class LLaDAAttention(MergedQKVAttention):
    """LLaDA attention."""

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


class LLaDAMLP(MergedSwiGLUMLP):
    """LLaDA MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__(hidden_size, intermediate_size, hidden_act=hidden_act)


class LLaDABlock(nn.Module):
    """LLaDA transformer block."""

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.self_attn = LLaDAAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.n_kv_heads,
            max_position=config.max_sequence_length,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "include_qkv_bias", getattr(config, "use_qkv_bias", False)),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
            attn_impl=getattr(config, "attn_impl", "triton"),
        )
        self.mlp = LLaDAMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.mlp_hidden_size,
            hidden_act=config.activation_type,
        )
        self.input_layernorm = LLaDARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LLaDARMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class LLaDAModel(nn.Module):
    """LLaDA backbone."""

    def __init__(
        self,
        config: LLaDAConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=VocabParallelEmbedding(config.embedding_size or config.vocab_size, config.d_model),
                emb_drop=nn.Dropout(config.embedding_dropout),
                ln_f=LLaDARMSNorm(config.hidden_size, config.rms_norm_eps),
            )
        )

        blocks = [LLaDABlock(config) for _ in range(config.n_layers)]
        self.transformer.update({"blocks": nn.ModuleList(blocks)})

        if not (self.config.alibi or self.config.rope):
            self.transformer.update(
                {
                    "wpe": nn.Embedding(
                        config.max_sequence_length,
                        config.d_model,
                        device=config.init_device,
                    )
                }
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.transformer.emb_drop(self.transformer.wte(input_ids))
        residual = None
        for block_idx, block in enumerate(self.transformer.blocks):
            hidden_states, residual = block(positions, hidden_states, residual, mask)
        hidden_states, _ = self.transformer.ln_f(hidden_states, residual)
        return hidden_states


@AutoModelForDiffusionLM.register("llada")
class LLaDAForDiffusionLM(nn.Module):
    """LLaDA with LM head."""

    packed_modules_mapping = {
        "q_proj": ("self_attn.qkv_proj", "q"),
        "k_proj": ("self_attn.qkv_proj", "k"),
        "v_proj": ("self_attn.qkv_proj", "v"),
        "attn_out": ("self_attn.o_proj", None),
        "attn_norm": ("input_layernorm", None),
        "ff_norm": ("post_attention_layernorm", None),
        "ff_proj": ("mlp.gate_up_proj", 0),
        "up_proj": ("mlp.gate_up_proj", 1),
        "ff_out": ("mlp.down_proj", None),
        "transformer.ff_out": ("lm_head", None),
    }

    def __init__(
        self,
        config: LLaDAConfig,
    ) -> None:
        super().__init__()
        self.model = LLaDAModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if getattr(config, "weight_tying", False):
            self.lm_head.weight.data = self.model.transformer.wte.weight.data

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
