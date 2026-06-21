from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation

from diffulex.hf_config_registry import HFConfigRegistry


class StableDiffCoderConfig(PretrainedConfig):
    """Local config for Stable-DiffCoder checkpoints.

    The checkpoint declares ``model_type=llama``. Loading Transformers'
    LlamaConfig currently pulls in optional vision dependencies in this env, so
    Diffulex reads the same JSON fields through this small local config.
    """

    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 155136,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = 8,
        head_dim: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 8192,
        initializer_range: float = 0.009882118,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 500000.0,
        rope_scaling=None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


def is_stable_diffcoder_config(config_dict: dict) -> bool:
    architectures = config_dict.get("architectures") or []
    auto_map = config_dict.get("auto_map") or {}
    auto_map_values = [value for value in auto_map.values() if isinstance(value, str)]
    return any("StableDiffcoder" in name for name in architectures) or any(
        "stable_diffcoder" in value for value in auto_map_values
    )


@HFConfigRegistry.register_predicate(is_stable_diffcoder_config)
def load_stable_diffcoder_config(model: str, config_dict: dict):
    del config_dict
    return StableDiffCoderConfig.from_pretrained(model)


__all__ = ["StableDiffCoderConfig"]
