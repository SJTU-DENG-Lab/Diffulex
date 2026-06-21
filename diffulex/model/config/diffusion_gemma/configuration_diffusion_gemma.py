from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig

from diffulex.hf_config_registry import HFConfigRegistry


class DiffusionGemmaTextConfig(PretrainedConfig):
    model_type = "diffusion_gemma_text"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if getattr(self, "num_experts", None):
            self.enable_moe_block = True
        self.attention_k_eq_v = True


class DiffusionGemmaConfig(PretrainedConfig):
    model_type = "diffusion_gemma"

    def __init__(
        self,
        text_config: dict[str, Any] | PretrainedConfig | None = None,
        canvas_length: int = 256,
        self_conditioning_size: int | None = None,
        **kwargs: Any,
    ):
        if isinstance(text_config, PretrainedConfig):
            self.text_config = text_config
        else:
            self.text_config = DiffusionGemmaTextConfig(**(text_config or {}))
        self.canvas_length = canvas_length
        self.self_conditioning_size = self_conditioning_size
        self.vision_config = kwargs.pop("vision_config", None)
        self.audio_config = None
        super().__init__(**kwargs)


@HFConfigRegistry.register_model_type(DiffusionGemmaConfig.model_type)
def load_diffusion_gemma_config(model: str, config_dict: dict):
    del config_dict
    return DiffusionGemmaConfig.from_pretrained(model)


@HFConfigRegistry.register_postprocessor(DiffusionGemmaConfig.model_type)
def postprocess_diffusion_gemma_config(hf_config, engine_config, config_dict: dict) -> None:
    del config_dict
    text_config = getattr(hf_config, "text_config", None)
    if text_config is None:
        return

    canvas_length = getattr(hf_config, "canvas_length", None)
    if canvas_length is not None and int(canvas_length) != int(engine_config.block_size):
        raise ValueError(
            "DiffusionGemma hf_config.canvas_length must match block_size, "
            f"got canvas_length={canvas_length}, block_size={engine_config.block_size}."
        )

    for name in (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "global_head_dim",
        "max_position_embeddings",
        "rms_norm_eps",
    ):
        if getattr(hf_config, name, None) is None and getattr(text_config, name, None) is not None:
            setattr(hf_config, name, getattr(text_config, name))

    head_dim = getattr(text_config, "head_dim", None)
    global_head_dim = getattr(text_config, "global_head_dim", None)
    if head_dim is not None and global_head_dim is not None:
        hf_config.head_dim = max(int(head_dim), int(global_head_dim))


__all__ = ["DiffusionGemmaConfig", "DiffusionGemmaTextConfig"]
