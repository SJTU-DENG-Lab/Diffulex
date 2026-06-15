from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig


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


try:
    AutoConfig.register(DiffusionGemmaConfig.model_type, DiffusionGemmaConfig)
except ValueError:
    pass


__all__ = ["DiffusionGemmaConfig", "DiffusionGemmaTextConfig"]
