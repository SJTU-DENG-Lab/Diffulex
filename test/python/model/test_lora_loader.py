from __future__ import annotations

import torch.nn as nn

from diffulex.utils.loader import enable_lora_for_model


class DummyLoraModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False

    def __init_lora__(self, r, lora_alpha, lora_dropout):
        self.initialized = True
        self.lora_args = (r, lora_alpha, lora_dropout)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = DummyLoraModule()
        self.self_attn = nn.Module()
        self.self_attn.o_proj = DummyLoraModule()


def test_enable_lora_accepts_string_target_modules():
    model = DummyModel()

    enable_lora_for_model(model, {"target_modules": "q_proj"})

    assert model.q_proj.initialized is True
    assert model.self_attn.o_proj.initialized is False


def test_enable_lora_matches_packed_module_checkpoint_names():
    model = DummyModel()

    enable_lora_for_model(
        model,
        {"target_modules": ["attn_out"]},
        packed_modules_mapping={"attn_out": ("self_attn.o_proj", None)},
    )

    assert model.q_proj.initialized is False
    assert model.self_attn.o_proj.initialized is True
