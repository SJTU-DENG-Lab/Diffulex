from __future__ import annotations

import json
from types import SimpleNamespace

import diffulex.distributed.parallel_state as parallel_state
import torch
from safetensors.torch import save_file

from diffulex.config import Config
from diffulex.model.auto_model import AutoModelForDiffusionLM
from diffulex.model.dream import DreamForDiffusionLM
from diffulex.sampler.auto_sampler import AutoSampler
from diffulex.sampler.base import DllmSamplerShiftBase
from diffulex.utils.loader import enable_lora_for_model, load_lora_weights


def _mock_single_rank(monkeypatch):
    parallel_state.reset_parallel_state()
    monkeypatch.setattr(
        parallel_state,
        "PARALLEL_STATE",
        parallel_state.build_parallel_state_for_test(
            tp_size=1,
            ep_size=1,
            dp_size=1,
            world_size=1,
            global_rank=0,
        ),
    )


def _make_config(**overrides):
    config = dict(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        hidden_act="silu",
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
        attn_impl="naive",
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def test_diffucoder_registered():
    assert "diffucoder" in AutoModelForDiffusionLM.available_models()
    assert "diffucoder" in AutoSampler.available_samplers()


def test_diffucoder_sampler_is_shifted():
    sampler = AutoSampler.SAMPLER_MAPPING["diffucoder"][0]()
    assert isinstance(sampler, DllmSamplerShiftBase)


def test_diffucoder_config_uses_local_dream_config(tmp_path):
    model_dir = tmp_path / "diffucoder"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DreamModel"],
                "auto_map": {
                    "AutoConfig": "configuration_dream.DreamConfig",
                    "AutoModel": "modeling_dream.DreamModel",
                },
                "model_type": "Dream",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "max_position_embeddings": 64,
                "vocab_size": 32,
                "mask_token_id": 151666,
            }
        )
    )

    config = Config(
        str(model_dir),
        model_name="diffucoder",
        decoding_strategy="d2f",
        tensor_parallel_size=1,
        max_model_len=32,
        max_num_batched_tokens=32,
        device_ids=[0],
        block_size=32,
        page_size=32,
    )

    assert type(config.hf_config).__name__ == "DreamConfig"
    assert config.mask_token_id == 151666
    assert config.hf_config.mask_token_id == 151666
    assert config.decoding_strategy == "d2f"
    assert config.multi_block_prefix_full is True
    assert config.enable_prefix_caching is False


def test_diffucoder_parameter_names_match_lora_adapter(monkeypatch):
    _mock_single_rank(monkeypatch)
    model = DreamForDiffusionLM(_make_config())
    state = model.state_dict()

    assert "model.layers.0.self_attn.q_proj.weight" in state
    assert "model.layers.0.self_attn.q_proj.bias" in state
    assert "model.layers.0.self_attn.k_proj.bias" in state
    assert "model.layers.0.self_attn.v_proj.bias" in state
    assert "model.layers.0.self_attn.o_proj.weight" in state
    assert "model.layers.0.self_attn.q_norm.weight" not in state
    assert "lm_head.weight" in state

    module_name = "model.layers.0.self_attn.q_proj"
    assert f"base_model.model.{module_name}.lora_A.weight" == (
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    )


def test_diffucoder_lora_prefix_loads_into_dream_module(monkeypatch, tmp_path):
    _mock_single_rank(monkeypatch)
    model = DreamForDiffusionLM(_make_config())
    model = enable_lora_for_model(
        model,
        {
            "r": 2,
            "lora_alpha": 4.0,
            "lora_dropout": 0.0,
            "target_modules": ["q_proj"],
        },
    )
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 2,
                "lora_alpha": 4.0,
                "lora_dropout": 0.0,
                "target_modules": ["q_proj"],
            }
        )
    )
    lora_a = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    lora_b = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    save_file(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": lora_a,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": lora_b,
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )

    load_lora_weights(model, str(adapter_dir), pre_merge_lora=False)

    q_proj = model.model.layers[0].self_attn.q_proj
    torch.testing.assert_close(q_proj.lora_A, lora_a)
    torch.testing.assert_close(q_proj.lora_B, lora_b)
