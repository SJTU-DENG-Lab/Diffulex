from __future__ import annotations

import json
from types import SimpleNamespace

import diffulex.distributed.parallel_state as parallel_state
from diffulex.config import Config
from diffulex.model.auto_model import AutoModelForDiffusionLM
from diffulex.model.dream_reasoner import DreamReasonerForDiffusionLM
from diffulex.sampler.auto_sampler import AutoSampler
from diffulex.sampler.base import DllmSamplerNoShiftBase


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
        attention_bias=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
        attn_impl="naive",
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def test_dream_reasoner_registered():
    assert "dream_reasoner" in AutoModelForDiffusionLM.available_models()
    assert "dream_reasoner" in AutoSampler.available_samplers()


def test_dream_reasoner_sampler_is_no_shift():
    sampler = AutoSampler.SAMPLER_MAPPING["dream_reasoner"][0]()
    assert isinstance(sampler, DllmSamplerNoShiftBase)


def test_dream_reasoner_config_uses_local_dream_config(tmp_path):
    model_dir = tmp_path / "dream_reasoner"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DreamForCausalLM"],
                "auto_map": {
                    "AutoConfig": "configuration_dream.DreamConfig",
                    "AutoModelForCausalLM": "modeling_dream.DreamForCausalLM",
                },
                "model_type": "Dream",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": None,
                "max_position_embeddings": 64,
                "vocab_size": 32,
                "block_size": 32,
                "mask_token_id": 151669,
            }
        )
    )

    config = Config(
        str(model_dir),
        model_name="dream_reasoner",
        decoding_strategy="multi_bd",
        tensor_parallel_size=1,
        max_model_len=32,
        max_num_batched_tokens=32,
        device_ids=[0],
        block_size=32,
        page_size=32,
    )

    assert type(config.hf_config).__name__ == "DreamConfig"
    assert config.mask_token_id == 151669
    assert config.hf_config.mask_token_id == 151669
    assert config.hf_config.head_dim == 4


def test_dream_reasoner_parameter_names_match_checkpoint(monkeypatch):
    _mock_single_rank(monkeypatch)
    model = DreamReasonerForDiffusionLM(_make_config())
    state = model.state_dict()

    assert "model.layers.0.self_attn.q_norm.weight" in state
    assert "model.layers.0.self_attn.k_norm.weight" in state
    assert tuple(state["model.layers.0.self_attn.q_norm.weight"].shape) == (4,)
    assert "model.layers.0.self_attn.q_proj.weight" in state
    assert "model.layers.0.self_attn.q_proj.bias" not in state
    assert "model.layers.0.self_attn.o_proj.bias" not in state
    assert "lm_head.weight" in state
