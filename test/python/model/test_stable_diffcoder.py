from __future__ import annotations

import json
from types import SimpleNamespace

import diffulex.distributed.parallel_state as parallel_state
from diffulex.config import Config
from diffulex.model.auto_model import AutoModelForDiffusionLM
from diffulex.model.stable_diffcoder import StableDiffCoderForDiffusionLM
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
        mlp_bias=False,
        rope_theta=500000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
        attn_impl="naive",
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def test_stable_diffcoder_registered():
    assert "stable_diffcoder" in AutoModelForDiffusionLM.available_models()
    assert "stable_diffcoder" in AutoSampler.available_samplers()


def test_stable_diffcoder_sampler_is_no_shift():
    sampler = AutoSampler.SAMPLER_MAPPING["stable_diffcoder"][0]()
    assert isinstance(sampler, DllmSamplerNoShiftBase)


def test_stable_diffcoder_config_uses_local_config_and_tokenizer_mask(tmp_path):
    model_dir = tmp_path / "stable_diffcoder"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["StableDiffcoderForCausalLM"],
                "auto_map": {
                    "AutoModelForCausalLM": "modeling_stable_diffcoder.StableDiffcoderForCausalLM",
                },
                "model_type": "llama",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "max_position_embeddings": 64,
                "vocab_size": 32,
                "attention_bias": False,
                "mlp_bias": False,
            }
        )
    )
    (model_dir / "special_tokens_map.json").write_text(
        json.dumps({"mask_token": {"content": "<[MASK_TOKEN]>"}})
    )
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "added_tokens_decoder": {
                    "5": {"content": "<[MASK_TOKEN]>", "special": True},
                }
            }
        )
    )

    config = Config(
        str(model_dir),
        model_name="stable_diffcoder",
        decoding_strategy="multi_bd",
        tensor_parallel_size=1,
        max_model_len=32,
        max_num_batched_tokens=32,
        device_ids=[0],
        block_size=32,
        page_size=32,
    )

    assert type(config.hf_config).__name__ == "StableDiffCoderConfig"
    assert config.hf_config.model_type == "llama"
    assert config.mask_token_id == 5


def test_stable_diffcoder_parameter_names_match_checkpoint(monkeypatch):
    _mock_single_rank(monkeypatch)
    model = StableDiffCoderForDiffusionLM(_make_config())
    state = model.state_dict()

    assert "model.embed_tokens.weight" in state
    assert "model.layers.0.self_attn.q_proj.weight" in state
    assert "model.layers.0.self_attn.k_proj.weight" in state
    assert "model.layers.0.self_attn.v_proj.weight" in state
    assert "model.layers.0.self_attn.o_proj.weight" in state
    assert "model.layers.0.self_attn.q_norm.weight" not in state
    assert "model.layers.0.self_attn.q_proj.bias" not in state
    assert "model.layers.0.mlp.gate_proj.bias" not in state
    assert "lm_head.weight" in state
