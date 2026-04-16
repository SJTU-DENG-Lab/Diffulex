from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from diffulex.model.sdar import SDARAttention
from diffulex.model.sdar_moe import SDARMoEDecoderLayer, SDARMoEForDiffusionLM
from diffulex.moe import SparseMoEBlock, build_mlp_or_moe, is_moe_layer
from diffulex.moe.layer.trivial_impl import TrivialFusedMoE
from diffulex.utils import parallelism


def _mock_tp(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    parallelism.reset_model_parallelism_metadata()
    monkeypatch.setattr(
        parallelism,
        "_MODEL_PARALLELISM_METADATA",
        parallelism.ModelParallelismMetadata.from_world(
            tp_size=1,
            ep_size=1,
            world_size=1,
            global_rank=0,
        ),
    )


def _make_config(**overrides):
    config = dict(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=12,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        hidden_act="silu",
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        attention_bias=False,
        rope_theta=10000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        decoder_sparse_step=2,
        mlp_only_layers=[0],
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def test_is_moe_layer_matches_reference_rule():
    config = _make_config()

    assert is_moe_layer(config, 0) is False
    assert is_moe_layer(config, 1) is True
    assert is_moe_layer(config, 2) is False


def test_build_mlp_or_moe_selects_sparse_block(monkeypatch):
    _mock_tp(monkeypatch)
    config = _make_config()

    dense = build_mlp_or_moe(config, 0, lambda: nn.Identity())
    sparse = build_mlp_or_moe(config, 1, lambda: nn.Identity())

    assert isinstance(dense, nn.Identity)
    assert isinstance(sparse, SparseMoEBlock)


def test_sdar_moe_parameter_names_match_reference_layout(monkeypatch):
    _mock_tp(monkeypatch)
    model = SDARMoEForDiffusionLM(_make_config())
    param_names = set(model.state_dict().keys())

    assert "model.layers.1.mlp.gate.weight" in param_names
    assert "model.layers.1.mlp.w13" in param_names
    assert "model.layers.1.mlp.w2" in param_names
    assert "model.layers.0.mlp.gate_proj.weight" in param_names


def test_sdar_moe_forward_shape(monkeypatch):
    _mock_tp(monkeypatch)
    monkeypatch.setattr(
        SDARAttention,
        "forward",
        lambda self, positions, hidden_states, mask=None: hidden_states,
    )
    monkeypatch.setattr(
        TrivialFusedMoE,
        "forward",
        lambda self, hidden_states: (hidden_states, None),
    )
    model = SDARMoEForDiffusionLM(_make_config())
    model.eval()

    input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    positions = torch.arange(input_ids.numel(), dtype=torch.long)
    with torch.no_grad():
        hidden_states = model(input_ids, positions)

    assert hidden_states.shape == (4, 8)


def test_decoder_layer_uses_sparse_block_on_selected_layers(monkeypatch):
    _mock_tp(monkeypatch)
    dense_layer = SDARMoEDecoderLayer(_make_config(), layer_idx=0)
    sparse_layer = SDARMoEDecoderLayer(_make_config(), layer_idx=1)

    assert not isinstance(dense_layer.mlp, SparseMoEBlock)
    assert isinstance(sparse_layer.mlp, SparseMoEBlock)
