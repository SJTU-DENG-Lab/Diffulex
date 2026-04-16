from __future__ import annotations

from types import SimpleNamespace

import torch

from diffulex.model.auto_model import AutoModelForDiffusionLM
from diffulex.model.llada2 import LLaDA2DecoderLayer, LLaDA2ForDiffusionLM, LLaDA2Model, LLaDA2TrivialMoE
from diffulex.moe import is_moe_layer
from diffulex.utils import parallelism


def _mock_single_rank(monkeypatch):
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
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=4,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        hidden_act="silu",
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling=None,
        partial_rotary_factor=0.5,
        rotary_dim=2,
        use_qkv_bias=False,
        use_bias=False,
        tie_word_embeddings=False,
        num_experts=4,
        num_shared_experts=1,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        first_k_dense_replace=1,
        n_group=2,
        topk_group=1,
        routed_scaling_factor=2.5,
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def test_llada2_registered():
    assert "llada2" in AutoModelForDiffusionLM.available_models()
    assert "llada2_moe" in AutoModelForDiffusionLM.available_models()
    assert "llada2_mini" in AutoModelForDiffusionLM.available_models()


def test_llada2_first_k_dense_replace_rule():
    config = _make_config()
    assert is_moe_layer(config, 0) is False
    assert is_moe_layer(config, 1) is True
    assert is_moe_layer(config, 2) is True


def test_llada2_decoder_layer_uses_moe_after_dense_prefix(monkeypatch):
    _mock_single_rank(monkeypatch)
    dense_layer = LLaDA2DecoderLayer(_make_config(), layer_idx=0)
    moe_layer = LLaDA2DecoderLayer(_make_config(), layer_idx=1)

    assert not isinstance(dense_layer.mlp, LLaDA2TrivialMoE)
    assert isinstance(moe_layer.mlp, LLaDA2TrivialMoE)
    assert moe_layer.mlp.shared_experts is not None
    assert tuple(moe_layer.mlp.gate.expert_bias.shape) == (4,)


def test_llada2_parameter_names_match_dmax_layout(monkeypatch):
    _mock_single_rank(monkeypatch)
    model = LLaDA2ForDiffusionLM(_make_config())
    param_names = set(model.state_dict().keys())

    assert "model.word_embeddings.weight" in param_names
    assert "model.layers.0.attention.query_key_value.weight" in param_names
    assert "model.layers.0.attention.dense.weight" in param_names
    assert "model.layers.0.mlp.gate_proj.weight" in param_names
    assert "model.layers.1.mlp.gate.weight" in param_names
    assert "model.layers.1.mlp.gate.expert_bias" in param_names
    assert "model.layers.1.mlp.shared_experts.gate_proj.weight" in param_names
    assert "lm_head.weight" in param_names


def test_llada2_qkv_loader_accepts_combined_weight(monkeypatch):
    _mock_single_rank(monkeypatch)
    layer = LLaDA2DecoderLayer(_make_config(), layer_idx=0)
    qkv = layer.attention.query_key_value
    loaded = torch.arange(qkv.weight.numel(), dtype=qkv.weight.dtype).reshape_as(qkv.weight)

    qkv.weight.weight_loader(qkv.weight, loaded)

    assert torch.equal(qkv.weight, loaded)


def test_llada2_token_merge_hook_uses_attention_metadata(monkeypatch):
    _mock_single_rank(monkeypatch)
    from diffulex.attention.metadata import set_fetch_fn_for_attn_metadata
    from diffulex.strategy.dmax.attention.metadata import DMaxAttnMetaData

    model = LLaDA2Model(_make_config(num_hidden_layers=0, vocab_size=8, hidden_size=4))
    with torch.no_grad():
        model.word_embeddings.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                ]
            )
        )

    metadata = DMaxAttnMetaData()
    metadata.init_token_merging(
        merge_mask=torch.tensor([True, False]),
        topk_ids=torch.tensor([[2, 3], [0, 0]]),
        topk_probs=torch.tensor([[0.25, 0.50], [0.0, 0.0]]),
        residual_probs=torch.tensor([[0.25], [0.0]]),
        mask_token_id=4,
        renormalize=False,
    )
    set_fetch_fn_for_attn_metadata(lambda: metadata)

    hidden = model.word_embeddings(torch.tensor([0, 1]))
    merged = model._maybe_apply_token_merging(hidden)

    expected_first = 0.25 * model.word_embeddings.weight[2] + 0.50 * model.word_embeddings.weight[3]
    expected_first = expected_first + 0.25 * model.word_embeddings.weight[4]
    assert torch.allclose(merged[0], expected_first)
    assert torch.allclose(merged[1], hidden[1])


def test_llada2_token_merge_hook_supports_iter_smooth_mode(monkeypatch):
    _mock_single_rank(monkeypatch)
    from diffulex.attention.metadata import set_fetch_fn_for_attn_metadata
    from diffulex.strategy.dmax.attention.metadata import DMaxAttnMetaData

    model = LLaDA2Model(_make_config(num_hidden_layers=0, vocab_size=8, hidden_size=4))
    with torch.no_grad():
        model.word_embeddings.weight.copy_(torch.eye(8, 4))

    metadata = DMaxAttnMetaData()
    metadata.init_token_merging(
        merge_mask=torch.tensor([True, False]),
        topk_ids=torch.tensor([[2], [0]]),
        topk_probs=torch.tensor([[0.5], [0.0]]),
        residual_probs=torch.tensor([[0.5], [0.0]]),
        mask_token_id=4,
        renormalize=False,
        mode="iter_smooth_topk",
        weight=0.25,
    )
    set_fetch_fn_for_attn_metadata(lambda: metadata)

    hidden = model.word_embeddings(torch.tensor([0, 1]))
    merged = model._maybe_apply_token_merging(hidden)

    assert torch.allclose(merged[0], hidden[0] + 0.25 * 0.5 * model.word_embeddings.weight[2])
    assert torch.allclose(merged[1], hidden[1])
