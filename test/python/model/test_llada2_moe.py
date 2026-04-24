from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import diffulex.distributed.parallel_state as parallel_state
from diffulex.model.auto_model import AutoModelForDiffusionLM
from diffulex.model.llada2 import (
    LLaDA2DecoderLayer,
    LLaDA2ForDiffusionLM,
    LLaDA2Model,
    LLaDA2NaiveMoE,
    LLaDA2TPMoE,
)
from diffulex.moe import is_moe_layer
from diffulex.moe.topk import GroupLimitedTopKRouter
from diffulex.utils.loader import apply_resolved_weight, resolve_weight_spec


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


def _mock_tp_rank(monkeypatch, *, tp_size: int, global_rank: int):
    parallel_state.reset_parallel_state()
    monkeypatch.setattr(
        parallel_state,
        "PARALLEL_STATE",
        parallel_state.build_parallel_state_for_test(
            tp_size=tp_size,
            ep_size=1,
            dp_size=1,
            global_rank=global_rank,
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

    assert not isinstance(dense_layer.mlp, LLaDA2NaiveMoE)
    assert isinstance(moe_layer.mlp, LLaDA2NaiveMoE)
    assert moe_layer.mlp.shared_experts is not None
    assert tuple(moe_layer.mlp.gate.expert_bias.shape) == (4,)


def test_llada2_tp_moe_shards_experts_not_intermediate(monkeypatch):
    _mock_tp_rank(monkeypatch, tp_size=2, global_rank=1)
    moe = LLaDA2TPMoE.from_config(_make_config(num_experts=4, moe_intermediate_size=6))

    assert moe.num_local_experts == 2
    assert moe.local_expert_start == 2
    assert moe.local_expert_end == 4
    assert tuple(moe.w13.shape) == (2, 12, 8)
    assert tuple(moe.w2.shape) == (2, 8, 6)


def test_llada2_moe_gemm_impl_is_configurable(monkeypatch):
    _mock_tp_rank(monkeypatch, tp_size=2, global_rank=0)
    moe = LLaDA2TPMoE.from_config(_make_config(moe_gemm_impl="vllm"))

    assert moe.moe_gemm_impl == "vllm"


def test_llada2_attention_impl_is_configurable(monkeypatch):
    _mock_single_rank(monkeypatch)
    layer = LLaDA2DecoderLayer(_make_config(attn_impl="naive"), layer_idx=0)

    assert layer.attention.attn.attn_impl == "naive"


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


def test_llada2_moe_loads_score_correction_bias_alias(monkeypatch):
    _mock_single_rank(monkeypatch)
    model = LLaDA2ForDiffusionLM(_make_config())
    weight_name = "model.layers.1.mlp.gate.e_score_correction_bias"
    loaded = torch.arange(4, dtype=torch.float32)

    spec = resolve_weight_spec(
        model,
        weight_name,
        config=SimpleNamespace(),
    )

    assert spec is not None
    apply_resolved_weight(spec, loaded)
    assert torch.equal(model.model.layers[1].mlp.gate.expert_bias, loaded)


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


def test_llada2_dmax_topk_token_merge_applies_renormalize(monkeypatch):
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
                    [0.0, 0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0, 3.0],
                    [4.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
        )

    metadata = DMaxAttnMetaData()
    metadata.init_token_merging(
        merge_mask=torch.tensor([True]),
        topk_ids=torch.tensor([[2, 3]]),
        topk_probs=torch.tensor([[0.25, 0.50]]),
        residual_probs=torch.tensor([[0.25]]),
        mask_token_id=4,
        renormalize=True,
        mode="dmax_topk",
    )
    set_fetch_fn_for_attn_metadata(lambda: metadata)

    hidden = model.word_embeddings(torch.tensor([0]))
    merged = model._maybe_apply_token_merging(hidden)

    blended = (
        0.25 * model.word_embeddings.weight[2]
        + 0.50 * model.word_embeddings.weight[3]
        + 0.25 * model.word_embeddings.weight[4]
    )
    target_norm = 0.25 * model.word_embeddings.weight[2].norm()
    target_norm = target_norm + 0.50 * model.word_embeddings.weight[3].norm()
    target_norm = target_norm + 0.25 * model.word_embeddings.weight[4].norm()
    expected = blended * (target_norm / blended.norm())
    assert torch.allclose(merged[0], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("num_experts", "top_k", "n_group", "topk_group"),
    [
        (4, 2, 2, 1),
        (16, 4, 4, 2),
        (16, 4, 0, 0),
    ],
)
def test_llada2_group_limited_router_triton_matches_torch(
    num_experts: int,
    top_k: int,
    n_group: int,
    topk_group: int,
):
    torch.manual_seed(0)
    expert_bias = torch.randn(num_experts, device="cuda", dtype=torch.float32) * 0.1
    router = GroupLimitedTopKRouter(
        top_k=top_k,
        num_experts=num_experts,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=2.5,
        renormalize=True,
        expert_bias_getter=lambda: expert_bias,
    )
    router_logits = torch.randn(11, num_experts, device="cuda", dtype=torch.float32)

    actual = router(router_logits)
    expected = router._forward_naive(router_logits)

    actual_order = actual.ids.argsort(dim=-1)
    expected_order = expected.ids.argsort(dim=-1)
    actual_ids = actual.ids.gather(1, actual_order).to(torch.long)
    expected_ids = expected.ids.gather(1, expected_order).to(torch.long)
    actual_weights = actual.weights.gather(1, actual_order)
    expected_weights = expected.weights.gather(1, expected_order)

    assert torch.equal(actual_ids, expected_ids.to(actual_ids.dtype))
    assert torch.allclose(actual_weights.float(), expected_weights.float(), rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_llada2_router_matches_torch_at_mini_shape():
    torch.manual_seed(0)
    num_experts = 256
    top_k = 8
    expert_bias = torch.randn(num_experts, device="cuda", dtype=torch.float32) * 0.1
    router = GroupLimitedTopKRouter(
        top_k=top_k,
        num_experts=num_experts,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        renormalize=True,
        expert_bias_getter=lambda: expert_bias,
    )
    router_logits = torch.randn(64, num_experts, device="cuda", dtype=torch.float32)

    actual = router(router_logits)
    expected = router._forward_naive(router_logits)

    actual_order = actual.ids.argsort(dim=-1)
    expected_order = expected.ids.argsort(dim=-1)
    actual_ids = actual.ids.gather(1, actual_order).to(torch.long)
    expected_ids = expected.ids.gather(1, expected_order).to(torch.long)
    actual_weights = actual.weights.gather(1, actual_order)
    expected_weights = expected.weights.gather(1, expected_order)

    assert torch.equal(actual_ids, expected_ids.to(actual_ids.dtype))
    assert torch.allclose(actual_weights.float(), expected_weights.float(), rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_llada2_fused_grouped_topk_matches_naive_at_mini_shape():
    from diffulex_kernel import fused_grouped_topk

    torch.manual_seed(0)
    num_experts = 256
    top_k = 8
    expert_bias = torch.randn(num_experts, device="cuda", dtype=torch.float32) * 0.1
    router = GroupLimitedTopKRouter(
        top_k=top_k,
        num_experts=num_experts,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        renormalize=True,
        expert_bias_getter=lambda: expert_bias,
    )
    router_logits = torch.randn(64, num_experts, device="cuda", dtype=torch.float32)

    actual_weights, actual_ids = fused_grouped_topk(
        router_logits=router_logits,
        expert_bias=expert_bias,
        top_k=top_k,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        renormalize=True,
    )
    expected = router._forward_naive(router_logits)

    actual_order = actual_ids.argsort(dim=-1)
    expected_order = expected.ids.argsort(dim=-1)
    sorted_actual_ids = actual_ids.gather(1, actual_order).to(torch.long)
    sorted_expected_ids = expected.ids.gather(1, expected_order).to(torch.long)
    sorted_actual_weights = actual_weights.gather(1, actual_order)
    sorted_expected_weights = expected.weights.gather(1, expected_order)

    assert torch.equal(sorted_actual_ids, sorted_expected_ids.to(sorted_actual_ids.dtype))
    assert torch.allclose(sorted_actual_weights.float(), sorted_expected_weights.float(), rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_llada2_moe_triton_gemm_matches_naive_gemm_for_fixed_routing(monkeypatch):
    _mock_single_rank(monkeypatch)
    torch.manual_seed(0)
    moe = LLaDA2NaiveMoE.from_config(
        _make_config(
            hidden_size=64,
            moe_intermediate_size=128,
            num_experts=16,
            num_experts_per_tok=4,
            n_group=4,
            topk_group=2,
            num_shared_experts=0,
        )
    ).cuda()
    moe = moe.to(dtype=torch.bfloat16)
    with torch.no_grad():
        moe.w13.normal_(mean=0.0, std=0.02)
        moe.w2.normal_(mean=0.0, std=0.02)
    hidden_states = torch.randn((23, 64), device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.randint(0, moe.num_experts, (23, moe.top_k), device="cuda", dtype=torch.int32)
    topk_weights = torch.rand((23, moe.top_k), device="cuda", dtype=torch.float32)
    topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).to(torch.bfloat16)

    actual = moe.expert_gemm(
        impl="triton",
        hidden_states=hidden_states,
        w13=moe.w13,
        w2=moe.w2,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        local_expert_start=0,
        hidden_act=moe.hidden_act,
    )
    expected = moe.expert_gemm(
        impl="naive",
        hidden_states=hidden_states,
        w13=moe.w13,
        w2=moe.w2,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        local_expert_start=0,
        hidden_act=moe.hidden_act,
    )

    assert torch.allclose(actual.float(), expected.float(), rtol=3e-2, atol=5e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_llada2_moe_triton_gemm_matches_naive_gemm_for_tp_shard_routing(monkeypatch):
    _mock_tp_rank(monkeypatch, tp_size=4, global_rank=2)
    torch.manual_seed(0)
    moe = LLaDA2TPMoE.from_config(
        _make_config(
            hidden_size=64,
            moe_intermediate_size=128,
            num_experts=16,
            num_experts_per_tok=4,
            n_group=4,
            topk_group=2,
            num_shared_experts=0,
        )
    ).cuda()
    moe = moe.to(dtype=torch.bfloat16)
    with torch.no_grad():
        moe.w13.normal_(mean=0.0, std=0.02)
        moe.w2.normal_(mean=0.0, std=0.02)
    hidden_states = torch.randn((23, 64), device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.randint(0, moe.num_experts, (23, moe.top_k), device="cuda", dtype=torch.int32)
    topk_ids[: moe.num_local_experts, 0] = torch.arange(
        moe.local_expert_start,
        moe.local_expert_end,
        device="cuda",
        dtype=torch.int32,
    )
    topk_weights = torch.rand((23, moe.top_k), device="cuda", dtype=torch.float32)
    topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).to(torch.bfloat16)

    actual = moe.expert_gemm(
        impl="triton",
        hidden_states=hidden_states,
        w13=moe.w13,
        w2=moe.w2,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        local_expert_start=moe.local_expert_start,
        hidden_act=moe.hidden_act,
    )
    expected = moe.expert_gemm(
        impl="naive",
        hidden_states=hidden_states,
        w13=moe.w13,
        w2=moe.w2,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        local_expert_start=moe.local_expert_start,
        hidden_act=moe.hidden_act,
    )

    assert torch.allclose(actual.float(), expected.float(), rtol=3e-2, atol=5e-1)
