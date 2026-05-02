from types import SimpleNamespace

import diffulex.distributed.parallel_state as parallel_state
from diffulex.model.llada import LLaDABlock


def _mock_single_rank():
    parallel_state.reset_parallel_state()
    parallel_state.PARALLEL_STATE = parallel_state.build_parallel_state_for_test(
        tp_size=1,
        ep_size=1,
        dp_size=1,
        world_size=1,
        global_rank=0,
    )


def _make_config(**overrides):
    config = dict(
        hidden_size=16,
        num_attention_heads=4,
        n_kv_heads=4,
        max_sequence_length=32,
        rms_norm_eps=1e-5,
        head_dim=4,
        rope_theta=500000.0,
        rope_scaling=None,
        mlp_hidden_size=32,
        activation_type="silu",
        include_qkv_bias=False,
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def test_llada_block_respects_include_qkv_bias_false():
    _mock_single_rank()
    block = LLaDABlock(_make_config(include_qkv_bias=False))

    assert block.self_attn.qkv_proj.bias is None


def test_llada_block_respects_include_qkv_bias_true():
    _mock_single_rank()
    block = LLaDABlock(_make_config(include_qkv_bias=True))

    assert block.self_attn.qkv_proj.bias is not None
