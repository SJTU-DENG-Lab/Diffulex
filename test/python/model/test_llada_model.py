from types import SimpleNamespace

from diffulex.model.llada import LLaDABlock
from diffulex.utils import parallelism


def _mock_single_rank():
    parallelism.reset_model_parallelism_metadata()
    parallelism._MODEL_PARALLELISM_METADATA = parallelism.ModelParallelismMetadata.from_world(
        tp_size=1,
        ep_size=1,
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

    assert block.self_attn.q_proj.bias is None
    assert block.self_attn.k_proj.bias is None
    assert block.self_attn.v_proj.bias is None


def test_llada_block_respects_include_qkv_bias_true():
    _mock_single_rank()
    block = LLaDABlock(_make_config(include_qkv_bias=True))

    assert block.self_attn.q_proj.bias is not None
    assert block.self_attn.k_proj.bias is not None
    assert block.self_attn.v_proj.bias is not None
