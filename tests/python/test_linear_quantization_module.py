import pytest


def test_linear_strategy_registry_bf16_pair():
    from diffulex.utils.quantization.registry import create_linear_strategy

    s = create_linear_strategy(weight_dtype="bf16", act_dtype="bf16")
    assert s.linear_weight_format == "bf16"
    assert s.linear_act_format == "bf16"


def test_linear_strategy_registry_int8_w8a16():
    """Test that int8+bf16 returns the real W8A16 strategy (not stub)."""
    from diffulex.utils.quantization.registry import create_linear_strategy

    s = create_linear_strategy(weight_dtype="int8", act_dtype="bf16")
    assert s.name == "linear_int8_w8a16"
    assert s.linear_weight_format == "int8"
    assert s.linear_act_format == "bf16"


def test_linear_strategy_registry_non_bf16_returns_stub():
    """Test that unimplemented combinations (e.g., int4) return stub."""
    from diffulex.utils.quantization.registry import create_linear_strategy

    s = create_linear_strategy(weight_dtype="int4", act_dtype="bf16")
    assert s.name.startswith("linear_stub")
    assert s.linear_weight_format == "int4"
    assert s.linear_act_format == "bf16"


def test_factory_injects_linear_strategies_into_context():
    from dataclasses import dataclass

    from diffulex.utils.quantization.factory import QuantizationStrategyFactory
    from diffulex.utils.quantization.context import get_quantization_context

    @dataclass
    class DummyConfig:
        kv_cache_dtype: str = "bf16"
        attn_q_dtype: str = "bf16"
        linear_attn_weight_dtype: str = "bf16"
        linear_mlp_weight_dtype: str = "bf16"
        linear_attn_act_dtype: str = "bf16"
        linear_mlp_act_dtype: str = "bf16"

    ctx = QuantizationStrategyFactory.create_from_config(DummyConfig())
    assert ctx is get_quantization_context()
    assert ctx.get_linear_strategy("attn") is not None
    assert ctx.get_linear_strategy("mlp") is not None


def test_linear_forward_raises_on_stub(monkeypatch):
    # Avoid requiring torch.distributed process group init in unit tests.
    import torch
    import torch.nn.functional as F
    import torch.distributed as dist

    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)

    from diffulex.layer.linear import ColumnParallelLinear
    from diffulex.utils.quantization.registry import create_linear_strategy
    from diffulex.utils.quantization.context import get_quantization_context

    # Install a stub strategy for attention linears (use int4, not implemented yet).
    ctx = get_quantization_context()
    ctx.set_linear_strategy("attn", create_linear_strategy(weight_dtype="int4", act_dtype="bf16"))

    lin = ColumnParallelLinear(4, 8, bias=False, quant_kind="attn")
    # NOTE: default Linear weights are float32 unless a checkpoint loader overwrites them.
    # Keep dtypes consistent for this unit test.
    x = torch.randn(2, 4, dtype=torch.float32)

    with pytest.raises(NotImplementedError):
        _ = lin(x)

    # Ensure bf16 path still works for other kinds.
    lin2 = ColumnParallelLinear(4, 8, bias=False, quant_kind="other")
    y = lin2(x)
    ref = F.linear(x, lin2.weight, None)
    assert torch.allclose(y, ref)


def test_linear_int8_w8a16_quantization():
    """Test that int8+bf16 strategy correctly quantizes and dequantizes weights."""
    from diffulex.utils.quantization.registry import create_linear_strategy
    import torch

    strategy = create_linear_strategy(weight_dtype="int8", act_dtype="bf16")
    assert strategy.name == "linear_int8_w8a16"
    assert strategy.linear_weight_format == "int8"
    assert strategy.linear_act_format == "bf16"

    # Test quantization/dequantization
    weight = torch.randn(8, 4, dtype=torch.bfloat16)
    quantized, scales = strategy.quantize(weight)
    assert quantized.dtype == torch.int8
    assert quantized.shape == weight.shape
    assert scales.shape == (weight.shape[0],)  # Per-output-channel scales

    dequantized = strategy.dequantize(quantized, scales)
    assert dequantized.dtype == torch.bfloat16
    assert dequantized.shape == weight.shape

    # Quantization error should be reasonable (int8 quantization introduces error)
    error = (weight - dequantized).abs().max()
    assert error.item() < 0.1, f"Quantization error too large: {error.item()}"


def test_linear_int8_w8a16_forward():
    """Test that int8+bf16 strategy's linear_forward produces reasonable outputs."""
    from diffulex.utils.quantization.registry import create_linear_strategy
    import torch
    import torch.nn.functional as F

    strategy = create_linear_strategy(weight_dtype="int8", act_dtype="bf16")

    x = torch.randn(2, 4, dtype=torch.bfloat16)
    weight = torch.randn(8, 4, dtype=torch.bfloat16)
    bias = torch.randn(8, dtype=torch.bfloat16)

    # Forward with quantized strategy
    y_quant = strategy.linear_forward(x, weight, bias, quant_kind="test")

    # Reference forward (should be close but not exact due to quantization)
    y_ref = F.linear(x, weight, bias)

    assert y_quant.shape == y_ref.shape
    assert y_quant.dtype == torch.bfloat16

    # Error should be reasonable (quantization introduces some error)
    error = (y_quant - y_ref).abs().max()
    assert error.item() < 0.5, f"Forward error too large: {error.item()}"


