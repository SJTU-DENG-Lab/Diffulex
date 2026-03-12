"""
Basic tests for quantization extension.

Run with: python -m diffulex.extensions.quantization.test_basic
"""

import torch
import sys

def test_import():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    from . import (
        enable, disable, is_enabled,
        QuantizationConfig, QuantizationContext,
        LinearQuantizationStrategy, KVCacheQuantizationStrategy,
        create_linear_strategy, create_kv_cache_strategy,
        LinearQuantizationMixin,
    )
    
    print("✓ All imports successful")
    return True


def test_config():
    """Test configuration classes."""
    print("\nTesting configuration...")
    
    from .config import QuantizationConfig, KVCacheQuantConfig, WeightQuantConfig
    
    # Default config
    config = QuantizationConfig()
    assert config.kv_cache.dtype == "bf16"
    assert config.weights.method == "bf16"
    
    # FP8 config
    config = QuantizationConfig(
        kv_cache=KVCacheQuantConfig(dtype="fp8_e4m3"),
        weights=WeightQuantConfig(method="fp8_w8a8")
    )
    assert config.kv_cache.dtype == "fp8_e4m3"
    assert config.weights.method == "fp8_w8a8"
    
    print("✓ Configuration tests passed")
    return True


def test_context():
    """Test quantization context."""
    print("\nTesting context...")
    
    from .context import QuantizationContext, get_context, clear_act_quant_cache
    
    # Get context
    ctx = get_context()
    assert ctx is not None
    
    # Test strategy storage
    dummy_strategy = object()
    ctx.set_strategy("test", dummy_strategy)
    assert ctx.get_strategy("test") is dummy_strategy
    
    # Test activation cache
    x = torch.randn(10, 10)
    q_x = torch.randn(10, 10)
    scale = torch.tensor(1.0)
    
    ctx.set_cached_act_quant(x, q_x, scale)
    cached = ctx.get_cached_act_quant(x)
    assert cached is not None
    assert torch.equal(cached[0], q_x)
    
    # Clear cache
    clear_act_quant_cache()
    cached = ctx.get_cached_act_quant(x)
    assert cached is None
    
    print("✓ Context tests passed")
    return True


def test_strategies():
    """Test strategy registration and creation."""
    print("\nTesting strategies...")
    
    # Import strategies to register them
    from . import strategies  # noqa: F401
    
    from .registry import (
        create_linear_strategy, create_kv_cache_strategy,
        registered_linear_strategies, registered_kv_cache_dtypes,
    )
    
    # Check registered strategies
    linear_strategies = registered_linear_strategies()
    kv_strategies = registered_kv_cache_dtypes()
    
    print(f"  Registered linear strategies: {len(linear_strategies)}")
    print(f"  Registered KV cache strategies: {kv_strategies}")
    
    # Test BF16 strategies
    bf16_linear = create_linear_strategy("bf16", "bf16")
    assert bf16_linear is not None
    assert bf16_linear.name == "bf16_linear"
    
    bf16_kv = create_kv_cache_strategy("bf16")
    assert bf16_kv is not None
    
    print("✓ Strategy tests passed")
    return True


def test_fp8_strategy():
    """Test FP8 strategy."""
    print("\nTesting FP8 strategy...")
    
    from .registry import create_linear_strategy, create_kv_cache_strategy
    from .kernels import check_vllm_op_available
    
    # FP8 KV cache
    fp8_kv = create_kv_cache_strategy("fp8_e4m3")
    assert fp8_kv is not None
    assert fp8_kv.requires_kv_cache_scales
    
    # FP8 W8A8
    fp8_linear = create_linear_strategy("fp8_e4m3", "fp8_e4m3")
    assert fp8_linear is not None
    assert "fp8" in fp8_linear.linear_weight_format
    
    # Test quantization (skip if vLLM ops not available)
    if check_vllm_op_available('scaled_fp8_quant'):
        x = torch.randn(10, 10, dtype=torch.bfloat16)
        q_x, scale = fp8_linear.quantize(x)
        assert q_x.dtype == torch.float8_e4m3fn
    else:
        print("  (skipped FP8 quantize test - vLLM ops not available)")
    
    print("✓ FP8 strategy tests passed")
    return True


def test_int8_strategy():
    """Test INT8 strategy."""
    print("\nTesting INT8 strategy...")
    
    from .registry import create_linear_strategy
    
    # INT8 W8A8
    int8_linear = create_linear_strategy("int8", "int8")
    assert int8_linear is not None
    assert int8_linear.linear_weight_format == "int8"
    
    # Test quantization
    x = torch.randn(10, 10, dtype=torch.bfloat16)
    q_x, scale = int8_linear.quantize(x)
    assert q_x.dtype == torch.int8
    
    print("✓ INT8 strategy tests passed")
    return True


def test_layer_mixin():
    """Test layer mixin."""
    print("\nTesting layer mixin...")
    
    from .layer_mixin import LinearQuantizationMixin
    
    # Create a simple test class
    class TestLinear(torch.nn.Linear, LinearQuantizationMixin):
        def __init__(self, in_features, out_features):
            super().__init__(in_features, out_features, dtype=torch.bfloat16)
            self.init_quantization(quant_kind="other")
        
        def forward(self, x):
            return self._forward_base(x, self.bias)
    
    # Create instance
    layer = TestLinear(10, 20)
    
    # Test forward
    x = torch.randn(5, 10, dtype=torch.bfloat16)
    y = layer(x)
    assert y.shape == (5, 20)
    
    # Test quantization methods
    assert not layer.has_quantized_weight()
    assert not layer.has_offline_quantized_weight()
    
    print("✓ Layer mixin tests passed")
    return True


def test_forward_plans():
    """Test forward plans."""
    print("\nTesting forward plans...")
    
    from .linear_plans import ForwardPlanSig, BF16Plan
    
    # Create a simple mock layer
    class MockLayer:
        def __init__(self):
            self.weight = torch.randn(20, 10, dtype=torch.bfloat16)
    
    layer = MockLayer()
    x = torch.randn(5, 10, dtype=torch.bfloat16)
    
    # Create plan signature
    sig = ForwardPlanSig(
        device_type="cuda",
        device_index=0,
        x_dtype=torch.bfloat16,
        x_shape=(5, 10),
        has_bias=False,
        mode="bf16",
        strategy_name="bf16"
    )
    
    # Create plan
    plan = BF16Plan(sig, layer.weight, None)
    y = plan(x)
    assert y.shape == (5, 20)
    
    print("✓ Forward plan tests passed")
    return True


def test_enable_disable():
    """Test enable/disable functionality."""
    print("\nTesting enable/disable...")
    
    from . import enable, disable, is_enabled
    from .config import QuantizationConfig, KVCacheQuantConfig, WeightQuantConfig
    
    # Initially disabled
    assert not is_enabled()
    
    # Enable with explicit config
    config = QuantizationConfig(
        kv_cache=KVCacheQuantConfig(dtype="bf16"),
        weights=WeightQuantConfig(method="bf16")
    )
    result = enable(config={
        'kv_cache': {'dtype': 'bf16'},
        'weights': {'method': 'bf16'},
        'activations': {}
    })
    assert result
    assert is_enabled()
    
    # Disable
    disable()
    assert not is_enabled()
    
    print("✓ Enable/disable tests passed")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Quantization Extension Tests")
    print("=" * 60)
    
    tests = [
        test_import,
        test_config,
        test_context,
        test_strategies,
        test_fp8_strategy,
        test_int8_strategy,
        test_layer_mixin,
        test_forward_plans,
        test_enable_disable,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
