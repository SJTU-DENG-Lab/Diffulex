# Diffulex Quantization Extension

Zero-coupling quantization support for Diffulex. This extension adds support for various quantization formats without modifying any original Diffulex source code.

## Features

### Supported Quantization Formats

**Online Quantization:**
- FP8 W8A8: FP8 weights + FP8 activations
- FP8 W8A16: FP8 weights + BF16 activations
- INT8 W8A8: INT8 weights + INT8 activations
- INT8 W8A16: INT8 weights + BF16 activations

**Offline Quantization:**
- GPTQ W4A16: 4-bit GPTQ quantized models
- AWQ W4A16: 4-bit AWQ quantized models
- GPTQ + Marlin: Optimized kernels for GPTQ
- AWQ + Marlin: Optimized kernels for AWQ
- CUTLASS W4A8: 4-bit weights + FP8 activations (Hopper+)

**GPU-Specific Kernels:**
- **CUTLASS W4A8**: Requires Hopper GPU (SM90+) for optimal performance
- **AllSpark W8A16**: Optimized for Ampere+ GPUs (SM80+)
- **Marlin**: Efficient 4-bit GEMM for various GPU architectures

**KV Cache Quantization:**
- FP8 E4M3: 8-bit KV cache with E4M3 format
- FP8 E5M2: 8-bit KV cache with E5M2 format
- **Custom FP8 Triton Kernel**: On-the-fly dequantization during attention
- BF16: No quantization (default)

## Installation

No additional installation needed. The extension is part of diffulex.

## Usage

### Basic Usage

```python
from diffulex.extensions import quantization

# Enable quantization BEFORE importing diffulex
quantization.enable(
    weight_quant_method="fp8_w8a8",
    kv_cache_dtype="fp8_e4m3"
)

# Now use diffulex normally
from diffulex import Config, LLMEngine

config = Config(model="your-model")
engine = LLMEngine(config)
```

### GPTQ Model Loading

```python
from diffulex.extensions import quantization

# Enable GPTQ quantization
quantization.enable(
    weight_quant_method="gptq_w4a16",
    group_size=128
)

from diffulex import Config, LLMEngine

# Load GPTQ model - weights will be automatically detected and loaded
config = Config(model="path/to/gptq-model")
engine = LLMEngine(config)
```

### CUTLASS W4A8 (Hopper GPUs)

```python
from diffulex.extensions import quantization

# Check kernel availability first
quantization.print_kernel_status()

# Enable CUTLASS W4A8 (4-bit weights + FP8 activations)
# Requires Hopper GPU (SM90+) for optimal performance
quantization.enable(
    weight_quant_method="cutlass_w4a8",
    group_size=128  # Must be 128
)

from diffulex import Config, LLMEngine

config = Config(model="your-model")
engine = LLMEngine(config)
```

### Mixed Quantization

```python
from diffulex.extensions import quantization

# Use FP8 for attention, BF16 for MLP
quantization.enable(
    linear_attn_dtype="fp8_e4m3",
    linear_mlp_dtype="bf16",
    kv_cache_dtype="fp8_e4m3"
)
```

### Advanced Configuration

```python
from diffulex.extensions import quantization
from diffulex.extensions.quantization import (
    QuantizationConfig,
    KVCacheQuantConfig,
    WeightQuantConfig,
)

# Create detailed configuration
config = QuantizationConfig(
    kv_cache=KVCacheQuantConfig(dtype="fp8_e4m3"),
    weights=WeightQuantConfig(
        method="int8_w8a8",
        group_size=128,
        linear_attn_dtype="int8",
        linear_mlp_dtype="bf16"
    )
)

# Enable with config dict
quantization.enable(config={
    'kv_cache': {'dtype': 'fp8_e4m3'},
    'weights': {
        'method': 'int8_w8a8',
        'group_size': 128,
        'linear_attn_dtype': 'int8',
        'linear_mlp_dtype': 'bf16'
    },
    'activations': {}
})
```

## API Reference

### Main Functions

#### `quantization.enable()`

Enable quantization extension. Must be called before importing diffulex.

**Parameters:**
- `config`: Full configuration dict (optional)
- `kv_cache_dtype`: KV cache dtype ("bf16", "fp8_e4m3", "fp8_e5m2")
- `weight_quant_method`: Weight quantization method
- `linear_attn_dtype`: Attention layer dtype override
- `linear_mlp_dtype`: MLP layer dtype override
- `group_size`: Quantization group size for GPTQ/AWQ
- `desc_act`: GPTQ desc_act flag

#### `quantization.disable()`

Disable quantization and restore original classes.

#### `quantization.is_enabled()`

Check if quantization is currently enabled.

### Configuration Classes

#### `QuantizationConfig`

Top-level configuration container.

```python
from diffulex.extensions.quantization import QuantizationConfig

config = QuantizationConfig(
    kv_cache=KVCacheQuantConfig(dtype="fp8_e4m3"),
    weights=WeightQuantConfig(method="fp8_w8a8"),
    activations=ActivationQuantConfig()
)
```

### Custom Strategies

You can register custom quantization strategies:

```python
from diffulex.extensions.quantization import (
    register_linear_strategy,
    LinearQuantizationStrategy,
)

@register_linear_strategy("custom", "bf16")
class CustomLinearStrategy(LinearQuantizationStrategy):
    @property
    def name(self):
        return "custom_strategy"
    
    def linear_forward(self, x, weight, bias, *, quant_kind, **kwargs):
        # Your custom implementation
        return output
```

## Architecture

### Directory Structure

```
diffulex/extensions/quantization/
├── __init__.py                 # Main API exports
├── bootstrap.py                # Extension initialization
├── config.py                   # Configuration classes
├── context.py                  # Strategy context management
├── registry.py                 # Strategy registry
├── strategy.py                 # Base strategy classes
├── layer_patch.py              # Layer monkey patching
├── layer_mixin.py              # Quantized layer mixin
├── kv_cache_patch.py           # KV cache quantization
├── linear_plans.py             # Forward plan definitions
├── linear_plan_builder.py      # Plan builder
├── loader_patch.py             # Weight loader patching
├── test_basic.py               # Basic tests
├── README.md                   # This documentation
├── kernels/                    # Kernel implementations
│   ├── __init__.py
│   ├── kernel_registry.py      # Kernel registry & base classes
│   ├── kernel_availability.py  # Availability checking & warnings
│   ├── vllm_kernels.py         # vLLM kernel wrappers
│   └── triton_kernels/         # Custom Triton kernels
│       ├── __init__.py
│       └── fp8_kv_attention.py # FP8 KV attention kernel
└── strategies/                 # Quantization strategies
    ├── __init__.py
    ├── kv_cache_bf16.py
    ├── kv_cache_fp8_running_max.py
    ├── linear_bf16.py
    ├── linear_fp8_w8a8.py
    ├── linear_fp8_w8a16.py
    ├── linear_int8_w8a8.py
    ├── linear_int8_w8a16.py
    ├── linear_gptq_w*.py
    ├── linear_awq_*.py
    └── linear_w4a8_cutlass.py
```

### Zero-Coupling Design

This extension uses a **zero-coupling architecture** that ensures:

1. **No modifications** to `diffulex/*.py` or `diffulex/**/*.py`
2. **Monkey Patching** for layer class replacement
3. **Dynamic attribute injection** for KV cache extension
4. **Wrapper functions** for loader extension

### Extension Points

1. **Config Extension**: Dynamic attributes on Config objects
2. **Layer Monkey Patch**: Replaces Linear classes with quantized versions
3. **KV Cache Patch**: Wraps KV cache allocation and access
4. **Loader Patch**: Intercepts weight loading for offline formats
5. **Bootstrap**: Import hooks for seamless integration

### Forward Plan Caching

The extension implements **Forward Plan caching** to minimize Python overhead:

- Plans are cached per-layer based on input signature
- Automatic validation and rebuilding when input changes
- Direct GEMM paths bypass Python strategy for maximum performance

## Testing

Run the test suite:

```bash
python -m diffulex.extensions.quantization.test_basic
```

## Advanced Features

### Custom FP8 KV Cache Triton Kernel

The extension includes a custom Triton kernel for FP8 KV cache attention that performs on-the-fly dequantization, avoiding explicit dequantize-copy operations:

```python
from diffulex.extensions import quantization

# Check if Triton kernel is available
if quantization._HAS_FP8_TRITON_KERNEL:
    print("FP8 Triton kernel available")

# Enable FP8 KV cache
quantization.enable(kv_cache_dtype="fp8_e4m3")

# The kernel will be automatically used for attention computation
```

**Benefits:**
- On-the-fly dequantization in Triton kernel
- Reduces memory bandwidth by ~50% for KV cache
- Faster than explicit dequantize + attention

**Requirements:**
- Triton >= 2.0
- CUDA-capable GPU

## Troubleshooting

### "Cannot import name 'enable'"

Make sure to import from the correct path:

```python
from diffulex.extensions import quantization
quantization.enable()
```

### "Unknown linear strategy"

Check that the strategy name is correct:

```python
from diffulex.extensions.quantization import registered_linear_strategies
print(registered_linear_strategies())
```

### Kernel Availability Warnings

When optimized kernels (e.g., vLLM's CUDA ops) are unavailable, the extension will:
1. **Warn once** with a `RuntimeWarning` about the fallback
2. Continue with slower fallback implementation

To check kernel availability before running:
```python
from diffulex.extensions import quantization

# Print kernel status
quantization.print_kernel_status()

# Check specific kernel
if quantization.check_vllm_op_available('gptq_gemm'):
    print("GPTQ kernel available")

# Enable strict mode (raise error instead of fallback)
quantization.set_strict_mode(True)
# Or set environment variable: DIFFULEX_QUANT_STRICT=1
```

**Available Kernels:**
- `gptq_gemm`: GPTQ quantization (W2/W3/W4/W8)
- `gptq_marlin_gemm`: GPTQ Marlin format
- `awq_gemm`: AWQ quantization
- `cutlass_scaled_mm`: INT8/Cutlass quantization
- `scaled_fp8_quant`: FP8 quantization

### CUDA errors with quantized models

Ensure:
1. CUDA is available: `torch.cuda.is_available()`
2. vLLM is installed with CUDA support: `pip install vllm`
3. Model weights are on the correct device

## Performance Notes

- **FP8 W8A8**: ~2x memory reduction, minimal speed impact on Hopper+
- **INT8 W8A8**: ~2x memory reduction, good speedup on Ampere+
- **GPTQ/AWQ W4A16**: ~4x memory reduction, optimized kernels available
- **FP8 KV Cache**: ~2x KV cache memory reduction

## License

Same as Diffulex.
