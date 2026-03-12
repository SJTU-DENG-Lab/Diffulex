"""
Diffulex Quantization Extension

Zero-coupling quantization support for Diffulex.
Enable before importing diffulex:

    from diffulex.extensions import quantization
    quantization.enable()
    
    # Then use diffulex normally
    from diffulex import Config, LLMEngine

Available quantization formats:
- FP8 W8A8: FP8 weights + FP8 activations
- FP8 W8A16: FP8 weights + BF16 activations
- INT8 W8A8: INT8 weights + INT8 activations
- INT8 W8A16: INT8 weights + BF16 activations
- GPTQ W4A16: 4-bit GPTQ quantized weights
- AWQ W4A16: 4-bit AWQ quantized weights
- GPTQ/AWQ + Marlin: Optimized kernels for above
- FP8 KV Cache: FP8 quantized KV cache
- Custom Triton Kernels: On-the-fly dequantization

Example usage:
    # FP8 W8A8 with FP8 KV Cache
    quantization.enable(
        weight_quant_method="fp8_w8a8",
        kv_cache_dtype="fp8_e4m3"
    )
    
    # GPTQ 4-bit model
    quantization.enable(
        weight_quant_method="gptq_w4a16",
        group_size=128
    )
    
    # Mixed quantization: FP8 attention, BF16 MLP
    quantization.enable(
        linear_attn_dtype="fp8_e4m3",
        linear_mlp_dtype="bf16",
        kv_cache_dtype="fp8_e4m3"
    )
"""

__version__ = "1.0.0"

# Main API
from .bootstrap import (
    enable,
    disable,
    is_enabled,
    get_config,
    configure_from_args,
    auto_enable_from_config,
)

# Kernels package (unified interface)
from .kernels import (
    # Registry
    KernelRegistry,
    register_kernel,
    get_kernel,
    list_available_kernels,
    # Availability
    check_vllm_op_available,
    check_kernel_available,
    get_kernel_status,
    print_kernel_status,
    set_strict_mode,
    is_strict_mode,
    warn_kernel_unavailable,
    # vLLM wrappers
    VllmGPTQGemm,
    VllmAWQGemm,
    VllmMarlinGemm,
    VllmCutlassScaledMM,
    VllmAllSparkW8A16,
    VllmCutlassW4A8,
    VllmFp8LinearOp,
    # Triton kernels
    Fp8KVAttentionKernel,
    fp8_kv_attention_forward,
    _HAS_TRITON_KERNELS,
)

# Configuration
from .config import (
    QuantizationConfig,
    KVCacheQuantConfig,
    WeightQuantConfig,
    ActivationQuantConfig,
)

# Context
from .context import (
    QuantizationContext,
    get_context,
    set_linear_strategy,
    get_linear_strategy,
    set_kv_cache_strategy,
    get_kv_cache_strategy,
    clear_act_quant_cache,
    step_end_cleanup,
)

# Registry and Factory
from .registry import (
    register_linear_strategy,
    register_kv_cache_strategy,
    create_linear_strategy,
    create_kv_cache_strategy,
    registered_linear_strategies,
    registered_kv_cache_dtypes,
    QuantizationStrategyFactory,
)

# Concrete strategies (for advanced usage)
from .strategies.kv_cache_bf16 import BF16KVCacheStrategy
from .strategies.linear_bf16 import BF16LinearStrategy
from .strategies.linear_fp8_w8a8 import FP8E4M3W8A8LinearStrategy, FP8E5M2W8A8LinearStrategy
from .strategies.linear_fp8_w8a16 import FP8E4M3W8A16LinearStrategy, FP8E5M2W8A16LinearStrategy
from .strategies.linear_int8_w8a8 import INT8W8A8LinearStrategy
from .strategies.linear_int8_w8a16 import INT8W8A16LinearStrategy
from .strategies.linear_gptq_wxa16 import (
    GPTQW2A16LinearStrategy,
    GPTQW3A16LinearStrategy,
    GPTQW4A16LinearStrategy,
    GPTQW8A16LinearStrategy,
)
from .strategies.linear_gptq_marlin_w4a16 import GPTQMarlinW4A16LinearStrategy
from .strategies.linear_gptq_marlin_w8a16 import GPTQMarlinW8A16LinearStrategy
from .strategies.linear_awq_w4a16 import AWQW4A16LinearStrategy
from .strategies.linear_awq_marlin_w4a16 import AWQMarlinW4A16LinearStrategy

# Strategy base classes
from .strategy import (
    QuantizationStrategy,
    KVCacheQuantizationStrategy,
    LinearQuantizationStrategy,
    WeightQuantizationStrategy,
)

# Layer Mixin and Patch
from .layer_mixin import LinearQuantizationMixin
from .layer_patch import (
    patch_linear_layers,
    unpatch_linear_layers,
    is_patched,
    create_quantized_layer,
)

# Forward Plans
from .linear_plans import (
    ForwardPlanBase,
    ForwardPlanSig,
    BF16Plan,
    QuantizedLinearPlan,
    QuantInt8W8A16Plan,
    QuantInt8W8A8Plan,
    QuantFP8W8A8Plan,
    QuantFP8W8A16Plan,
    OfflineGPTQPlan,
    OfflineAWQPlan,
    DirectGPTQGemmPlan,
    DirectAWQGemmPlan,
    DirectMarlinGemmPlan,
)
from .linear_plan_builder import build_forward_plan, rebuild_plan_if_needed

# Offline quantization
from .quantize_model import quantize_model

__all__ = [
    # Bootstrap
    "enable",
    "disable",
    "is_enabled",
    "get_config",
    "configure_from_args",
    "auto_enable_from_config",
    
    # Kernels
    "KernelRegistry",
    "register_kernel",
    "get_kernel",
    "list_available_kernels",
    "check_vllm_op_available",
    "check_kernel_available",
    "get_kernel_status",
    "print_kernel_status",
    "set_strict_mode",
    "is_strict_mode",
    "warn_kernel_unavailable",
    "VllmGPTQGemm",
    "VllmAWQGemm",
    "VllmMarlinGemm",
    "VllmCutlassScaledMM",
    "VllmAllSparkW8A16",
    "VllmCutlassW4A8",
    "VllmFp8LinearOp",
    "Fp8KVAttentionKernel",
    "fp8_kv_attention_forward",
    "_HAS_TRITON_KERNELS",
    
    # Configuration
    "QuantizationConfig",
    "KVCacheQuantConfig",
    "WeightQuantConfig",
    "ActivationQuantConfig",
    
    # Concrete strategies
    "BF16KVCacheStrategy",
    "BF16LinearStrategy",
    "FP8W8A8LinearStrategy",
    "FP8W8A16LinearStrategy",
    "INT8W8A8LinearStrategy",
    "INT8W8A16LinearStrategy",
    "GPTQW2A16LinearStrategy",
    "GPTQW3A16LinearStrategy",
    "GPTQW4A16LinearStrategy",
    "GPTQW8A16LinearStrategy",
    "GPTQMarlinW4A16LinearStrategy",
    "GPTQMarlinW8A16LinearStrategy",
    "AWQW4A16LinearStrategy",
    "AWQMarlinW4A16LinearStrategy",
    
    # Context
    "QuantizationContext",
    "get_context",
    "set_linear_strategy",
    "get_linear_strategy",
    "set_kv_cache_strategy",
    "get_kv_cache_strategy",
    "clear_act_quant_cache",
    "step_end_cleanup",
    
    # Registry
    "register_linear_strategy",
    "register_kv_cache_strategy",
    "create_linear_strategy",
    "create_kv_cache_strategy",
    "registered_linear_strategies",
    "registered_kv_cache_dtypes",
    "QuantizationStrategyFactory",
    
    # Strategy
    "QuantizationStrategy",
    "KVCacheQuantizationStrategy",
    "LinearQuantizationStrategy",
    "WeightQuantizationStrategy",
    
    # Layer
    "LinearQuantizationMixin",
    "patch_linear_layers",
    "unpatch_linear_layers",
    "is_patched",
    "create_quantized_layer",
    
    # Forward Plans
    "ForwardPlanBase",
    "ForwardPlanSig",
    "BF16Plan",
    "QuantizedLinearPlan",
    "QuantInt8W8A16Plan",
    "QuantInt8W8A8Plan",
    "QuantFP8W8A8Plan",
    "QuantFP8W8A16Plan",
    "OfflineGPTQPlan",
    "OfflineAWQPlan",
    "DirectGPTQGemmPlan",
    "DirectAWQGemmPlan",
    "DirectMarlinGemmPlan",
    "build_forward_plan",
    "rebuild_plan_if_needed",
    
    # Offline quantization
    "quantize_model",
]
