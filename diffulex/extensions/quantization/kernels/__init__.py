"""
Quantization Kernels Package

Unified interface for all quantization kernels:
- vLLM optimized kernels (CUTLASS, Marlin, GPTQ, AWQ)
- Custom Triton kernels for specialized operations
- Kernel availability checking and management
"""

from .kernel_registry import (
    KernelRegistry,
    register_kernel,
    get_kernel,
    list_available_kernels,
)

from .kernel_availability import (
    check_vllm_op_available,
    check_kernel_available,
    get_kernel_status,
    print_kernel_status,
    set_strict_mode,
    is_strict_mode,
    warn_kernel_unavailable,
)

# Import kernel wrappers
from .vllm_kernels import (
    VllmGPTQGemm,
    VllmAWQGemm,
    VllmMarlinGemm,
    VllmCutlassScaledMM,
    VllmAllSparkW8A16,
    VllmCutlassW4A8,
    VllmFp8LinearOp,
)

# Import custom Triton kernels
try:
    from .triton_kernels import (
        chunked_prefill_attn_unified_fp8,
        _HAS_FP8_UNIFIED_KERNEL,
    )
    _HAS_TRITON_KERNELS = True
except ImportError:
    _HAS_TRITON_KERNELS = False
    chunked_prefill_attn_unified_fp8 = None

__all__ = [
    # Registry
    "KernelRegistry",
    "register_kernel",
    "get_kernel",
    "list_available_kernels",
    # Availability
    "check_vllm_op_available",
    "check_kernel_available",
    "get_kernel_status",
    "print_kernel_status",
    "set_strict_mode",
    "is_strict_mode",
    "warn_kernel_unavailable",
    # vLLM wrappers
    "VllmGPTQGemm",
    "VllmAWQGemm",
    "VllmMarlinGemm",
    "VllmCutlassScaledMM",
    "VllmAllSparkW8A16",
    "VllmCutlassW4A8",
    "VllmFp8LinearOp",
    # Triton kernels
    "chunked_prefill_attn_unified_fp8",
    "_HAS_TRITON_KERNELS",
]
