"""
Kernel Availability Checker

Tracks availability of optimized kernels and warns when falling back.
"""

import warnings
import os
from typing import Set, Optional

# Track which warnings have been issued to avoid spamming
_issued_warnings: Set[str] = set()

# Global flag: if True, raise error instead of fallback when kernel is unavailable
_STRICT_MODE = os.environ.get('DIFFULEX_QUANT_STRICT', '0') == '1'


def set_strict_mode(enabled: bool = True):
    """Set strict mode: raise error when optimized kernels are unavailable."""
    global _STRICT_MODE
    _STRICT_MODE = enabled


def is_strict_mode() -> bool:
    """Check if strict mode is enabled."""
    return _STRICT_MODE


def check_vllm_op_available(op_name: str) -> bool:
    """Check if a vLLM custom op is available."""
    try:
        import vllm._custom_ops as ops
        return hasattr(ops, op_name)
    except (ImportError, AttributeError):
        return False


def check_kernel_available(kernel_name: str, op_checker: Optional[callable] = None) -> bool:
    """
    Check if a kernel is available.
    
    Args:
        kernel_name: Name of the kernel
        op_checker: Optional function to check availability
        
    Returns:
        True if available
    """
    if op_checker is not None:
        return op_checker()
    
    # Default: check vLLM ops
    return check_vllm_op_available(kernel_name)


def warn_kernel_unavailable(
    kernel_name: str,
    strategy_name: str,
    fallback_desc: str = "slow fallback"
) -> None:
    """
    Warn once when an optimized kernel is unavailable.
    
    Args:
        kernel_name: Name of the missing kernel (e.g., "vllm.gptq_gemm")
        strategy_name: Name of the quantization strategy
        fallback_desc: Description of the fallback method
    """
    warning_key = f"{strategy_name}:{kernel_name}"
    if warning_key in _issued_warnings:
        return
    
    _issued_warnings.add(warning_key)
    
    if _STRICT_MODE:
        raise RuntimeError(
            f"[{strategy_name}] Optimized kernel '{kernel_name}' is unavailable, "
            f"but strict mode is enabled. Either install the required dependency "
            f"(e.g., vLLM with CUDA ops) or disable strict mode by setting "
            f"DIFFULEX_QUANT_STRICT=0."
        )
    
    warnings.warn(
        f"[{strategy_name}] Optimized kernel '{kernel_name}' is unavailable. "
        f"Falling back to {fallback_desc}. "
        f"Performance will be significantly degraded. "
        f"Install vLLM with CUDA support for optimal performance. "
        f"Set DIFFULEX_QUANT_STRICT=1 to raise error instead of fallback.",
        RuntimeWarning,
        stacklevel=3
    )


def get_kernel_status() -> dict:
    """Get status of all required kernels for quantization."""
    kernels = {
        "gptq_gemm": "GPTQ quantization",
        "gptq_marlin_gemm": "GPTQ Marlin quantization", 
        "awq_gemm": "AWQ quantization",
        "awq_marlin_gemm": "AWQ Marlin quantization",
        "cutlass_scaled_mm": "INT8 W8A8 Cutlass quantization",
        "cutlass_w4a8_mm": "W4A8 Cutlass quantization (Hopper+)",
        "cutlass_encode_and_reorder_int4b": "CUTLASS int4 encoding",
        "cutlass_pack_scale_fp8": "CUTLASS FP8 scale packing",
        "allspark_w8a16_gemm": "INT8 W8A16 AllSpark quantization (Ampere+)",
        "allspark_repack_weight": "AllSpark weight repacking",
        "scaled_fp8_quant": "FP8 quantization",
    }
    
    return {
        name: {
            "available": check_vllm_op_available(name),
            "description": desc,
        }
        for name, desc in kernels.items()
    }


def print_kernel_status():
    """Print kernel availability status."""
    status = get_kernel_status()
    print("=" * 60)
    print("Quantization Kernel Availability")
    print("=" * 60)
    for name, info in status.items():
        status_str = "✓ Available" if info["available"] else "✗ Not Available"
        print(f"  {name:25s} {status_str:15s} ({info['description']})")
    print("=" * 60)
