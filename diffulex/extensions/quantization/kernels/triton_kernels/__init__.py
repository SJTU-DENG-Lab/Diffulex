"""
Custom Triton Kernels

Pure Triton implementations for operations not covered by vLLM kernels.
"""

try:
    from .fp8_kv_attention import (
        Fp8KVAttentionKernel,
        fp8_kv_attention_forward,
    )
    _HAS_FP8_KERNEL = True
except ImportError:
    _HAS_FP8_KERNEL = False

__all__ = [
    "Fp8KVAttentionKernel",
    "fp8_kv_attention_forward",
    "_HAS_FP8_KERNEL",
]
