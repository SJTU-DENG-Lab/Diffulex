"""
Custom Triton Kernels

Pure Triton implementations for operations not covered by vLLM kernels.
"""

# Unified FP8 kernel (Stage 1 + Stage 2)
try:
    from .chunked_prefill_attn_unified_fp8 import (
        chunked_prefill_attn_unified_fp8,
    )
    _HAS_FP8_UNIFIED_KERNEL = True
except ImportError:
    _HAS_FP8_UNIFIED_KERNEL = False

__all__ = [
    "chunked_prefill_attn_unified_fp8",
    "_HAS_FP8_UNIFIED_KERNEL",
]
