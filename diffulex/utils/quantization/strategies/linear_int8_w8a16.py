"""
W8A16 Linear quantization strategy (int8 weight + bf16 activation).

This path is now implemented by reusing Diffulex's marlin(AllSpark)-style W8A16
strategy, which matches vLLM's effective fast path and avoids TileLang.
"""

from __future__ import annotations

from diffulex.utils.quantization.registry import register_linear_strategy
from diffulex.utils.quantization.strategy import LinearQuantizationStrategy

from .linear_marlin_int8_w8a16 import LinearMarlinInt8W8A16Strategy


class LinearInt8W8A16Strategy(LinearMarlinInt8W8A16Strategy):
    """
    Compatibility alias for the historical Diffulex strategy name.

    This keeps the registry and `strategies.__init__` imports stable while
    reusing the vLLM-aligned marlin(AllSpark) W8A16 implementation.
    """


@register_linear_strategy(weight_dtype="int8", act_dtype="bf16")
def _build_linear_int8_w8a16() -> LinearQuantizationStrategy:
    # Alias to marlin(AllSpark) W8A16 implementation.
    return LinearInt8W8A16Strategy()

