"""
Explicit No-Quantization Strategy

Explicit no-op strategy that explicitly does no quantization.
Similar to BF16 but explicitly named for clarity.
"""

from .linear_bf16 import BF16LinearStrategy
from ..registry import register_linear_strategy


@register_linear_strategy("none", "bf16")
@register_linear_strategy("none", "none")
class NoQuantizationStrategy(BF16LinearStrategy):
    """Explicit no quantization strategy."""
    
    @property
    def name(self) -> str:
        return "no_quantization"
