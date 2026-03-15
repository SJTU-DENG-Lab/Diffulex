"""
Quantization Context

Thread-local context for managing quantization strategies.
Provides activation quantization caching for step-local reuse.
"""

from threading import local
from typing import Dict, Optional, Tuple, Any
import torch


class QuantizationContext:
    """
    Thread-local quantization context.
    
    Manages quantization strategies and activation caching.
    Each thread has its own isolated context.
    """
    
    _thread_local = local()
    
    def __init__(self):
        """Initialize context for current thread."""
        self._strategies: Dict[str, Any] = {}
        self._act_quant_cache: Dict[tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
    
    @classmethod
    def get_current(cls) -> "QuantizationContext":
        """Get or create the current thread's context."""
        if not hasattr(cls._thread_local, 'context'):
            cls._thread_local.context = cls()
        return cls._thread_local.context
    
    @classmethod
    def clear_current(cls):
        """Clear the current thread's context."""
        if hasattr(cls._thread_local, 'context'):
            cls._thread_local.context._strategies.clear()
            cls._thread_local.context._act_quant_cache.clear()
            delattr(cls._thread_local, 'context')
    
    # Strategy management
    def set_strategy(self, key: str, strategy):
        """Register a strategy by key."""
        self._strategies[key] = strategy
    
    def get_strategy(self, key: str):
        """Get a strategy by key."""
        return self._strategies.get(key)
    
    def has_strategy(self, key: str) -> bool:
        """Check if a strategy is registered."""
        return key in self._strategies
    
    # Linear strategy helpers
    def set_linear_strategy(self, kind: str, strategy):
        """
        Register a linear quantization strategy.
        
        Args:
            kind: "attn", "mlp", or "other"
            strategy: LinearQuantizationStrategy instance
        """
        key = f"linear_{(kind or 'other').strip().lower() or 'other'}"
        self.set_strategy(key, strategy)
    
    def get_linear_strategy(self, kind: str):
        """Get linear quantization strategy for a layer kind."""
        key = f"linear_{(kind or 'other').strip().lower() or 'other'}"
        return self.get_strategy(key)
    
    # KV Cache strategy helpers
    def set_kv_cache_strategy(self, strategy):
        """Register KV cache quantization strategy."""
        self.set_strategy("kv_cache", strategy)
    
    def get_kv_cache_strategy(self):
        """Get KV cache quantization strategy."""
        return self.get_strategy("kv_cache")
    
    # Activation quantization cache
    def _act_quant_cache_key(self, x: torch.Tensor) -> tuple:
        """
        Generate a unique cache key for a tensor.
        
        Uses data pointer, shape, stride, dtype, device, and version
        to ensure cache correctness.
        """
        # Handle inference tensors (no version tracking in no_grad mode)
        try:
            version = int(x._version)
        except (RuntimeError, AttributeError):
            version = -1
        
        return (
            int(x.data_ptr()),
            tuple(x.shape),
            tuple(x.stride()),
            str(x.dtype),
            str(x.device),
            version,
        )
    
    def get_cached_act_quant(self, x: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get cached quantized activation.
        
        Returns:
            Tuple of (quantized_tensor, scale) or None if not cached
        """
        key = self._act_quant_cache_key(x)
        return self._act_quant_cache.get(key)
    
    def set_cached_act_quant(self, x: torch.Tensor, q_x: torch.Tensor, scale: torch.Tensor):
        """Cache quantized activation."""
        key = self._act_quant_cache_key(x)
        self._act_quant_cache[key] = (q_x, scale)
    
    def clear_act_quant_cache(self):
        """Clear activation quantization cache."""
        self._act_quant_cache.clear()


# Module-level convenience functions
def get_context() -> QuantizationContext:
    """Get the current thread's quantization context."""
    return QuantizationContext.get_current()


def set_linear_strategy(kind: str, strategy):
    """Set linear quantization strategy for current thread."""
    get_context().set_linear_strategy(kind, strategy)


def get_linear_strategy(kind: str):
    """Get linear quantization strategy for current thread."""
    return get_context().get_linear_strategy(kind)


def set_kv_cache_strategy(strategy):
    """Set KV cache quantization strategy for current thread."""
    get_context().set_kv_cache_strategy(strategy)


def get_kv_cache_strategy():
    """Get KV cache quantization strategy for current thread."""
    return get_context().get_kv_cache_strategy()


def clear_act_quant_cache():
    """Clear activation quantization cache for current thread."""
    get_context().clear_act_quant_cache()


def get_cached_act_quant(x: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Get cached quantized activation for current thread."""
    return get_context().get_cached_act_quant(x)


def set_cached_act_quant(x: torch.Tensor, q_x: torch.Tensor, scale: torch.Tensor):
    """Cache quantized activation for current thread."""
    get_context().set_cached_act_quant(x, q_x, scale)


def step_end_cleanup():
    """
    Call at the end of each generation step.
    
    Clears activation quantization cache to prepare for next step.
    """
    clear_act_quant_cache()
