"""
Quantization Strategy Registry

Registration and factory pattern for quantization strategies.
"""

from typing import Dict, Callable, Any, Optional, List, Tuple
import functools


# Global registries
_kv_cache_strategies: Dict[str, Callable[..., Any]] = {}
_linear_strategies: Dict[Tuple[str, str], Callable[..., Any]] = {}


def register_kv_cache_strategy(name: str):
    """
    Decorator to register a KV cache quantization strategy.
    
    Usage:
        @register_kv_cache_strategy("fp8_e4m3")
        class FP8E4M3KVCacheStrategy(KVCacheQuantizationStrategy):
            ...
    """
    def decorator(cls_or_fn):
        _kv_cache_strategies[name] = cls_or_fn
        return cls_or_fn
    return decorator


def register_linear_strategy(weight_dtype: str, act_dtype: str = "bf16"):
    """
    Decorator to register a linear quantization strategy.
    
    Args:
        weight_dtype: Weight quantization format ("bf16", "fp8", "int8", etc.)
        act_dtype: Activation quantization format ("bf16", "fp8", "int8", etc.)
        
    Usage:
        @register_linear_strategy("fp8_e4m3", "fp8_e4m3")
        class FP8W8A8LinearStrategy(LinearQuantizationStrategy):
            ...
    """
    def decorator(cls_or_fn):
        key = (weight_dtype, act_dtype)
        _linear_strategies[key] = cls_or_fn
        return cls_or_fn
    return decorator


def create_kv_cache_strategy(name: str, **kwargs) -> Any:
    """
    Factory function to create a KV cache strategy.
    
    Args:
        name: Strategy name (e.g., "bf16", "fp8_e4m3")
        **kwargs: Strategy-specific arguments
        
    Returns:
        Strategy instance
        
    Raises:
        ValueError: If strategy not found
    """
    if name not in _kv_cache_strategies:
        raise ValueError(
            f"Unknown KV cache strategy: {name}. "
            f"Available: {list(_kv_cache_strategies.keys())}"
        )
    
    strategy_cls = _kv_cache_strategies[name]
    return strategy_cls(**kwargs)


def create_linear_strategy(weight_dtype: str, act_dtype: str = "bf16", 
                           **kwargs) -> Any:
    """
    Factory function to create a linear quantization strategy.
    
    Args:
        weight_dtype: Weight quantization format
        act_dtype: Activation quantization format
        **kwargs: Strategy-specific arguments
        
    Returns:
        Strategy instance
        
    Raises:
        ValueError: If strategy not found
    """
    # Normalize dtype names
    weight_dtype = _normalize_dtype(weight_dtype)
    act_dtype = _normalize_dtype(act_dtype)
    
    # Try exact match first
    key = (weight_dtype, act_dtype)
    if key in _linear_strategies:
        strategy_cls = _linear_strategies[key]
        return strategy_cls(**kwargs)
    
    # Try fallback to bf16 activation
    key_fallback = (weight_dtype, "bf16")
    if key_fallback in _linear_strategies:
        strategy_cls = _linear_strategies[key_fallback]
        return strategy_cls(**kwargs)
    
    raise ValueError(
        f"Unknown linear strategy: weight_dtype={weight_dtype}, act_dtype={act_dtype}. "
        f"Available: {list(_linear_strategies.keys())}"
    )


def registered_kv_cache_dtypes() -> List[str]:
    """Get list of registered KV cache strategy names."""
    return list(_kv_cache_strategies.keys())


def registered_linear_strategies() -> List[Tuple[str, str]]:
    """Get list of registered linear strategy keys."""
    return list(_linear_strategies.keys())


def is_kv_cache_strategy_registered(name: str) -> bool:
    """Check if a KV cache strategy is registered."""
    return name in _kv_cache_strategies


def is_linear_strategy_registered(weight_dtype: str, act_dtype: str = "bf16") -> bool:
    """Check if a linear strategy is registered."""
    weight_dtype = _normalize_dtype(weight_dtype)
    act_dtype = _normalize_dtype(act_dtype)
    return (weight_dtype, act_dtype) in _linear_strategies


def _normalize_dtype(dtype: str) -> str:
    """Normalize dtype name for registry lookup."""
    dtype = dtype.lower().strip()
    
    # Handle common aliases
    aliases = {
        "fp8": "fp8_e4m3",
        "int8": "int8",
        "int4": "int4",
        "bf16": "bf16",
        "fp16": "fp16",
        "fp32": "fp32",
        "fp8_e4m3": "fp8_e4m3",
        "fp8_e5m2": "fp8_e5m2",
        "none": "bf16",
    }
    
    return aliases.get(dtype, dtype)


class QuantizationStrategyFactory:
    """
    Factory for creating complete strategy sets from configuration.
    """
    
    @staticmethod
    def create_from_config(config) -> Dict[str, Any]:
        """
        Create all strategies from a QuantizationConfig.
        
        Args:
            config: QuantizationConfig instance
            
        Returns:
            Dict mapping strategy names to instances
        """
        from .config import QuantizationConfig
        
        if isinstance(config, QuantizationConfig):
            quant_config = config
        else:
            quant_config = QuantizationConfig.from_diffulex_config(config)
        
        strategies = {}
        
        # Create KV cache strategy
        kv_dtype = quant_config.kv_cache.dtype
        if is_kv_cache_strategy_registered(kv_dtype):
            strategies['kv_cache'] = create_kv_cache_strategy(kv_dtype)
        else:
            # Fall back to bf16
            strategies['kv_cache'] = create_kv_cache_strategy("bf16")
        
        # Create linear strategies for each kind
        for kind in ["attn", "mlp", "other"]:
            weight_dtype, act_dtype = quant_config.get_linear_dtype(kind)
            
            if is_linear_strategy_registered(weight_dtype, act_dtype):
                strategy = create_linear_strategy(weight_dtype, act_dtype)
                strategies[f'linear_{kind}'] = strategy
            else:
                # Fall back to bf16
                strategies[f'linear_{kind}'] = create_linear_strategy("bf16", "bf16")
        
        return strategies
