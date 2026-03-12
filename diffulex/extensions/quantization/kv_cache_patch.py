"""
KV Cache Quantization Extension

Extends KV Cache management with quantization support.
Uses dynamic attribute injection to avoid modifying original code.
"""

import torch
from typing import Optional, Tuple, Any, Dict

from .context import get_kv_cache_strategy

# Import custom FP8 Triton kernel
try:
    from .kernels.triton_kernels import fp8_kv_attention_forward
    _HAS_FP8_TRITON_KERNEL = True
except ImportError:
    _HAS_FP8_TRITON_KERNEL = False


def patch_kv_cache_manager(cache_manager):
    """
    Patch a KV cache manager with quantization support.
    
    Args:
        cache_manager: KVCacheManager instance to patch
    """
    if hasattr(cache_manager, '_quantization_patched'):
        return
    
    # Store original methods
    cache_manager._original_allocate = cache_manager.allocate
    cache_manager._original_get_kv = cache_manager.get_kv if hasattr(cache_manager, 'get_kv') else None
    cache_manager._original_set_kv = cache_manager.set_kv if hasattr(cache_manager, 'set_kv') else None
    
    # Add quantization attributes
    cache_manager._quantization_patched = True
    cache_manager._kv_cache_strategy = None
    cache_manager._kv_cache_scales_k = None
    cache_manager._kv_cache_scales_v = None
    cache_manager._kv_cache_dtype = "bf16"
    
    # Replace methods
    cache_manager.allocate = _wrap_allocate(cache_manager)
    if cache_manager._original_get_kv:
        cache_manager.get_kv = _wrap_get_kv(cache_manager)
    if cache_manager._original_set_kv:
        cache_manager.set_kv = _wrap_set_kv(cache_manager)


def _wrap_allocate(cache_manager):
    """Wrap allocate method to support quantization."""
    original = cache_manager._original_allocate
    
    def allocate(*args, **kwargs):
        # Call original allocation
        result = original(*args, **kwargs)
        
        # Initialize quantization if needed
        _init_kv_cache_quantization(cache_manager)
        
        return result
    
    return allocate


def _wrap_get_kv(cache_manager):
    """Wrap get_kv method to support dequantization."""
    original = cache_manager._original_get_kv
    
    def get_kv(slot_mapping, *args, **kwargs):
        # Get raw KV
        k, v = original(slot_mapping, *args, **kwargs)
        
        # Dequantize if needed
        if cache_manager._kv_cache_strategy is not None:
            k_scale = _get_scale_for_slot(cache_manager, 'k', slot_mapping)
            v_scale = _get_scale_for_slot(cache_manager, 'v', slot_mapping)
            
            k, v = cache_manager._kv_cache_strategy.dequantize_kv_for_compute(
                k, v, k_scale, v_scale
            )
        
        return k, v
    
    return get_kv


def _wrap_set_kv(cache_manager):
    """Wrap set_kv method to support quantization."""
    original = cache_manager._original_set_kv
    
    def set_kv(slot_mapping, key, value, *args, **kwargs):
        # Quantize if needed
        if cache_manager._kv_cache_strategy is not None:
            # Update scales
            k_scale, v_scale = _update_scales(cache_manager, key, value)
            
            # Quantize
            key, value = cache_manager._kv_cache_strategy.quantize_kv_for_store(
                key, value, k_scale, v_scale
            )
        
        # Store
        return original(slot_mapping, key, value, *args, **kwargs)
    
    return set_kv


def _init_kv_cache_quantization(cache_manager):
    """Initialize KV cache quantization based on config."""
    # Get strategy from context
    strategy = get_kv_cache_strategy()
    
    if strategy is None:
        return
    
    cache_manager._kv_cache_strategy = strategy
    cache_manager._kv_cache_dtype = getattr(strategy, 'name', 'bf16')
    
    # Initialize scale tensors if needed
    if strategy.requires_kv_cache_scales:
        # Scale shape depends on strategy (per-token, per-head, etc.)
        num_layers = getattr(cache_manager, 'num_layers', 1)
        num_heads = getattr(cache_manager, 'num_heads', 1)
        head_size = getattr(cache_manager, 'head_size', 128)
        max_num_blocks = getattr(cache_manager, 'max_num_blocks', 1)
        
        # Simple per-token scales
        cache_manager._kv_cache_scales_k = torch.zeros(
            num_layers, max_num_blocks, num_heads, head_size,
            dtype=torch.float32,
            device=cache_manager.device if hasattr(cache_manager, 'device') else 'cuda'
        )
        cache_manager._kv_cache_scales_v = torch.zeros(
            num_layers, max_num_blocks, num_heads, head_size,
            dtype=torch.float32,
            device=cache_manager.device if hasattr(cache_manager, 'device') else 'cuda'
        )


def _get_scale_for_slot(cache_manager, kv_type: str, slot_mapping):
    """Get scale for a specific slot."""
    if kv_type == 'k':
        scales = cache_manager._kv_cache_scales_k
    else:
        scales = cache_manager._kv_cache_scales_v
    
    if scales is None:
        return None
    
    # slot_mapping determines which scales to use
    return scales[slot_mapping]


def _update_scales(cache_manager, k: torch.Tensor, v: torch.Tensor):
    """Update KV cache scales based on new values."""
    strategy = cache_manager._kv_cache_strategy
    
    if strategy is None or not strategy.requires_kv_cache_scales:
        return None, None
    
    # Get current scales
    k_scale = cache_manager._kv_cache_scales_k
    v_scale = cache_manager._kv_cache_scales_v
    
    # Update scales
    new_k_scale, new_v_scale = strategy.update_scales(k, v, k_scale, v_scale)
    
    # Store updated scales
    cache_manager._kv_cache_scales_k = new_k_scale
    cache_manager._kv_cache_scales_v = new_v_scale
    
    return new_k_scale, new_v_scale


def use_fp8_triton_kernel() -> bool:
    """Check if FP8 Triton kernel is available."""
    return _HAS_FP8_TRITON_KERNEL


def run_fp8_kv_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    page_tables: torch.Tensor,
    context_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    softmax_scale: float,
    is_e4m3: bool = True,
) -> Optional[torch.Tensor]:
    """
    Run FP8 KV attention using custom Triton kernel.
    
    Returns None if kernel is not available.
    """
    if not _HAS_FP8_TRITON_KERNEL:
        return None
    
    try:
        return fp8_kv_attention_forward(
            q, k_cache, v_cache, k_scale, v_scale,
            page_tables, context_lens, cu_seqlens_q,
            softmax_scale, is_e4m3
        )
    except Exception:
        return None


# Model Runner patching
def patch_model_runner(model_runner):
    """
    Patch ModelRunner with KV cache quantization support.
    
    This patches the model_runner's KV cache allocation and access methods.
    """
    if hasattr(model_runner, '_kv_quant_patched'):
        return
    
    model_runner._kv_quant_patched = True
    
    # Store original allocate_kv_cache
    if hasattr(model_runner, 'allocate_kv_cache'):
        original_allocate = model_runner.allocate_kv_cache
        
        def allocate_kv_cache_with_quant(*args, **kwargs):
            # Call original
            result = original_allocate(*args, **kwargs)
            
            # Initialize quantization
            _init_runner_kv_quantization(model_runner)
            
            return result
        
        model_runner.allocate_kv_cache = allocate_kv_cache_with_quant
    
    # Store original get_kv_cache
    if hasattr(model_runner, 'get_kv_cache'):
        original_get = model_runner.get_kv_cache
        
        def get_kv_cache_with_dequant(*args, **kwargs):
            # Get raw KV
            result = original_get(*args, **kwargs)
            
            # Dequantize if needed
            if result is not None and model_runner._kv_cache_strategy is not None:
                k, v = result
                k, v = model_runner._kv_cache_strategy.dequantize_kv_for_compute(k, v)
                result = (k, v)
            
            return result
        
        model_runner.get_kv_cache = get_kv_cache_with_dequant


def _init_runner_kv_quantization(model_runner):
    """Initialize KV quantization for model runner."""
    # Get strategy from context
    strategy = get_kv_cache_strategy()
    
    if strategy is None:
        return
    
    model_runner._kv_cache_strategy = strategy
    model_runner.kv_cache_dtype = getattr(strategy, 'name', 'bf16')
    
    # Initialize scales in runner
    config = getattr(model_runner, 'config', None)
    if config is None:
        return
    
    if strategy.requires_kv_cache_scales:
        # Get dimensions from config
        num_layers = getattr(config, 'num_hidden_layers', 1)
        num_heads = getattr(config, 'num_key_value_heads', 
                           getattr(config, 'num_attention_heads', 1))
        head_size = getattr(config, 'hidden_size', 4096) // getattr(config, 'num_attention_heads', 32)
        max_num_seqs = getattr(config, 'max_num_seqs', 1)
        max_seq_len = getattr(config, 'max_seq_len', 2048)
        
        # Allocate scale tensors
        device = getattr(model_runner, 'device', 'cuda')
        model_runner.kv_cache_scales_k = torch.zeros(
            num_layers, max_num_seqs, max_seq_len, num_heads,
            dtype=torch.float32, device=device
        )
        model_runner.kv_cache_scales_v = torch.zeros(
            num_layers, max_num_seqs, max_seq_len, num_heads,
            dtype=torch.float32, device=device
        )
