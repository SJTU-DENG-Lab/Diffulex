"""
KV Cache Quantization Extension

Extends KV Cache management with quantization support.
Uses dynamic attribute injection to avoid modifying original code.
"""

import torch
from typing import Optional, Tuple, Any, Dict
import logging

from .context import get_kv_cache_strategy

logger = logging.getLogger(__name__)

# Import custom FP8 Triton kernel (new unified version with Stage 1 + Stage 2)
try:
    from .kernels.triton_kernels import chunked_prefill_attn_unified_fp8
    # Enable Triton kernel for all CUDA devices (including RTX 4090 sm_89)
    # NOTE: Kernel uses FP32 intermediate to avoid cvt.bf16.f16 (sm_90+ requirement)
    import torch
    if torch.cuda.is_available():
        _HAS_FP8_TRITON_KERNEL = True
        print(f"[Quantization] FP8 unified Triton kernel enabled for device capability {torch.cuda.get_device_capability()}")
    else:
        _HAS_FP8_TRITON_KERNEL = False
except ImportError as e:
    print(f"[Quantization] FP8 unified Triton kernel not available: {e}")
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


# Model Runner patching
def patch_allocate_kv_cache_method(model_runner_class):
    """
    Patch ModelRunnerBase.allocate_kv_cache class method to support quantization.
    
    This must be called before any model runner instance is created.
    """
    if hasattr(model_runner_class, '_allocate_kv_cache_patched'):
        return
    model_runner_class._allocate_kv_cache_patched = True
    
    # Store original allocate_kv_cache
    original_allocate = model_runner_class.allocate_kv_cache
    
    def allocate_kv_cache_with_quant(self):
        """Allocate KV cache with quantization support."""
        # Get quantization strategy before allocation
        strategy = get_kv_cache_strategy()
        
        if strategy is not None:
            # Store strategy for later use
            self._kv_cache_strategy = strategy
            self.kv_cache_dtype = getattr(strategy, 'name', 'bf16')
            
            # Determine storage dtype from strategy
            storage_dtype_info = strategy.get_storage_dtype(self)
            if isinstance(storage_dtype_info, tuple):
                storage_dtype = storage_dtype_info[0]
            else:
                storage_dtype = storage_dtype_info
            
            # Temporarily override dtype for allocation
            original_dtype = getattr(self, 'default_dtype', torch.bfloat16)
            self._original_dtype_for_kv = original_dtype
            self.default_dtype = storage_dtype if storage_dtype != torch.float8_e4m3fn else torch.bfloat16
            
            # Store FP8 info if needed
            if storage_dtype == torch.float8_e4m3fn:
                self._kv_cache_storage_dtype = torch.float8_e4m3fn
                self._kv_cache_compute_dtype = torch.bfloat16
        
        # Call original allocation
        result = original_allocate(self)
        
        # Restore dtype and convert allocated cache if needed
        if hasattr(self, '_original_dtype_for_kv'):
            self.default_dtype = self._original_dtype_for_kv
            delattr(self, '_original_dtype_for_kv')
        
        # If FP8 was requested, convert the allocated cache
        if hasattr(self, '_kv_cache_storage_dtype') and self._kv_cache_storage_dtype == torch.float8_e4m3fn:
            # Convert allocated caches to FP8
            if hasattr(self, 'kv_cache') and self.kv_cache is not None:
                self.kv_cache = self.kv_cache.to(torch.float8_e4m3fn)
                
                # Update attention module references for unified layout
                # Find attention modules and re-assign their cache views
                attn_modules = [m for m in self.model.modules() if hasattr(m, 'k_cache') and hasattr(m, 'v_cache')]
                for layer_id, m in enumerate(attn_modules):
                    m.k_cache = self.kv_cache[0, layer_id]
                    m.v_cache = self.kv_cache[1, layer_id]
            
            if hasattr(self, 'k_cache') and self.k_cache is not None:
                self.k_cache = self.k_cache.to(torch.float8_e4m3fn)
                self.v_cache = self.v_cache.to(torch.float8_e4m3fn)
            
            logger.info(f"KV cache allocated with dtype: {torch.float8_e4m3fn}")
        
        return result
    
    model_runner_class.allocate_kv_cache = allocate_kv_cache_with_quant


def patch_model_runner(model_runner):
    """
    Patch ModelRunner with KV cache quantization support.
    
    This patches the model_runner's KV cache allocation and access methods.
    """
    if hasattr(model_runner, '_kv_quant_patched'):
        return
    
    model_runner._kv_quant_patched = True
    
    # Store original allocate_kv_cache (instance level)
    if hasattr(model_runner, 'allocate_kv_cache') and not hasattr(model_runner.__class__, '_allocate_kv_cache_patched'):
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
                
                # For FP8 with custom kernel, skip dequantization - kernel handles it
                strategy = model_runner._kv_cache_strategy
                if hasattr(strategy, 'name') and 'fp8' in strategy.name.lower():
                    has_kernel = getattr(strategy, 'has_triton_kernel', lambda: False)()
                    if has_kernel:
                        # Skip dequantization - kernel will handle it
                        return result
                
                # Dequantize for non-FP8 or when kernel not available
                k, v = strategy.dequantize_kv_for_compute(k, v)
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


# Attention class patching
def patch_attention_class():
    """
    Patch Attention class to use custom FP8 Triton kernel when available.
    This is called during extension initialization.
    """
    import warnings
    
    try:
        from diffulex.attention.attn_impl import Attention
        from diffulex_kernel import chunked_prefill_attn_unified
    except ImportError:
        return
    
    # Store original forward
    if hasattr(Attention, '_original_forward'):
        return  # Already patched
    
    Attention._original_forward = Attention.forward
    
    def forward_with_fp8_kernel(self, q, k, v, mask=None):
        """Forward that uses custom FP8 unified kernel when available."""
        from einops import rearrange
        from diffulex.attention import fetch_attn_metadata
        from diffulex_kernel import (
            store_kv_cache_distinct_layout,
            store_kv_cache_unified_layout,
            chunked_prefill_attn_unified,
        )
        from .context import get_kv_cache_strategy
        
        # Reshape
        q = rearrange(q, "s (nh hd) -> s nh hd", nh=self.num_heads, hd=self.head_dim)
        k = rearrange(k, "s (nkvh hd) -> s nkvh hd", nkvh=self.num_kv_heads, hd=self.head_dim)
        v = rearrange(v, "s (nkvh hd) -> s nkvh hd", nkvh=self.num_kv_heads, hd=self.head_dim)
        
        attn_metadata = fetch_attn_metadata()
        k_cache, v_cache = self.k_cache, self.v_cache
        is_unified_layout = attn_metadata.kv_cache_layout == "unified"
        
        # Store KV cache
        if k_cache.numel() and v_cache.numel():
            if attn_metadata.need_kv_cache_store:
                store_kv_cache = store_kv_cache_unified_layout if is_unified_layout else store_kv_cache_distinct_layout
                store_kv_cache(k, v, k_cache, v_cache, attn_metadata.slot_mapping, attn_metadata)
        
        # Try to use custom FP8 Triton kernel through strategy layer
        strategy = get_kv_cache_strategy()
        if strategy is not None and k_cache.dtype == torch.float8_e4m3fn and _HAS_FP8_TRITON_KERNEL:
            try:
                if hasattr(strategy, 'has_triton_kernel') and strategy.has_triton_kernel():
                    # Get scales from strategy (per-tensor running max)
                    # For now use default scales (1.0) as scale management needs integration
                    num_reqs = len(attn_metadata.context_lens)
                    k_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)
                    v_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)
                    
                    # Call unified FP8 kernel through strategy
                    # This handles both Stage 1 (cached FP8 KV) and Stage 2 (new BF16 KV)
                    o = strategy.triton_attention(
                        q=q,
                        k=k,  # New K (BF16) for Stage 2
                        v=v,  # New V (BF16) for Stage 2
                        k_cache=k_cache,  # Cached K (FP8) for Stage 1
                        v_cache=v_cache,  # Cached V (FP8) for Stage 1
                        attn_metadata=attn_metadata,
                        k_scale=k_scale,
                        v_scale=v_scale,
                    )
                    if o is not None:
                        return rearrange(o, "s nh hd -> s (nh hd)").contiguous()
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"FP8 unified kernel failed: {e}")
                pass  # Fallback to standard kernel
        
        # Standard kernel with on-the-fly dequantization
        if k_cache.dtype == torch.float8_e4m3fn:
            k_cache_bf16 = k_cache.to(torch.bfloat16)
            v_cache_bf16 = v_cache.to(torch.bfloat16)
        else:
            k_cache_bf16, v_cache_bf16 = k_cache, v_cache
            
        o = chunked_prefill_attn_unified(q, k, v, k_cache_bf16, v_cache_bf16, attn_metadata)
        return rearrange(o, "s nh hd -> s (nh hd)").contiguous()
    
    Attention.forward = forward_with_fp8_kernel
