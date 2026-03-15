"""
Quantization Extension Bootstrap

Main entry point for enabling quantization support.
Must be called BEFORE importing diffulex.

Usage:
    from diffulex.extensions import quantization
    quantization.enable()
    
    # Now import and use diffulex
    from diffulex import Config, LLMEngine
"""

import sys
from typing import Optional, Dict, Any

import torch

# Global state
_is_enabled = False
_quant_config: Optional[Dict[str, Any]] = None


def enable(config: Optional[Dict[str, Any]] = None, 
           kv_cache_dtype: str = "bf16",
           weight_quant_method: str = "bf16",
           linear_attn_dtype: Optional[str] = None,
           linear_mlp_dtype: Optional[str] = None,
           group_size: int = 128,
           desc_act: bool = False):
    """
    Enable quantization extension.
    
    This must be called BEFORE importing diffulex.
    
    Args:
        config: Full quantization configuration dict (optional)
        kv_cache_dtype: KV cache dtype ("bf16", "fp8_e4m3", "fp8_e5m2")
        weight_quant_method: Weight quantization method ("bf16", "fp8_w8a8", "gptq_w4a16", etc.)
        linear_attn_dtype: Attention layer weight dtype override
        linear_mlp_dtype: MLP layer weight dtype override
        group_size: Quantization group size for GPTQ/AWQ
        desc_act: GPTQ desc_act flag
    
    Returns:
        True if successfully enabled
    """
    global _is_enabled, _quant_config
    
    if _is_enabled:
        return True
    
    # Build config
    if config is None:
        config = {
            'kv_cache': {'dtype': kv_cache_dtype},
            'weights': {
                'method': weight_quant_method,
                'group_size': group_size,
                'desc_act': desc_act,
                'linear_attn_dtype': linear_attn_dtype,
                'linear_mlp_dtype': linear_mlp_dtype,
            },
            'activations': {}
        }
    
    _quant_config = config
    
    # Setup import hooks first (before any diffulex imports)
    _setup_import_hooks()
    
    # Import and initialize all components
    try:
        # 0. Import and patch loader first (before auto_model imports it)
        # This ensures load_model is patched before auto_model binds to it
        try:
            import diffulex.utils.loader
            from .loader_patch import patch_loader
            patch_loader()
        except Exception:
            pass
        
        # 1. Import strategies to register them
        from . import strategies  # noqa: F401
        
        # 2. Create strategies from config
        from .config import QuantizationConfig
        from .registry import QuantizationStrategyFactory
        from .context import get_context, set_linear_strategy, set_kv_cache_strategy
        
        from .config import KVCacheQuantConfig, WeightQuantConfig, ActivationQuantConfig
        
        # Build config from provided dict or from individual parameters
        if config is None or config == {}:
            config = {
                'kv_cache': {'dtype': kv_cache_dtype},
                'weights': {
                    'method': weight_quant_method,
                    'group_size': group_size,
                    'desc_act': desc_act,
                    'linear_attn_dtype': linear_attn_dtype,
                    'linear_mlp_dtype': linear_mlp_dtype,
                },
                'activations': {}
            }
        
        kv_cache_config = config.get('kv_cache', {'dtype': kv_cache_dtype})
        weights_config = config.get('weights', {})
        activations_config = config.get('activations', {})
        
        # Ensure method is set
        if 'method' not in weights_config:
            weights_config['method'] = weight_quant_method
        
        quant_config_obj = QuantizationConfig(
            kv_cache=KVCacheQuantConfig(**kv_cache_config),
            weights=WeightQuantConfig(**weights_config),
            activations=ActivationQuantConfig(**activations_config)
        )
        
        # 3. Create and register strategies in context
        strategies_dict = QuantizationStrategyFactory.create_from_config(quant_config_obj)
        
        ctx = get_context()
        for key, strategy in strategies_dict.items():
            ctx.set_strategy(key, strategy)
        
        # 4. Patch linear layers (must happen before diffulex import)
        from .layer_patch import patch_linear_layers
        patch_linear_layers()
        
        # 5. Setup import hooks for post-import patching
        _setup_import_hooks()
        
        # 6. Explicitly import and patch Attention class for FP8 kernel
        # Attention is lazily imported via __getattr__, so we need to trigger the import
        try:
            from diffulex.attention import Attention
            from .kv_cache_patch import patch_attention_class
            patch_attention_class()
        except Exception:
            pass
        
        # 7. Patch already-imported model runners (e.g., d2f imported before hook was set)
        #    Also patch ModelRunnerBase class directly to ensure allocate_kv_cache uses quantization
        try:
            from .kv_cache_patch import patch_model_runner, patch_allocate_kv_cache_method
            import sys
            
            # Explicitly import and patch ModelRunnerBase class
            # This must be done before any runner instances are created
            try:
                from diffulex.engine.model_runner import ModelRunnerBase
                patch_allocate_kv_cache_method(ModelRunnerBase)
                print(f"[Quantization] Patched ModelRunnerBase.allocate_kv_cache")
            except ImportError as e:
                print(f"[Quantization] Warning: Could not patch ModelRunnerBase: {e}")
            
            # Also patch instance __init__ for any runtime setup
            for mod_name in list(sys.modules.keys()):
                if 'model_runner' in mod_name:
                    mod = sys.modules[mod_name]
                    for attr_name in dir(mod):
                        attr = getattr(mod, attr_name)
                        if isinstance(attr, type) and 'runner' in attr_name.lower():
                            # Patch the class
                            original_init = attr.__init__
                            def make_patched_init(orig_init):
                                def patched_init(self, *args, **kwargs):
                                    orig_init(self, *args, **kwargs)
                                    patch_model_runner(self)
                                return patched_init
                            attr.__init__ = make_patched_init(original_init)
        except Exception:
            pass
        
        _is_enabled = True
        
        # Eager patch loader if it was already imported before hook was set
        try:
            import sys
            if 'diffulex.utils.loader' in sys.modules:
                from .loader_patch import patch_loader
                patch_loader()
        except Exception:
            pass
        
        return True
        
    except Exception as e:
        print(f"Warning: Failed to enable quantization: {e}")
        return False


def disable():
    """
    Disable quantization extension.
    
    Restores original classes and clears quantization state.
    """
    global _is_enabled, _quant_config
    
    if not _is_enabled:
        return
    
    try:
        # Unpatch layers
        from .layer_patch import unpatch_linear_layers
        unpatch_linear_layers()
        
        # Clear context
        from .context import QuantizationContext
        QuantizationContext.clear_current()
        
    except Exception:
        pass
    
    _is_enabled = False
    _quant_config = None


def is_enabled() -> bool:
    """Check if quantization extension is enabled."""
    return _is_enabled


def get_config() -> Optional[Dict[str, Any]]:
    """Get current quantization configuration."""
    return _quant_config


def _setup_import_hooks():
    """
    Setup import hooks to patch diffulex modules after they are imported.
    
    This allows patching modules that haven't been loaded yet.
    """
    import builtins
    
    # Store original import
    original_import = builtins.__import__
    
    def import_hook(name, *args, **kwargs):
        module = original_import(name, *args, **kwargs)
        
        # Patch after diffulex modules are imported
        if name.startswith('diffulex') and _is_enabled:
            _post_import_patch(name, module)
        
        return module
    
    # Only hook if not already hooked
    if not hasattr(builtins, '_quant_import_hooked'):
        builtins.__import__ = import_hook
        builtins._quant_import_hooked = True


def _post_import_patch(module_name: str, module):
    """
    Patch modules after they are imported.
    
    This handles patching of modules that are imported after enable() is called.
    """
    # Patch model runner classes
    if 'model_runner' in module_name or module_name.endswith('.d2f'):
        try:
            from .kv_cache_patch import patch_model_runner
            
            # Patch all runner classes in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and 'runner' in attr_name.lower():
                    original_init = attr.__init__
                    
                    def make_patched_init(orig_init):
                        def patched_init(self, *args, **kwargs):
                            orig_init(self, *args, **kwargs)
                            patch_model_runner(self)
                        return patched_init
                    
                    attr.__init__ = make_patched_init(original_init)
        except Exception:
            pass
    
    # Patch KV cache manager
    if 'kv_cache' in module_name:
        try:
            from .kv_cache_patch import patch_kv_cache_manager
            
            if hasattr(module, 'KVCacheManager'):
                original_init = module.KVCacheManager.__init__
                
                def patched_init(self, *args, **kwargs):
                    original_init(self, *args, **kwargs)
                    patch_kv_cache_manager(self)
                
                module.KVCacheManager.__init__ = patched_init
        except Exception:
            pass
    
    # Patch Attention class for FP8 custom kernel
    print(f"[_post_import_patch] Checking module: {module_name}, 'attn_impl' in name: {'attn_impl' in module_name}")
    if 'attn_impl' in module_name or module_name == 'diffulex.attention.attn_impl':
        print(f"[_post_import_patch] Patching Attention for {module_name}")
        try:
            from .kv_cache_patch import patch_attention_class
            patch_attention_class()
            print(f"[_post_import_patch] Patching Attention succeeded for {module_name}")
        except Exception as e:
            print(f"[_post_import_patch] Patching Attention failed: {e}")
    
    # Patch loader when diffulex.utils is imported (loader is a submodule)
    if module_name == 'diffulex.utils':
        try:
            # Check if loader submodule exists
            if hasattr(module, 'loader'):
                from .loader_patch import patch_loader
                patch_loader()
        except Exception:
            pass
    
    # Patch model to quantize weights after loading
    # Only patch when the actual engine module is imported (not during recursion)
    if module_name == 'diffulex.diffulex' or module_name == 'diffulex.engine.tp_worker':
        try:
            _patch_model_for_weight_quantization(module)
        except Exception:
            pass


# Convenience function for configuring quantization from CLI args
def configure_from_args(args) -> Dict[str, Any]:
    """
    Build quantization config from command line arguments.
    
    Args:
        args: Namespace or dict with quantization args
        
    Returns:
        Quantization configuration dict
    """
    if hasattr(args, '__dict__'):
        args = vars(args)
    
    config = {
        'kv_cache': {
            'dtype': args.get('kv_cache_dtype', 'bf16'),
        },
        'weights': {
            'method': args.get('weight_quant_method', 'bf16'),
            'group_size': args.get('quant_group_size', 128),
            'desc_act': args.get('quant_desc_act', False),
        },
        'activations': {}
    }
    
    # Add per-layer dtype overrides if present
    if 'linear_attn_weight_dtype' in args:
        config['weights']['linear_attn_dtype'] = args['linear_attn_weight_dtype']
    if 'linear_mlp_weight_dtype' in args:
        config['weights']['linear_mlp_dtype'] = args['linear_mlp_weight_dtype']
    
    return config


# Auto-enable function for integration with diffulex CLI
def auto_enable_from_config(config):
    """
    Automatically enable quantization from a loaded config.
    
    This is called internally by diffulex when it detects quantization settings.
    
    Args:
        config: Diffulex Config object with quantization attributes
    """
    global _is_enabled, _quant_config
    
    if _is_enabled:
        return True
    
    # Check if config has quantization settings
    has_quant = (
        hasattr(config, 'kv_cache_dtype') and config.kv_cache_dtype != 'bf16'
    ) or (
        hasattr(config, 'weight_quant_method') and config.weight_quant_method != 'bf16'
    ) or (
        hasattr(config, 'quantization_config') and config.quantization_config
    )
    
    if not has_quant:
        return False
    
    # Build config dict from config object
    quant_config = {
        'kv_cache': {
            'dtype': getattr(config, 'kv_cache_dtype', 'bf16'),
        },
        'weights': {
            'method': getattr(config, 'weight_quant_method', 'bf16'),
            'group_size': getattr(config, 'quant_group_size', 128),
            'desc_act': getattr(config, 'quant_desc_act', False),
            'linear_attn_dtype': getattr(config, 'linear_attn_weight_dtype', None),
            'linear_mlp_dtype': getattr(config, 'linear_mlp_weight_dtype', None),
        },
        'activations': {
            'linear_attn_dtype': getattr(config, 'linear_attn_act_dtype', None),
            'linear_mlp_dtype': getattr(config, 'linear_mlp_act_dtype', None),
        }
    }
    
    # Handle nested quantization_config
    nested_config = getattr(config, 'quantization_config', None)
    if nested_config is not None:
        if isinstance(nested_config, dict):
            quant_config.update(nested_config)
        else:
            # It's a QuantizationConfig object
            quant_config['kv_cache'] = {'dtype': nested_config.kv_cache.dtype}
            quant_config['weights'] = {
                'method': nested_config.weights.method,
                'group_size': nested_config.weights.group_size,
                'desc_act': nested_config.weights.desc_act,
            }
    
    return enable(config=quant_config)


def _patch_model_for_weight_quantization(module):
    """
    Patch model initialization to quantize weights after loading.
    
    This ensures online quantization (INT8/FP8) is applied to weights
    immediately after model creation, not during each forward pass.
    """
    from .context import get_linear_strategy
    from .layer_mixin import LinearQuantizationMixin
    
    # Find the Diffulex class
    DiffulexClass = None
    for attr_name in ['DiffulexTPWorker', 'Diffulex', 'DiffulexDPWorker']:
        if hasattr(module, attr_name):
            DiffulexClass = getattr(module, attr_name)
            break
    
    if DiffulexClass is None:
        return
    
    original_init = DiffulexClass.__init__
    
    def patched_init(self, *args, **kwargs):
        # Call original init
        original_init(self, *args, **kwargs)
        
        # After initialization, quantize weights if needed
        _quantize_model_weights(self)
    
    DiffulexClass.__init__ = patched_init


def _quantize_model_weights(model_wrapper):
    """
    Verify offline quantized weights are properly loaded.
    
    Raises error if user specified GPTQ/AWQ but weights are not loaded.
    """
    from .layer_mixin import LinearQuantizationMixin
    from .context import get_linear_strategy
    
    if getattr(model_wrapper, '_weights_quantized', False):
        return
    model_wrapper._weights_quantized = True
    
    # Get current quantization config
    global _quant_config
    if _quant_config is None:
        return
    
    weight_method = _quant_config.get('weights', {}).get('method', 'bf16')
    
    # Skip if not offline quantization
    if weight_method not in ['gptq_w4a16', 'gptq_w8a16', 'awq_w4a16', 'gptq_marlin_w4a16', 'awq_marlin_w4a16']:
        return
    
    # Get model
    model_runner = getattr(model_wrapper, 'model_runner', None)
    if model_runner is None:
        return
    model = getattr(model_runner, 'model', None)
    if model is None:
        return
    
    # Check offline quantized layers
    offline_count = 0
    total_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LinearQuantizationMixin):
            total_count += 1
            if module.has_offline_quantized_weight():
                offline_count += 1
    
    if offline_count == 0 and total_count > 0:
        raise RuntimeError(
            f"Quantization mismatch: weight_quant_method='{weight_method}' specified, "
            f"but no offline quantized weights found in model. "
            f"Please ensure you're loading a {weight_method.upper()} quantized model, "
            f"or set weight_quant_method='bf16' for non-quantized models."
        )
    
    if offline_count > 0:
        print(f"[Quantization] {offline_count}/{total_count} layers using {weight_method}")
