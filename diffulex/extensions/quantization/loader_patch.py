"""
Weight Loader Extension

Extends model loading with quantization-aware weight loading.
Handles offline quantized formats (GPTQ, AWQ, Marlin).
"""

import torch
from typing import Dict, Any, Optional, Callable
import re

from .registry import create_linear_strategy, registered_linear_strategies


# Global state to track if already patched
_loader_patched = False
_original_load_checkpoint = None
_original_load_model = None
_patch_lock = False  # Simple lock to prevent concurrent patching

def patch_loader():
    """
    Patch the weight loader to support quantized models.
    
    This intercepts weight loading to:
    1. Detect offline quantized formats (GPTQ, AWQ)
    2. Route weights to appropriate buffers
    3. Prepare weights for quantized linear layers
    """
    global _loader_patched, _original_load_checkpoint, _original_load_model, _patch_lock
    
    # Double-check locking pattern
    if _loader_patched:
        return True
    
    if _patch_lock:
        # Another thread is patching, wait for it to complete
        import time
        while _patch_lock and not _loader_patched:
            time.sleep(0.001)
        return _loader_patched
    
    _patch_lock = True
    
    try:
        if _loader_patched:  # Check again after acquiring lock
            return True
        
        # Use sys.modules to avoid re-triggering import hooks
        import sys
        loader_module = sys.modules.get('diffulex.utils.loader')
        if loader_module is None:
            try:
                import diffulex.utils.loader as loader_module
            except ImportError:
                _patch_lock = False
                return False
        
        # Patch load_checkpoint function
        if hasattr(loader_module, 'load_checkpoint'):
            _original_load_checkpoint = loader_module.load_checkpoint
            
            def quantized_load_checkpoint(checkpoint_path: str, *args, **kwargs):
                """Load checkpoint with GPTQ/AWQ weights support."""
                state_dict = _original_load_checkpoint(checkpoint_path, *args, **kwargs)
                
                # Check for separate GPTQ/AWQ weights file
                # This merges pre-quantized weights without auto-detecting format
                # User must explicitly specify weight_quant_method when loading
                gptq_state_dict = _load_gptq_weights_file(checkpoint_path)
                if gptq_state_dict is not None:
                    # Merge GPTQ weights into state_dict (keys are already converted)
                    state_dict.update(gptq_state_dict)
                
                # Note: No auto-detection per user requirement.
                # User must explicitly set weight_quant_method="gptq_w4a16" etc.
                state_dict['_quantization_config'] = None
                
                return state_dict
            
            loader_module.load_checkpoint = quantized_load_checkpoint
        
        # Patch load_model function
        if hasattr(loader_module, 'load_model'):
            _original_load_model = loader_module.load_model
            
            def quantized_load_model(model: torch.nn.Module, config, *args, **kwargs):
                """Load model with quantization support."""
                import os
                
                # Check if config has quantization settings
                quant_config = _get_quant_config_from_model_config(config)
                
                if quant_config is not None:
                    # Initialize quantization for model
                    _init_model_quantization(model, quant_config)
                
                # Call original load (loads standard weights)
                result = _original_load_model(model, config, *args, **kwargs)
                
                # Load GPTQ/AWQ weights if needed
                if quant_config is not None and 'gptq' in quant_config.get('format', '').lower():
                    model_path = config.model if hasattr(config, 'model') else str(config)
                    gptq_state_dict = _load_gptq_weights_file(model_path)
                    if gptq_state_dict is not None:
                        # Group GPTQ weights by layer
                        layer_weights = {}
                        for key, tensor in gptq_state_dict.items():
                            # Parse key like "model.layers.0.mlp.down_proj.gptq_qweight"
                            parts = key.rsplit('.', 1)
                            if len(parts) == 2:
                                module_name, buffer_name = parts
                                if module_name not in layer_weights:
                                    layer_weights[module_name] = {}
                                layer_weights[module_name][buffer_name] = tensor
                        
                        # Set offline quantized weights for each layer
                        bits = quant_config.get('bits', 4)
                        group_size = quant_config.get('group_size', 128)
                        for module_name, buffers in layer_weights.items():
                            try:
                                module = model.get_submodule(module_name)
                                if hasattr(module, 'set_offline_quantized_weight'):
                                    # Map buffer names from gptq_* to standard names
                                    qweight = buffers.get('gptq_qweight')
                                    qzeros = buffers.get('gptq_qzeros')
                                    scales = buffers.get('gptq_scales')
                                    g_idx = buffers.get('gptq_g_idx')
                                    
                                    if qweight is not None and qzeros is not None and scales is not None:
                                        # Move tensors to model's device before setting
                                        device = next(module.parameters()).device if list(module.parameters()) else 'cpu'
                                        qweight = qweight.to(device)
                                        qzeros = qzeros.to(device)
                                        scales = scales.to(device)
                                        if g_idx is not None and g_idx.numel() > 0:
                                            g_idx = g_idx.to(device)
                                        
                                        module.set_offline_quantized_weight(
                                            qweight=qweight,
                                            qzeros=qzeros,
                                            scales=scales,
                                            g_idx=g_idx if g_idx is not None and g_idx.numel() > 0 else None,
                                            bits=bits,
                                            group_size=group_size,
                                            format_type='gptq'
                                        )
                            except Exception:
                                pass
                
                # Post-process loaded weights (prepare/shuffle if needed)
                if quant_config is not None:
                    _post_process_loaded_weights(model, quant_config)
                
                return result
            
            loader_module.load_model = quantized_load_model
        
        _loader_patched = True
        _patch_lock = False
        return True
    except Exception:
        _patch_lock = False
        return False
    return True


def _detect_quantization_config(state_dict: Dict[str, torch.Tensor]) -> Optional[Dict[str, Any]]:
    """
    Detect quantization format from state dict.
    
    Returns:
        Quantization config dict or None if not quantized
    """
    config = {
        'format': None,
        'bits': 4,
        'group_size': 128,
        'layers': {},
    }
    
    # Check for GPTQ keys
    has_gptq = any('qweight' in k or 'qzeros' in k or 'g_idx' in k for k in state_dict.keys())
    has_awq = any('awq' in k.lower() for k in state_dict.keys())
    
    if has_gptq:
        config['format'] = 'gptq'
        
        # Detect bits from weight shape
        for key, tensor in state_dict.items():
            if 'qweight' in key:
                # Infer bits from packing
                config['bits'] = _infer_bits_from_qweight(tensor)
                break
        
        # Detect group size
        for key, tensor in state_dict.items():
            if 'scales' in key:
                config['group_size'] = _infer_group_size(tensor, state_dict)
                break
        
        # Map layers
        config['layers'] = _map_gptq_layers(state_dict)
        
    elif has_awq:
        config['format'] = 'awq'
        config['bits'] = 4  # AWQ is typically 4-bit
        config['layers'] = _map_awq_layers(state_dict)
    
    else:
        return None
    
    return config


def _load_gptq_weights_file(checkpoint_path: str) -> Optional[Dict[str, torch.Tensor]]:
    """
    Load separate GPTQ/AWQ weights file if exists.
    
    Some models store quantized weights in separate files like:
    - model_quantized_gptq.safetensors
    - model_quantized_awq.safetensors
    
    Returns:
        State dict with quantized weights (keys converted to match layer buffer names),
        or None if not found
    """
    import os
    
    # Get checkpoint directory
    checkpoint_dir = os.path.dirname(checkpoint_path) if os.path.isfile(checkpoint_path) else checkpoint_path
    
    # Possible quantized weights file names
    quant_file_names = [
        'model_quantized_gptq.safetensors',
        'model_quantized_awq.safetensors',
        'model_quantized.safetensors',
        'gptq_model.safetensors',
        'awq_model.safetensors',
    ]
    
    for file_name in quant_file_names:
        quant_path = os.path.join(checkpoint_dir, file_name)
        if os.path.exists(quant_path):
            try:
                # Load quantized weights
                if quant_path.endswith('.safetensors'):
                    from safetensors.torch import load_file
                    state_dict = load_file(quant_path)
                else:
                    state_dict = torch.load(quant_path, map_location='cpu')
                
                # Convert to regular dict if needed
                if not isinstance(state_dict, dict):
                    state_dict = dict(state_dict)
                
                # Convert keys to match layer buffer names
                # e.g., "model.layers.0.self_attn.q_proj.qweight" -> "model.layers.0.self_attn.q_proj.gptq_qweight"
                converted_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key
                    # GPTQ format: convert qweight/qzeros/scales/g_idx to gptq_*
                    if '.qweight' in key and not key.endswith('.gptq_qweight'):
                        new_key = key.replace('.qweight', '.gptq_qweight')
                    elif '.qzeros' in key and not key.endswith('.gptq_qzeros'):
                        new_key = key.replace('.qzeros', '.gptq_qzeros')
                    elif '.g_idx' in key and not key.endswith('.gptq_g_idx'):
                        new_key = key.replace('.g_idx', '.gptq_g_idx')
                    elif key.endswith('.scales') and not key.endswith('.gptq_scales'):
                        # Be careful not to match other scales, only exact .scales suffix
                        parts = key.split('.')
                        if parts[-1] == 'scales':
                            new_key = key[:-7] + '.gptq_scales'  # Replace .scales with .gptq_scales
                    converted_state_dict[new_key] = value
                
                return converted_state_dict
            except Exception:
                # If loading fails, try next file
                continue
    
    return None


def _infer_bits_from_qweight(qweight: torch.Tensor) -> int:
    """Infer quantization bits from qweight tensor."""
    # qweight is int32 packed, each element contains 32/bits weights
    # Common: 4-bit (8 per int32), 8-bit (4 per int32)
    if qweight.dtype != torch.int32:
        return 4  # Default
    
    # Try to detect from model config if available
    return 4  # Most common


def _infer_group_size(scales: torch.Tensor, state_dict: Dict) -> int:
    """Infer group size from scales tensor."""
    # Typical group sizes: 32, 64, 128
    return 128  # Default


def _map_gptq_layers(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, str]]:
    """Map GPTQ layer names to their buffer names."""
    layers = {}
    
    for key in state_dict.keys():
        # Match patterns like "model.layers.0.self_attn.q_proj.qweight"
        match = re.match(r'(.+\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj))\.(.+)', key)
        if match:
            layer_name = match.group(1)
            buffer_name = match.group(2)
            
            if layer_name not in layers:
                layers[layer_name] = {}
            
            layers[layer_name][buffer_name] = key
    
    return layers


def _map_awq_layers(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, str]]:
    """Map AWQ layer names to their buffer names."""
    layers = {}
    
    for key in state_dict.keys():
        match = re.match(r'(.+\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj))\.(.+)', key)
        if match:
            layer_name = match.group(1)
            buffer_name = match.group(2)
            
            if layer_name not in layers:
                layers[layer_name] = {}
            
            layers[layer_name][buffer_name] = key
    
    return layers


def _process_quantized_weights(state_dict: Dict[str, torch.Tensor],
                               quant_config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Process quantized weights for loading.
    
    This prepares weights for the quantized linear layers.
    """
    # Validate all required buffers are present
    for layer_name, buffers in quant_config['layers'].items():
        required = ['qweight', 'qzeros', 'scales']
        for req in required:
            if req not in buffers:
                # Try to find with different naming
                for key in state_dict.keys():
                    if layer_name in key and req in key:
                        buffers[req] = key
                        break
    
    return state_dict


def _get_quant_config_from_model_config(config) -> Optional[Dict[str, Any]]:
    """Extract quantization config from model config or global config."""
    # First check model config
    if config is not None:
        # Check for quantization config attributes
        quant_config = getattr(config, 'quantization_config', None)
        if quant_config is not None:
            return quant_config
        
        # Check for individual attributes
        weight_quant = getattr(config, 'weight_quant_method', None)
        if weight_quant is not None and weight_quant != 'bf16':
            return {
                'format': weight_quant,
                'bits': 4 if '4' in weight_quant else 8,
                'group_size': getattr(config, 'quant_group_size', 128),
            }
    
    # Then check global quantization config (set via quantization.enable())
    try:
        from .bootstrap import get_config
        global_config = get_config()
        if global_config is not None:
            weights_config = global_config.get('weights', {})
            method = weights_config.get('method', 'bf16')
            if method != 'bf16':
                return {
                    'format': method,
                    'bits': 4 if '4' in method else 8,
                    'group_size': weights_config.get('group_size', 128),
                }
    except Exception:
        pass
    
    return None


def _init_model_quantization(model: torch.nn.Module, quant_config: Dict[str, Any]):
    """Initialize quantization for model layers."""
    # Create appropriate strategies based on format
    fmt = quant_config.get('format', 'bf16')
    bits = quant_config.get('bits', 4)
    group_size = quant_config.get('group_size', 128)
    
    if 'gptq_w4a16' in fmt.lower():
        from .strategies.linear_gptq_wxa16 import GPTQW4A16LinearStrategy
        strategy = GPTQW4A16LinearStrategy(group_size=group_size)
    elif 'gptq_w8a16' in fmt.lower():
        from .strategies.linear_gptq_wxa16 import GPTQW8A16LinearStrategy
        strategy = GPTQW8A16LinearStrategy(group_size=group_size)
    elif 'gptq_w2a16' in fmt.lower():
        from .strategies.linear_gptq_wxa16 import GPTQW2A16LinearStrategy
        strategy = GPTQW2A16LinearStrategy(group_size=group_size)
    elif 'gptq_w3a16' in fmt.lower():
        from .strategies.linear_gptq_wxa16 import GPTQW3A16LinearStrategy
        strategy = GPTQW3A16LinearStrategy(group_size=group_size)
    elif 'gptq' in fmt.lower():
        # Default to w4a16 for generic gptq
        from .strategies.linear_gptq_wxa16 import GPTQW4A16LinearStrategy
        strategy = GPTQW4A16LinearStrategy(group_size=group_size)
    elif 'awq' in fmt.lower():
        from .strategies.linear_awq_w4a16 import AWQW4A16LinearStrategy
        strategy = AWQW4A16LinearStrategy(bits=bits, group_size=group_size)
    else:
        return
    
    # Store strategy in context
    from .context import set_linear_strategy
    for kind in ['attn', 'mlp', 'other']:
        set_linear_strategy(kind, strategy)
    
    # Initialize quantization for each linear layer
    # This sets quant_kind so weight loader knows how to handle the weights
    for name, module in model.named_modules():
        # Check if this is a linear layer that supports quantization
        if hasattr(module, 'init_quantization'):
            # Determine quant_kind based on layer name
            quant_kind = 'other'
            if any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                quant_kind = 'attn'
            elif any(x in name for x in ['gate_proj', 'up_proj', 'down_proj']):
                quant_kind = 'mlp'
            
            module.init_quantization(quant_kind)
            module._quant_strategy = strategy


def _post_process_loaded_weights(model: torch.nn.Module, quant_config: Dict[str, Any]):
    """
    Post-process weights after loading.
    
    This prepares offline quantized weights for use.
    """
    fmt = quant_config.get('format', '')
    bits = quant_config.get('bits', 4)
    group_size = quant_config.get('group_size', 128)
    
    # Process GPTQ weights: set offline quantized state from loaded buffers
    if 'gptq' in fmt.lower():
        for name, module in model.named_modules():
            if hasattr(module, 'set_offline_quantized_weight'):
                # Check if GPTQ buffers were loaded
                if hasattr(module, 'gptq_qweight'):
                    g_idx = getattr(module, 'gptq_g_idx', None)
                    module.set_offline_quantized_weight(
                        qweight=module.gptq_qweight,
                        qzeros=module.gptq_qzeros,
                        scales=module.gptq_scales,
                        g_idx=g_idx,
                        bits=bits,
                        group_size=group_size,
                        format_type='gptq'
                    )
    
    # Prepare GPTQ weights (shuffle if needed)
    if 'gptq' in fmt.lower():
        _prepare_gptq_weights(model, quant_config)
    
    # Prepare Marlin format if enabled
    if 'marlin' in fmt.lower() or getattr(model, 'use_marlin', False):
        _prepare_marlin_weights(model, quant_config)


def _prepare_gptq_weights(model: torch.nn.Module, quant_config: Dict[str, Any]):
    """Prepare GPTQ weights (shuffle)."""
    try:
        import vllm._custom_ops as ops
        if not hasattr(ops, 'gptq_shuffle'):
            return
        
        for name, module in model.named_modules():
            # Check if this is a quantized linear layer
            if hasattr(module, '_maybe_prepare_offline_gptq'):
                module._maybe_prepare_offline_gptq()
    except ImportError:
        pass


def _prepare_marlin_weights(model: torch.nn.Module, quant_config: Dict[str, Any]):
    """Prepare Marlin format weights."""
    try:
        import vllm._custom_ops as ops
        if not hasattr(ops, 'gptq_marlin_repack'):
            return
        
        for name, module in model.named_modules():
            if hasattr(module, '_maybe_prepare_marlin'):
                module._maybe_prepare_marlin()
    except ImportError:
        pass


# Utility functions for manual weight loading
def load_quantized_weights(model: torch.nn.Module, 
                          checkpoint_path: str,
                          quant_format: str = "gptq"):
    """
    Manually load quantized weights into model.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to quantized checkpoint
        quant_format: "gptq" or "awq"
    """
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Detect config
    quant_config = _detect_quantization_config(state_dict)
    if quant_config is None:
        quant_config = {'format': quant_format, 'bits': 4, 'group_size': 128}
    
    # Load weights
    _load_quantized_state_dict(model, state_dict, quant_config)


def _load_quantized_state_dict(model: torch.nn.Module,
                               state_dict: Dict[str, torch.Tensor],
                               quant_config: Dict[str, Any]):
    """Load quantized state dict into model."""
    layers = quant_config.get('layers', {})
    
    for layer_name, buffers in layers.items():
        # Find corresponding module
        module = model
        for part in layer_name.split('.'):
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                module = None
                break
        
        if module is None or not hasattr(module, 'set_offline_quantized_weight'):
            continue
        
        # Get buffers
        qweight = state_dict.get(buffers.get('qweight'))
        qzeros = state_dict.get(buffers.get('qzeros'))
        scales = state_dict.get(buffers.get('scales'))
        g_idx = state_dict.get(buffers.get('g_idx')) if 'g_idx' in buffers else None
        
        if qweight is None or qzeros is None or scales is None:
            continue
        
        # Set quantized weight
        module.set_offline_quantized_weight(
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            g_idx=g_idx,
            bits=quant_config.get('bits', 4),
            group_size=quant_config.get('group_size', 128),
            format_type=quant_config.get('format', 'gptq')
        )
