"""
Layer Monkey Patch

Zero-intrusive layer replacement using Monkey Patching.
Replaces original Linear classes with quantization-aware versions.
"""

import torch
import torch.nn as nn
from typing import Optional, Type

from .layer_mixin import LinearQuantizationMixin


# Store original classes
_original_classes: dict = {}


def create_quantized_linear_class(base_class: Type[nn.Module]) -> Type[nn.Module]:
    """
    Create a quantized version of a linear class.
    
    Args:
        base_class: Original linear class (e.g., ColumnParallelLinear)
        
    Returns:
        New class that inherits from both base_class and LinearQuantizationMixin
    """
    
    class QuantizedLinear(LinearQuantizationMixin, base_class):
        """
        Quantized version of linear layer.
        
        Inherits from both the original class and LinearQuantizationMixin.
        
        Features:
        - Load-time quantization: quantizes weights immediately after loading
        - Memory efficient: deletes original BF16 weights after quantization
        """
        
        def __init__(self, *args, quant_kind: str = "other", **kwargs):
            # Initialize base class
            super().__init__(*args, **kwargs)
            
            # Initialize quantization
            self.init_quantization(quant_kind)
            
            # Override weight_loader to enable load-time quantization
            self._setup_quantized_weight_loader()
        
        def _setup_quantized_weight_loader(self):
            """
            Setup weight loader that quantizes weights immediately after loading.
            
            This overrides the original weight_loader to implement:
            1. Load weight from checkpoint (via original weight_loader)
            2. Quantize weight to INT8/FP8
            3. Delete original BF16 weight to save memory
            4. Store quantized weight
            """
            # Store reference to original weight_loader
            if hasattr(self, 'weight') and self.weight is not None:
                original_loader = getattr(self.weight, 'weight_loader', None)
                if original_loader is not None:
                    # Bind our quantized loader
                    self.weight.weight_loader = self._quantized_weight_loader
                    self._original_weight_loader = original_loader
        
        def _quantized_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, *args, **kwargs):
            """
            Weight loader that quantizes immediately after loading.
            
            Flow:
            1. Call original weight_loader to load weight into param
            2. Get quantization strategy
            3. Quantize weight
            4. Store quantized weight
            5. Delete original weight to free memory
            """
            import torch
            from .context import get_linear_strategy
            
            # Step 1: Call original weight loader
            if hasattr(self, '_original_weight_loader') and self._original_weight_loader is not None:
                # Handle different signature for different linear types
                try:
                    if args or kwargs:
                        self._original_weight_loader(param, loaded_weight, *args, **kwargs)
                    else:
                        self._original_weight_loader(param, loaded_weight)
                except Exception:
                    # Fallback: direct copy
                    param.data.copy_(loaded_weight)
            else:
                # No original loader, direct copy
                param.data.copy_(loaded_weight)
            
            # Step 2: Get quantization strategy
            strategy = get_linear_strategy(self.quant_kind)
            if strategy is None:
                return  # No quantization enabled
            
            # Step 3: Check if we should quantize this layer
            # Skip if already quantized or if it's not a weight we want to quantize
            if self.has_quantized_weight() or self.has_offline_quantized_weight():
                return
            
            # Step 4: Quantize weight immediately after loading
            try:
                # Get the loaded weight data
                weight = param.data
                
                # Only quantize if it's BF16/FP16/FP32 (not already quantized)
                if weight.dtype not in [torch.bfloat16, torch.float16, torch.float32]:
                    return
                
                # Quantize weight
                q_weight, w_meta = strategy.quantize_weight_for_kernel(weight)
                w_scale = w_meta.get("scale")
                w_zero = w_meta.get("zero_point")
                
                # Step 5: Store quantized weight
                self.set_quantized_weight(q_weight, w_scale, w_zero)
                
                # Step 6: Delete original weight to save memory
                # Replace param.data with empty tensor to free memory
                # The actual data is now stored in quant_weight_int8 buffer
                param.data = torch.empty(0, dtype=weight.dtype, device=weight.device)
                
                # Remove from parameters (convert to buffer or just delete)
                if 'weight' in self._parameters:
                    del self._parameters['weight']
                
                # Mark as quantized
                self._weight_is_quantized_py = True
                
            except Exception as e:
                # Quantization failed, keep original weight
                # This ensures model can still work even if quantization fails
                pass
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward with quantization support."""
            # Get bias if available
            bias = getattr(self, 'bias', None)
            
            # Use unified forward dispatcher
            return self._forward_base(x, bias)
    
    # Set class name
    QuantizedLinear.__name__ = f"Quant{base_class.__name__}"
    QuantizedLinear.__qualname__ = f"Quant{base_class.__name__}"
    
    return QuantizedLinear


def patch_linear_layers():
    """
    Monkey patch linear layer classes with quantization support.
    
    This function replaces the original Linear classes in diffulex.layer.linear
    with quantized versions that support all quantization formats.
    """
    try:
        import diffulex.layer.linear as linear_module
    except ImportError:
        # diffulex not loaded yet, will patch when available
        return False
    
    global _original_classes
    
    # Classes to patch
    classes_to_patch = [
        'ReplicatedLinear',
        'ColumnParallelLinear', 
        'RowParallelLinear',
        'MergedColumnParallelLinear',
        'QKVParallelLinear',
    ]
    
    for class_name in classes_to_patch:
        if hasattr(linear_module, class_name):
            base_class = getattr(linear_module, class_name)
            
            # Skip if already patched
            if class_name in _original_classes:
                continue
            
            # Store original
            _original_classes[class_name] = base_class
            
            # Create quantized version
            quant_class = create_quantized_linear_class(base_class)
            
            # Replace in module
            setattr(linear_module, class_name, quant_class)
    
    return True


def unpatch_linear_layers():
    """
    Restore original linear layer classes.
    
    Reverses the monkey patching done by patch_linear_layers().
    """
    try:
        import diffulex.layer.linear as linear_module
    except ImportError:
        return False
    
    global _original_classes
    
    for class_name, original_class in _original_classes.items():
        setattr(linear_module, class_name, original_class)
    
    _original_classes.clear()
    return True


def is_patched() -> bool:
    """Check if layers are currently patched."""
    return len(_original_classes) > 0


def get_original_class(class_name: str) -> Optional[Type[nn.Module]]:
    """Get original (unpatched) class by name."""
    return _original_classes.get(class_name)


# Helper functions for creating layer instances with specific quantization kinds
def create_quantized_layer(base_class_name: str, *args, quant_kind: str = "other", **kwargs):
    """
    Create a quantized layer instance.
    
    Args:
        base_class_name: Name of base class (e.g., "ColumnParallelLinear")
        quant_kind: Layer kind ("attn", "mlp", "other")
        *args, **kwargs: Arguments for base class constructor
    """
    try:
        import diffulex.layer.linear as linear_module
    except ImportError:
        raise RuntimeError("diffulex not loaded")
    
    base_class = getattr(linear_module, base_class_name, None)
    if base_class is None:
        raise ValueError(f"Unknown linear class: {base_class_name}")
    
    # Create with quant_kind
    instance = base_class(*args, **kwargs)
    
    # Set quant kind if mixin is applied
    if hasattr(instance, 'init_quantization'):
        instance.init_quantization(quant_kind)
    
    return instance


# Dynamic layer replacement for TP (Tensor Parallel) scenarios
class DynamicQuantizedLinear:
    """
    Dynamic layer that chooses implementation based on context.
    
    This allows runtime selection between quantized and non-quantized paths.
    """
    
    def __new__(cls, base_class: Type[nn.Module], *args, 
                quant_kind: str = "other", enable_quant: bool = True, **kwargs):
        """
        Create appropriate layer instance.
        
        Args:
            base_class: Base linear class
            quant_kind: Layer kind for strategy selection
            enable_quant: Whether to enable quantization for this layer
            *args, **kwargs: Constructor arguments
        """
        # Get context to check if quantization is enabled
        if enable_quant:
            from .context import get_context
            ctx = get_context()
            
            # Check if we have a strategy for this kind
            if ctx.has_strategy(f"linear_{quant_kind}"):
                # Create quantized instance
                instance = base_class(*args, **kwargs)
                
                # Apply mixin dynamically if not already patched
                if not hasattr(instance, '_forward_base'):
                    LinearQuantizationMixin.init_quantization(instance, quant_kind)
                    # Bind methods
                    instance._forward_base = LinearQuantizationMixin._forward_base.__get__(instance, type(instance))
                    instance.has_quantized_weight = LinearQuantizationMixin.has_quantized_weight.__get__(instance, type(instance))
                    instance.has_offline_quantized_weight = LinearQuantizationMixin.has_offline_quantized_weight.__get__(instance, type(instance))
                    instance.set_quantized_weight = LinearQuantizationMixin.set_quantized_weight.__get__(instance, type(instance))
                    instance.set_offline_quantized_weight = LinearQuantizationMixin.set_offline_quantized_weight.__get__(instance, type(instance))
                    instance.enable_forward_plan = LinearQuantizationMixin.enable_forward_plan.__get__(instance, type(instance))
                
                return instance
        
        # Return standard instance
        return base_class(*args, **kwargs)
