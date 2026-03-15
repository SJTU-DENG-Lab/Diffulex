"""
Quantization Configuration

Configuration dataclasses for quantization settings.
All fields are frozen/immutable for safety.
"""

from dataclasses import dataclass, field, fields
from typing import Optional, Dict, Any, Tuple


@dataclass(frozen=True)
class KVCacheQuantConfig:
    """KV Cache quantization configuration."""
    dtype: str = "bf16"  # "bf16", "fp8_e4m3", "fp8_e5m2"
    
    def __post_init__(self):
        valid_dtypes = ["bf16", "fp8_e4m3", "fp8_e5m2", "fp8"]
        if self.dtype not in valid_dtypes:
            raise ValueError(f"Invalid kv_cache dtype: {self.dtype}, valid: {valid_dtypes}")


@dataclass(frozen=True)
class WeightQuantConfig:
    """Weight quantization configuration."""
    method: str = "bf16"  # "bf16", "fp8_w8a8", "fp8_w8a16", "int8_w8a8", "int8_w8a16"
                          # "gptq_w4a16", "awq_w4a16", "gptq_marlin_w4a16", "awq_marlin_w4a16"
    group_size: int = 128
    desc_act: bool = False  # GPTQ: whether to use activation-dependent grouping
    
    # Per-layer-type dtype overrides
    linear_attn_dtype: Optional[str] = None
    linear_mlp_dtype: Optional[str] = None
    linear_other_dtype: Optional[str] = None
    
    def __post_init__(self):
        valid_methods = [
            "bf16", "none",
            "fp8_w8a8", "fp8_w8a16",
            "int8_w8a8", "int8_w8a16",
            "gptq_w2a16", "gptq_w3a16", "gptq_w4a16", "gptq_w8a16",
            "awq_w4a16",
            "gptq_marlin_w4a16", "gptq_marlin_w8a16",
            "awq_marlin_w4a16",
            "cutlass_w4a8",
        ]
        if self.method not in valid_methods:
            raise ValueError(f"Invalid weight quant method: {self.method}")


@dataclass(frozen=True)
class ActivationQuantConfig:
    """Activation quantization configuration."""
    # Per-layer-type dtype overrides
    linear_attn_dtype: Optional[str] = None
    linear_mlp_dtype: Optional[str] = None
    linear_other_dtype: Optional[str] = None


@dataclass(frozen=True)
class QuantizationConfig:
    """
    Top-level quantization configuration container.
    
    This is the main configuration object passed to the engine.
    """
    kv_cache: KVCacheQuantConfig = field(default_factory=KVCacheQuantConfig)
    weights: WeightQuantConfig = field(default_factory=WeightQuantConfig)
    activations: ActivationQuantConfig = field(default_factory=ActivationQuantConfig)
    
    @classmethod
    def from_diffulex_config(cls, config) -> "QuantizationConfig":
        """
        Extract quantization config from a Diffulex Config object.
        
        Looks for quantization-related attributes dynamically added to config.
        """
        # Extract kv_cache config
        kv_dtype = getattr(config, 'kv_cache_dtype', 'bf16')
        kv_config = KVCacheQuantConfig(dtype=kv_dtype)
        
        # Extract weight config
        weight_method = getattr(config, 'weight_quant_method', 'bf16')
        weight_config = WeightQuantConfig(
            method=weight_method,
            group_size=getattr(config, 'quant_group_size', 128),
            desc_act=getattr(config, 'quant_desc_act', False),
            linear_attn_dtype=getattr(config, 'linear_attn_weight_dtype', None),
            linear_mlp_dtype=getattr(config, 'linear_mlp_weight_dtype', None),
            linear_other_dtype=getattr(config, 'linear_other_weight_dtype', None),
        )
        
        # Extract activation config
        act_config = ActivationQuantConfig(
            linear_attn_dtype=getattr(config, 'linear_attn_act_dtype', None),
            linear_mlp_dtype=getattr(config, 'linear_mlp_act_dtype', None),
            linear_other_dtype=getattr(config, 'linear_other_act_dtype', None),
        )
        
        return cls(
            kv_cache=kv_config,
            weights=weight_config,
            activations=act_config,
        )
    
    def get_linear_dtype(self, kind: str) -> Tuple[str, str]:
        """
        Get weight and activation dtype for a linear layer kind.
        
        Args:
            kind: "attn", "mlp", or "other"
            
        Returns:
            Tuple of (weight_dtype, act_dtype)
        """
        # Get weight dtype
        weight_attr = f"linear_{kind}_dtype"
        weight_dtype = getattr(self.weights, weight_attr, None) or self.weights.method
        
        # Get activation dtype
        act_attr = f"linear_{kind}_dtype"
        act_dtype = getattr(self.activations, act_attr, None)
        
        # Default activation dtype based on weight dtype
        if act_dtype is None:
            if weight_dtype in ["fp8_w8a8", "cutlass_w4a8"]:
                # FP8 W8A8 or W4A8: activation is fp8
                act_dtype = "fp8_e4m3"
            elif weight_dtype == "int8_w8a8":
                # INT8 W8A8: activation is int8
                act_dtype = "int8"
            elif weight_dtype in ["fp8_w8a16", "int8_w8a16"]:
                # W8A16: activation is bf16
                act_dtype = "bf16"
            else:
                act_dtype = "bf16"
        
        # Normalize dtype names
        weight_dtype = self._normalize_dtype(weight_dtype)
        act_dtype = self._normalize_dtype(act_dtype)
        
        return weight_dtype, act_dtype
    
    def _normalize_dtype(self, dtype: str) -> str:
        """Normalize dtype names."""
        # Handle compound formats like "int8_w8a8", "fp8_w8a16", etc.
        if dtype in ["int8_w8a8", "int8_w8a16"]:
            return "int8"
        if dtype in ["fp8_w8a8", "fp8_w8a16"]:
            return "fp8_e4m3"
        if dtype == "cutlass_w4a8":
            return "int4"
        
        dtype_map = {
            "fp8": "fp8_e4m3",
            "int8": "int8",
            "int4": "int4",
            "bf16": "bf16",
            "fp16": "fp16",
            "fp32": "fp32",
        }
        return dtype_map.get(dtype, dtype)
    
    def is_quantization_enabled(self) -> bool:
        """Check if any quantization is enabled."""
        return (
            self.kv_cache.dtype != "bf16" or
            self.weights.method not in ["bf16", "none"]
        )
