"""
Quantization Strategy Base Classes

Abstract base classes for quantization strategies.
All concrete strategies must inherit from these base classes.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import torch


class QuantizationStrategy(ABC):
    """Base class for all quantization strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this strategy."""
        pass
    
    @abstractmethod
    def get_storage_dtype(self, device: torch.device) -> Tuple[torch.dtype, int]:
        """
        Get storage dtype and element size in bytes.
        
        Returns:
            Tuple of (dtype, element_size_in_bytes)
        """
        pass
    
    @abstractmethod
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a tensor.
        
        Returns:
            Tuple of (quantized_tensor, scale)
        """
        pass
    
    @abstractmethod
    def dequantize(self, q_x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize a tensor."""
        pass
    
    def get_scale_shape(self, x_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Get the shape of the scale tensor for a given input shape.
        
        Default is per-tensor scaling (scalar scale).
        Override for per-channel or per-group scaling.
        """
        return ()


class KVCacheQuantizationStrategy(QuantizationStrategy):
    """Base class for KV cache quantization strategies."""
    
    @property
    @abstractmethod
    def requires_kv_cache_scales(self) -> bool:
        """Whether this strategy requires explicit scale tensors."""
        pass
    
    @abstractmethod
    def compute_scales(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute quantization scales for K and V tensors.
        
        Returns:
            Tuple of (k_scale, v_scale)
        """
        pass
    
    @abstractmethod
    def update_scales(self, k: torch.Tensor, v: torch.Tensor, 
                      k_scale: torch.Tensor, v_scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update scales based on new K/V values (for running max, etc.).
        
        Returns:
            Updated (k_scale, v_scale)
        """
        pass
    
    @abstractmethod
    def init_scales(self, batch_size: int, num_heads: int, 
                    device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize scale tensors.
        
        Returns:
            Initialized (k_scale, v_scale) tensors
        """
        pass
    
    def maybe_set_attn_metadata_scales(self, attn_metadata, k_scale: torch.Tensor, v_scale: torch.Tensor):
        """
        Set scales in attention metadata if supported.
        
        This is called before attention computation to pass scale information.
        """
        pass
    
    @abstractmethod
    def quantize_kv_for_store(self, k: torch.Tensor, v: torch.Tensor,
                              k_scale: Optional[torch.Tensor] = None,
                              v_scale: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize K and V tensors for storage in KV cache.
        
        Returns:
            Tuple of (quantized_k, quantized_v)
        """
        pass
    
    @abstractmethod
    def dequantize_kv_for_compute(self, q_k: torch.Tensor, q_v: torch.Tensor,
                                  k_scale: Optional[torch.Tensor] = None,
                                  v_scale: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dequantize K and V tensors for attention computation.
        
        Returns:
            Tuple of (dequantized_k, dequantized_v)
        """
        pass


class LinearQuantizationStrategy(QuantizationStrategy):
    """Base class for linear layer quantization strategies."""
    
    @property
    @abstractmethod
    def linear_weight_format(self) -> str:
        """
        Weight format for linear layers.
        
        Returns one of: "bf16", "fp16", "fp32", "int8", "int4", 
                        "fp8_e4m3", "fp8_e5m2"
        """
        pass
    
    @property
    @abstractmethod
    def linear_act_format(self) -> str:
        """
        Activation format for linear layers.
        
        Returns one of: "bf16", "fp16", "fp32", "int8",
                        "fp8_e4m3", "fp8_e5m2"
        """
        pass
    
    @abstractmethod
    def quantize_weight_for_kernel(self, weight: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """
        Quantize weight tensor for kernel consumption.
        
        Returns:
            Tuple of (quantized_weight, metadata_dict)
        """
        pass
    
    @abstractmethod
    def quantize_act_for_kernel(self, x: torch.Tensor, 
                                cache_key: Optional[str] = None) -> Tuple[torch.Tensor, Any]:
        """
        Quantize activation tensor for kernel consumption.
        
        Args:
            x: Input activation
            cache_key: Optional key for step-local caching
            
        Returns:
            Tuple of (quantized_activation, metadata_dict)
        """
        pass
    
    @abstractmethod
    def linear_forward(self, x: torch.Tensor, weight: torch.Tensor, 
                       bias: Optional[torch.Tensor] = None,
                       *, quant_kind: str = "other", **kwargs) -> torch.Tensor:
        """
        Perform quantized linear forward pass.
        
        Args:
            x: Input tensor
            weight: Weight tensor (may be pre-quantized)
            bias: Optional bias tensor
            quant_kind: Layer kind ("attn", "mlp", "other")
            **kwargs: Additional strategy-specific arguments
            
        Returns:
            Output tensor
        """
        pass


class WeightQuantizationStrategy(QuantizationStrategy):
    """Base class for weight-only quantization strategies."""
    
    @property
    @abstractmethod
    def weight_format(self) -> str:
        """Weight storage format."""
        pass
    
    @abstractmethod
    def prepare_weight_for_linear(self, weight: torch.Tensor, 
                                  bias: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Prepare weight for use in linear layer.
        
        Returns:
            Dict containing prepared weight buffers and metadata
        """
        pass
