"""
Forward Plans for Quantized Linear Layers - vLLM-aligned minimal overhead implementation.

CUDA-graph friendly quantized linear dispatch with plan caching.
Each plan binds tensors at init time to eliminate Python overhead during forward.
"""

from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ForwardPlanSig:
    """
    Validation signature for forward plan caching.
    
    Used to validate if cached plan is still valid for current inputs.
    """
    device_type: str
    device_index: int
    x_dtype: torch.dtype
    x_shape: tuple
    has_bias: bool
    mode: str  # "bf16", "quant", "offline"
    strategy_name: str


class ForwardPlanBase:
    """Base class for forward execution plans."""
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the plan."""
        raise NotImplementedError
    
    def get_signature(self) -> ForwardPlanSig:
        """Get the signature for cache validation."""
        raise NotImplementedError


class BF16Plan(ForwardPlanBase):
    """Standard BF16 linear forward plan."""
    
    def __init__(self, signature: ForwardPlanSig,
                 weight: torch.Tensor, bias: Optional[torch.Tensor]):
        self.signature = signature
        self._weight = weight
        self._bias = bias
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Execute BF16 linear forward."""
        return F.linear(x, self._weight, self._bias)
    
    def get_signature(self) -> ForwardPlanSig:
        return self.signature


class QuantizedLinearPlan(ForwardPlanBase):
    """
    Unified quantized linear plan for INT8/FP8 W8A8/W8A16.
    
    Supports:
    - INT8 W8A8: int8 weight + int8 activation
    - INT8 W8A16: int8 weight + bf16 activation  
    - FP8 W8A8: fp8 weight + fp8 activation
    - FP8 W8A16: fp8 weight + bf16 activation
    """
    
    def __init__(self, 
                 signature: ForwardPlanSig,
                 strategy: Any, 
                 quant_kind: str,
                 qweight: torch.Tensor, 
                 scales: torch.Tensor,
                 out_features: int, 
                 bias: Optional[torch.Tensor],
                 weight_format: str,  # "int8" or "fp8_e4m3" or "fp8_e5m2"
                 act_format: str):    # "int8", "fp8_e4m3", "fp8_e5m2", or "bf16"
        self.signature = signature
        self._strategy = strategy
        self._quant_kind = (quant_kind or "other").strip().lower() or "other"
        self._qweight = qweight
        self._scales = scales
        self._out_features = int(out_features)
        self._bias = bias
        self._weight_format = weight_format
        self._act_format = act_format
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Execute quantized linear forward with minimal Python overhead."""
        return self._strategy.linear_forward(
            x, self._qweight, self._bias,
            quant_kind=self._quant_kind,
            quant_scales=self._scales,
            out_features=self._out_features,
        )
    
    def get_signature(self) -> ForwardPlanSig:
        return self.signature


# Backward compatibility aliases
QuantInt8W8A8Plan = QuantizedLinearPlan
QuantInt8W8A16Plan = QuantizedLinearPlan
QuantFP8W8A8Plan = QuantizedLinearPlan
QuantFP8W8A16Plan = QuantizedLinearPlan


class OfflineGPTQPlan(ForwardPlanBase):
    """GPTQ offline quantized plan - binds all tensors at init."""
    
    def __init__(self, signature: ForwardPlanSig,
                 strategy: Any, quant_kind: str,
                 qweight: torch.Tensor, qzeros: torch.Tensor,
                 scales: torch.Tensor, g_idx: Optional[torch.Tensor],
                 bits: int, is_shuffled: bool,
                 out_features: int, in_features: int, group_size: int,
                 bias: Optional[torch.Tensor]):
        self.signature = signature
        self._strategy = strategy
        self._quant_kind = (quant_kind or "other").strip().lower() or "other"
        self._qweight = qweight
        self._qzeros = qzeros
        self._scales = scales
        self._g_idx = g_idx
        self._bits = bits
        self._is_shuffled = is_shuffled
        self._out_features = out_features
        self._in_features = in_features
        self._group_size = group_size
        self._bias = bias
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Execute GPTQ forward with bound tensors."""
        return self._strategy.linear_forward(
            x, None, self._bias,
            quant_kind=self._quant_kind,
            gptq_qweight=self._qweight,
            gptq_qzeros=self._qzeros,
            gptq_scales=self._scales,
            gptq_g_idx=self._g_idx,
            weight_bits=self._bits,
            use_v2_format=False,
            out_features=self._out_features,
            in_features=self._in_features,
            group_size=self._group_size,
        )
    
    def get_signature(self) -> ForwardPlanSig:
        return self.signature


class OfflineAWQPlan(ForwardPlanBase):
    """AWQ offline quantized plan - binds all tensors at init."""
    
    def __init__(self, signature: ForwardPlanSig,
                 strategy: Any, quant_kind: str,
                 qweight: torch.Tensor, qzeros: torch.Tensor,
                 scales: torch.Tensor, bits: int, pack_factor: int,
                 out_features: int, in_features: int, group_size: int,
                 bias: Optional[torch.Tensor]):
        self.signature = signature
        self._strategy = strategy
        self._quant_kind = (quant_kind or "other").strip().lower() or "other"
        self._qweight = qweight
        self._qzeros = qzeros
        self._scales = scales
        self._bits = bits
        self._pack_factor = pack_factor
        self._out_features = out_features
        self._in_features = in_features
        self._group_size = group_size
        self._bias = bias
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Execute AWQ forward with bound tensors."""
        return self._strategy.linear_forward(
            x, None, self._bias,
            quant_kind=self._quant_kind,
            awq_qweight=self._qweight,
            awq_qzeros=self._qzeros,
            awq_scales=self._scales,
            pack_factor=self._pack_factor,
            out_features=self._out_features,
            in_features=self._in_features,
            group_size=self._group_size,
        )
    
    def get_signature(self) -> ForwardPlanSig:
        return self.signature


class DirectGPTQGemmPlan(ForwardPlanBase):
    """Direct GPTQ GEMM via vLLM ops (bypasses Python strategy glue)."""
    
    def __init__(self, signature: ForwardPlanSig,
                 qweight: torch.Tensor, qzeros: torch.Tensor,
                 scales: torch.Tensor, g_idx: torch.Tensor,
                 bits: int, is_shuffled: bool,
                 out_features: int, bias: Optional[torch.Tensor],
                 gemm_op: Callable):
        self.signature = signature
        self._qweight = qweight
        self._qzeros = qzeros
        self._scales = scales
        self._g_idx = g_idx
        self._bits = bits
        self._is_shuffled = is_shuffled
        self._out_features = out_features
        self._bias = bias
        self._gemm_op = gemm_op
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Execute direct GPTQ GEMM."""
        x_2d = x.reshape(-1, x.shape[-1])
        
        output = self._gemm_op(
            x_2d,
            self._qweight,
            self._qzeros,
            self._scales,
            self._g_idx if self._g_idx is not None else torch.empty(0, dtype=torch.int32, device=x.device),
            self._is_shuffled,
            self._bits
        )
        
        output = output.reshape(x.shape[:-1] + (self._out_features,))
        if self._bias is not None:
            output = output + self._bias
        
        return output
    
    def get_signature(self) -> ForwardPlanSig:
        return self.signature


class DirectAWQGemmPlan(ForwardPlanBase):
    """Direct AWQ GEMM via vLLM ops (bypasses Python strategy glue)."""
    
    def __init__(self, signature: ForwardPlanSig,
                 qweight: torch.Tensor, qzeros: torch.Tensor,
                 scales: torch.Tensor, bits: int,
                 out_features: int, bias: Optional[torch.Tensor],
                 gemm_op: Callable):
        self.signature = signature
        self._qweight = qweight
        self._qzeros = qzeros
        self._scales = scales
        self._bits = bits
        self._out_features = out_features
        self._bias = bias
        self._gemm_op = gemm_op
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Execute direct AWQ GEMM."""
        x_2d = x.reshape(-1, x.shape[-1])
        
        output = self._gemm_op(
            x_2d,
            self._qweight,
            self._scales,
            self._qzeros,
            self._bits
        )
        
        output = output.reshape(x.shape[:-1] + (self._out_features,))
        if self._bias is not None:
            output = output + self._bias
        
        return output
    
    def get_signature(self) -> ForwardPlanSig:
        return self.signature


class DirectMarlinGemmPlan(ForwardPlanBase):
    """Direct Marlin GEMM via vLLM ops (bypasses Python strategy glue)."""
    
    def __init__(self, signature: ForwardPlanSig,
                 marlin_qweight: torch.Tensor, marlin_scales: torch.Tensor,
                 marlin_workspace: torch.Tensor, num_bits: int, is_k_full: bool,
                 out_features: int, bias: Optional[torch.Tensor],
                 gemm_op: Callable):
        self.signature = signature
        self._marlin_qweight = marlin_qweight
        self._marlin_scales = marlin_scales
        self._marlin_workspace = marlin_workspace
        self._num_bits = num_bits
        self._is_k_full = is_k_full
        self._out_features = out_features
        self._bias = bias
        self._gemm_op = gemm_op
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Execute direct Marlin GEMM."""
        x_2d = x.reshape(-1, x.shape[-1])
        
        output = self._gemm_op(
            x_2d,
            self._marlin_qweight,
            self._marlin_scales,
            self._marlin_workspace,
            self._num_bits,
            self._is_k_full
        )
        
        output = output.reshape(x.shape[:-1] + (self._marlin_scales.shape[1],))
        if self._bias is not None:
            output = output + self._bias
        
        return output
    
    def get_signature(self) -> ForwardPlanSig:
        return self.signature


class OfflineGPTQMarlinPlan(ForwardPlanBase):
    """GPTQ + Marlin repacked format plan."""
    
    def __init__(self, signature: ForwardPlanSig,
                 strategy: Any, quant_kind: str,
                 marlin_qweight: torch.Tensor, marlin_scales: torch.Tensor,
                 marlin_workspace: torch.Tensor, num_bits: int, is_k_full: bool,
                 in_features: int, out_features: int, group_size: int,
                 bias: Optional[torch.Tensor]):
        self.signature = signature
        self._strategy = strategy
        self._quant_kind = (quant_kind or "other").strip().lower() or "other"
        self._marlin_qweight = marlin_qweight
        self._marlin_scales = marlin_scales
        self._marlin_workspace = marlin_workspace
        self._num_bits = num_bits
        self._is_k_full = is_k_full
        self._in_features = in_features
        self._out_features = out_features
        self._group_size = group_size
        self._bias = bias
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Execute GPTQ Marlin forward with bound tensors."""
        return self._strategy.linear_forward(
            x, None, self._bias,
            quant_kind=self._quant_kind,
            qweight=self._marlin_qweight,
            scales=self._marlin_scales,
            workspace=self._marlin_workspace,
            in_features=self._in_features,
            out_features=self._out_features,
            group_size=self._group_size,
            weight_bits=self._num_bits,
        )
    
    def get_signature(self) -> ForwardPlanSig:
        return self.signature


class OfflineAWQMarlinPlan(ForwardPlanBase):
    """AWQ + Marlin repacked format plan."""
    
    def __init__(self, signature: ForwardPlanSig,
                 strategy: Any, quant_kind: str,
                 marlin_qweight: torch.Tensor, marlin_scales: torch.Tensor,
                 marlin_workspace: torch.Tensor, num_bits: int, is_k_full: bool,
                 in_features: int, out_features: int, group_size: int,
                 bias: Optional[torch.Tensor]):
        self.signature = signature
        self._strategy = strategy
        self._quant_kind = (quant_kind or "other").strip().lower() or "other"
        self._marlin_qweight = marlin_qweight
        self._marlin_scales = marlin_scales
        self._marlin_workspace = marlin_workspace
        self._num_bits = num_bits
        self._is_k_full = is_k_full
        self._in_features = in_features
        self._out_features = out_features
        self._group_size = group_size
        self._bias = bias
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Execute AWQ Marlin forward with bound tensors."""
        return self._strategy.linear_forward(
            x, None, self._bias,
            quant_kind=self._quant_kind,
            qweight=self._marlin_qweight,
            scales=self._marlin_scales,
            workspace=self._marlin_workspace,
            in_features=self._in_features,
            out_features=self._out_features,
            group_size=self._group_size,
            weight_bits=self._num_bits,
        )
    
    def get_signature(self) -> ForwardPlanSig:
        return self.signature
