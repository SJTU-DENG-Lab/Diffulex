"""
Linear Quantization Mixin

Core quantization logic for Linear layers.
Provides forward dispatch, weight management, and plan caching.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Any, Dict

from .linear_plan_builder import build_forward_plan
from .context import get_linear_strategy


class LinearQuantizationMixin:
    """
    Mixin for quantized linear layers.
    
    Adds quantization capabilities to any linear layer class.
    """
    
    def init_quantization(self, quant_kind: str = "other"):
        """
        Initialize quantization-related buffers.
        
        Call this in __init__ after weight initialization.
        """
        self.quant_kind = quant_kind
        self._forward_out_features = None
        self._forward_plan_enabled = True
        self._forward_plan = None
        self._quant_strategy = None
        
        # Quantization state (Python-only, no tensor buffers needed)
        self._weight_is_quantized = False
        self._offline_quant_format = None  # None, "gptq", "awq", "gptq_marlin", "awq_marlin"
        self._offline_quant_bits = 0
        self._offline_quant_group_size = 0
        self._gptq_is_shuffled = False
        self._gptq_marlin_is_prepared = False
    
    def enable_forward_plan(self, enabled: bool = True):
        """Enable/disable forward plan caching."""
        self._forward_plan_enabled = enabled
        if not enabled:
            self._forward_plan = None
    
    def build_forward_plan_for_static(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Build forward plan for static shapes (decode step)."""
        if not self._forward_plan_enabled:
            return
        self._forward_plan = build_forward_plan(self, x, bias)
    
    def has_quantized_weight(self) -> bool:
        """Check if layer has online quantized weights (INT8/FP8)."""
        return self._weight_is_quantized
    
    def has_offline_quantized_weight(self) -> bool:
        """Check if layer has offline quantized weights (GPTQ/AWQ)."""
        return self._offline_quant_format is not None
    
    def set_quantized_weight(self, qweight: torch.Tensor, scales: torch.Tensor,
                             zero_points: Optional[torch.Tensor] = None):
        """
        Set online quantized weight (INT8/FP8).
        
        Args:
            qweight: Quantized weight tensor
            scales: Scale tensor
            zero_points: Optional zero points for asymmetric quantization
        """
        self.register_buffer('quant_weight', qweight)
        self.register_buffer('quant_scales', scales)
        if zero_points is not None:
            self.register_buffer('quant_zero_points', zero_points)
        
        self._weight_is_quantized = True
    
    def set_offline_quantized_weight(self, qweight: torch.Tensor, 
                                     qzeros: torch.Tensor,
                                     scales: torch.Tensor,
                                     g_idx: Optional[torch.Tensor] = None,
                                     bits: int = 4,
                                     group_size: int = 128,
                                     format_type: str = "gptq"):
        """
        Set offline quantized weight (GPTQ/AWQ).
        
        Args:
            qweight: Packed quantized weights (int32)
            qzeros: Packed zero points (int32)
            scales: Scale tensor
            g_idx: Optional group indices
            bits: Quantization bits (4 or 8)
            group_size: Group size for quantization
            format_type: "gptq" or "awq"
        """
        if format_type == "gptq":
            self.register_buffer('gptq_qweight', qweight)
            self.register_buffer('gptq_qzeros', qzeros)
            self.register_buffer('gptq_scales', scales)
            if g_idx is not None:
                self.register_buffer('gptq_g_idx', g_idx)
            self._offline_quant_format = "gptq"
        elif format_type == "awq":
            self.register_buffer('awq_qweight', qweight)
            self.register_buffer('awq_qzeros', qzeros)
            self.register_buffer('awq_scales', scales)
            self._offline_quant_format = "awq"
        
        self._offline_quant_bits = bits
        self._offline_quant_group_size = group_size
    
    def _forward_base(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Unified forward dispatcher.
        
        Routes to appropriate implementation based on quantization state.
        Uses cached forward plan when enabled for minimal Python overhead.
        """
        # Try to use cached plan for minimal Python overhead
        if self._forward_plan_enabled:
            if self._forward_plan is not None:
                if not self._plan_signature_matches(x, bias):
                    self._forward_plan = build_forward_plan(self, x, bias)
            else:
                self._forward_plan = build_forward_plan(self, x, bias)
            
            if self._forward_plan is not None:
                return self._forward_plan(x)
        
        # Fallback: direct dispatch without plan caching
        if self._quant_strategy is None:
            self._quant_strategy = get_linear_strategy(self.quant_kind)
        
        if self.has_offline_quantized_weight():
            return self._forward_offline_quantized(x, bias)
        elif self.has_quantized_weight():
            return self._forward_online_quantized(x, bias)
        else:
            return self._forward_bf16(x, bias)
    
    def _plan_signature_matches(self, x: torch.Tensor, bias: Optional[torch.Tensor]) -> bool:
        """Check if cached plan signature matches current inputs."""
        if self._forward_plan is None:
            return False
        sig = self._forward_plan.get_signature()
        if sig is None:
            return False
        dev = x.device
        return (
            sig.device_type == dev.type
            and sig.device_index == (dev.index if dev.index is not None else 0)
            and sig.x_dtype == x.dtype
            and sig.x_shape == tuple(x.shape)
            and sig.has_bias == (bias is not None)
        )
    
    def _forward_bf16(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard BF16 forward."""
        return F.linear(x, self.weight, bias)
    
    def _forward_online_quantized(self, x: torch.Tensor, 
                                   bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward with online quantized weights."""
        if self._quant_strategy is None:
            return self._forward_bf16(x, bias)
        
        qweight = getattr(self, 'quant_weight', None)
        if qweight is None:
            return self._forward_bf16(x, bias)
        
        return self._quant_strategy.linear_forward(
            x, qweight, bias,
            quant_kind=self.quant_kind,
            quant_scales=getattr(self, 'quant_scales', None),
        )
    
    def _forward_offline_quantized(self, x: torch.Tensor,
                                    bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward with offline quantized weights (GPTQ/AWQ)."""
        if self._quant_strategy is None:
            return self._forward_bf16(x, bias)
        
        fmt = self._offline_quant_format
        kwargs = {'quant_kind': self.quant_kind}
        
        if fmt == "gptq":
            kwargs.update({
                'qweight': getattr(self, 'gptq_qweight', None),
                'qzeros': getattr(self, 'gptq_qzeros', None),
                'scales': getattr(self, 'gptq_scales', None),
                'g_idx': getattr(self, 'gptq_g_idx', None),
                'bits': self._offline_quant_bits,
                'is_shuffled': self._gptq_is_shuffled,
            })
        elif fmt == "awq":
            kwargs.update({
                'qweight': getattr(self, 'awq_qweight', None),
                'qzeros': getattr(self, 'awq_qzeros', None),
                'scales': getattr(self, 'awq_scales', None),
                'bits': self._offline_quant_bits,
            })
        else:
            # Marlin or other formats handled by strategy
            pass
        
        return self._quant_strategy.linear_forward(x, None, bias, **kwargs)
    
    def _maybe_prepare_offline_gptq(self):
        """Prepare GPTQ weights for use (shuffle if needed)."""
        if not self.has_offline_quantized_weight():
            return
        
        if self._offline_quant_format != "gptq" or self._gptq_is_shuffled:
            return
        
        try:
            import vllm._custom_ops as ops
            if hasattr(ops, 'gptq_shuffle'):
                qweight = getattr(self, 'gptq_qweight', None)
                g_idx = getattr(self, 'gptq_g_idx', None)
                if qweight is not None and g_idx is not None:
                    ops.gptq_shuffle(qweight, g_idx, self._offline_quant_bits)
                    self._gptq_is_shuffled = True
        except (ImportError, AttributeError):
            pass
    
    def _maybe_prepare_marlin(self):
        """Prepare Marlin format weights (repack if needed)."""
        if not self.has_offline_quantized_weight():
            return
        
        if self._gptq_marlin_is_prepared:
            return
        
        try:
            import vllm._custom_ops as ops
            
            if self._offline_quant_format == "gptq" and hasattr(ops, 'gptq_marlin_repack'):
                self._repack_gptq_to_marlin(ops)
            elif self._offline_quant_format == "awq" and hasattr(ops, 'awq_marlin_repack'):
                self._repack_awq_to_marlin(ops)
        except (ImportError, AttributeError):
            pass
    
    def _repack_gptq_to_marlin(self, ops):
        """Repack GPTQ weights to Marlin format."""
        qweight = getattr(self, 'gptq_qweight', None)
        qzeros = getattr(self, 'gptq_qzeros', None)
        scales = getattr(self, 'gptq_scales', None)
        g_idx = getattr(self, 'gptq_g_idx', None)
        
        if qweight is None or qzeros is None or scales is None:
            return
        
        marlin_qweight, marlin_scales = ops.gptq_marlin_repack(
            qweight, qzeros, scales, g_idx, self._offline_quant_bits
        )
        
        self.register_buffer('gptq_marlin_qweight', marlin_qweight)
        self.register_buffer('gptq_marlin_scales', marlin_scales)
        
        workspace = torch.zeros(
            marlin_qweight.shape[1] * 32 // self._offline_quant_bits,
            dtype=torch.int32,
            device=qweight.device
        )
        self.register_buffer('gptq_marlin_workspace', workspace)
        
        self._gptq_marlin_is_prepared = True
        self._offline_quant_format = "gptq_marlin"
    
    def _repack_awq_to_marlin(self, ops):
        """Repack AWQ weights to Marlin format."""
        qweight = getattr(self, 'awq_qweight', None)
        qzeros = getattr(self, 'awq_qzeros', None)
        scales = getattr(self, 'awq_scales', None)
        
        if qweight is None or qzeros is None or scales is None:
            return
        
        marlin_qweight, marlin_scales = ops.awq_marlin_repack(
            qweight, qzeros, scales, self._offline_quant_bits
        )
        
        self.register_buffer('awq_marlin_qweight', marlin_qweight)
        self.register_buffer('awq_marlin_scales', marlin_scales)
        
        workspace = torch.zeros(
            marlin_qweight.shape[1] * 32 // self._offline_quant_bits,
            dtype=torch.int32,
            device=qweight.device
        )
        self.register_buffer('awq_marlin_workspace', workspace)
        
        self._gptq_marlin_is_prepared = True
        self._offline_quant_format = "awq_marlin"
