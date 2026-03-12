"""
Cutlass W4A8 Linear Strategy

4-bit weight + FP8 E4M3 activation quantization.
Uses vLLM's CUTLASS W4A8 kernel for optimized inference on Hopper (SM90+).

Requirements:
- Hopper GPU (compute capability 90+)
- group_size = 128
- K and N must be divisible by 128
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

from ..strategy import LinearQuantizationStrategy
from ..registry import register_linear_strategy
from ..kernels.kernel_availability import warn_kernel_unavailable, check_vllm_op_available


# FP8 constants for activation quantization
FP8_E4M3_MAX = 448.0


def fp8_quantize_per_token(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamic per-token FP8 quantization.
    
    Args:
        x: Input tensor [M, K]
        
    Returns:
        Tuple of (quantized tensor, scales per token)
    """
    # Per-token max
    x_max = x.abs().max(dim=-1, keepdim=True)[0].float()
    scale = x_max / FP8_E4M3_MAX
    scale = torch.clamp(scale, min=1e-12)
    x_fp8 = (x / scale).to(torch.float8_e4m3fn)
    return x_fp8, scale


def fp8_dequantize_per_token(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 tensor with per-token scales."""
    return x_fp8.to(torch.bfloat16) * scale.to(torch.bfloat16)


@register_linear_strategy("cutlass_w4a8", "fp8_e4m3")
class CutlassW4A8LinearStrategy(LinearQuantizationStrategy):
    """
    CUTLASS W4A8 linear quantization.
    
    4-bit weights with FP8 E4M3 activations.
    Optimized for Hopper GPUs (SM90+).
    
    Requirements:
    - Hopper GPU (SM90+)
    - group_size = 128 (fixed)
    - in_features % 128 == 0
    - out_features % 128 == 0
    """
    
    def __init__(self, group_size: int = 128):
        if group_size != 128:
            raise ValueError(f"CutlassW4A8 only supports group_size=128, got {group_size}")
        
        self.bits = 4
        self.group_size = group_size
        
        # Check for CUTLASS W4A8 ops
        self.cutlass_w4a8_mm = None
        self.cutlass_encode_int4 = None
        self.cutlass_pack_scale = None
        self._kernel_warned = False
        
        if check_vllm_op_available('cutlass_w4a8_mm'):
            import vllm._custom_ops as ops
            self.cutlass_w4a8_mm = ops.cutlass_w4a8_mm
            if hasattr(ops, 'cutlass_encode_and_reorder_int4b'):
                self.cutlass_encode_int4 = ops.cutlass_encode_and_reorder_int4b
            if hasattr(ops, 'cutlass_pack_scale_fp8'):
                self.cutlass_pack_scale = ops.cutlass_pack_scale_fp8
        # Note: Warning is deferred to first forward call
    
    @property
    def name(self) -> str:
        return f"cutlass_w4a8_g{self.group_size}"
    
    @property
    def linear_weight_format(self) -> str:
        return "int4"
    
    @property
    def linear_act_format(self) -> str:
        return "fp8_e4m3"
    
    def get_storage_dtype(self, device: torch.device) -> Tuple[torch.dtype, int]:
        # 4-bit packed into int32
        return (torch.int32, 0)
    
    def _check_gpu_compatibility(self, device: torch.device) -> bool:
        """Check if GPU supports CUTLASS W4A8 (Hopper+)."""
        if not device.type == 'cuda':
            return False
        
        properties = torch.cuda.get_device_properties(device)
        sm_version = properties.major * 10 + properties.minor
        
        # Requires SM90+ (Hopper)
        return sm_version >= 90
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize activation to FP8 E4M3 (per-token)."""
        return fp8_quantize_per_token(x)
    
    def dequantize(self, q_x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize from FP8 E4M3."""
        return fp8_dequantize_per_token(q_x, scale)
    
    def quantize_weight_for_kernel(self, weight: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Quantize weight for CUTLASS W4A8 kernel.
        
        Note: This is for runtime quantization. For pre-quantized weights,
        use prepare_cutlass_weight() during loading.
        """
        raise NotImplementedError(
            "CUTLASS W4A8 requires offline weight quantization with special packing. "
            "Use prepare_cutlass_weight() during weight loading."
        )
    
    def quantize_act_for_kernel(self, x: torch.Tensor,
                                cache_key: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize activation to FP8 E4M3 (per-token)."""
        x_fp8, scale = fp8_quantize_per_token(x)
        return x_fp8, {'scale': scale}
    
    def linear_forward(self, x: torch.Tensor, weight: torch.Tensor,
                       bias: Optional[torch.Tensor] = None,
                       *, quant_kind: str = "other", **kwargs) -> torch.Tensor:
        """
        CUTLASS W4A8 linear forward.
        
        Args:
            x: Input tensor [..., K] (BF16)
            weight: Not used directly, use packed weights from kwargs
            bias: Optional bias
            
        Keyword Args:
            qweight: Packed int4 weights (prepared by CUTLASS)
            scales: FP8 packed group scales
            chan_scales: Per-channel scales (BF16)
        """
        qweight = kwargs.get('qweight')
        scales = kwargs.get('scales')
        chan_scales = kwargs.get('chan_scales')
        
        if qweight is None:
            raise ValueError("CUTLASS W4A8 forward requires 'qweight' buffer")
        if scales is None:
            raise ValueError("CUTLASS W4A8 forward requires 'scales' buffer")
        if chan_scales is None:
            raise ValueError("CUTLASS W4A8 forward requires 'chan_scales' buffer")
        
        # Check GPU compatibility
        if not self._check_gpu_compatibility(x.device):
            if not self._kernel_warned:
                warn_kernel_unavailable(
                    "cutlass_w4a8_mm (Hopper SM90+)",
                    self.name,
                    "dequantize + FP8 matmul (slow fallback)"
                )
                self._kernel_warned = True
        
        # Use CUTLASS kernel if available and GPU is compatible
        if self.cutlass_w4a8_mm is not None and self._check_gpu_compatibility(x.device):
            try:
                # Quantize activation to FP8 (per-token)
                x_2d = x.reshape(-1, x.shape[-1])
                x_fp8, act_scales = fp8_quantize_per_token(x_2d)
                
                # Call CUTLASS W4A8 kernel
                output = self.cutlass_w4a8_mm(
                    a=x_fp8,
                    b_q=qweight,
                    b_group_scales=scales,
                    b_group_size=self.group_size,
                    a_token_scales=act_scales,
                    b_channel_scales=chan_scales,
                )
                
                output_shape = list(x.shape[:-1]) + [chan_scales.shape[0]]
                output = output.reshape(output_shape)
                
                if bias is not None:
                    output = output + bias
                
                return output
            except Exception as e:
                if not self._kernel_warned:
                    warn_kernel_unavailable(
                        "vllm.cutlass_w4a8_mm",
                        self.name,
                        f"fallback matmul (error: {e})"
                    )
                    self._kernel_warned = True
        else:
            if not self._kernel_warned:
                reason = "kernel unavailable"
                if not self._check_gpu_compatibility(x.device):
                    reason = "requires Hopper GPU (SM90+)"
                warn_kernel_unavailable(
                    "vllm.cutlass_w4a8_mm",
                    self.name,
                    f"dequantize + FP8 matmul ({reason})"
                )
                self._kernel_warned = True
        
        # Fallback: manual dequantize + FP8 matmul
        # This is very slow and should be avoided
        return self._fallback_forward(x, qweight, scales, chan_scales, bias)
    
    def _fallback_forward(self, x: torch.Tensor, qweight: torch.Tensor,
                         scales: torch.Tensor, chan_scales: torch.Tensor,
                         bias: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Fallback forward using manual dequantization.
        Very slow, only for compatibility.
        """
        # Dequantize weights (simplified - actual implementation is complex)
        # This is a placeholder - real implementation would need proper int4 unpacking
        raise RuntimeError(
            f"[{self.name}] CUTLASS W4A8 kernel is required. "
            f"Manual dequantization is not supported for W4A8 format. "
            f"Please ensure you have: (1) Hopper GPU (SM90+), "
            f"(2) vLLM with CUDA support, (3) Valid CUTLASS W4A8 kernel."
        )
    
    def prepare_cutlass_weight(self, qweight: torch.Tensor, scales: torch.Tensor,
                               device: torch.device) -> Dict[str, Any]:
        """
        Prepare weight for CUTLASS W4A8 kernel.
        
        This should be called during weight loading to prepare weights
        in the format expected by CUTLASS.
        
        Args:
            qweight: Packed int4 weights [K/8, N] int32
            scales: Group scales [K/group_size, N] BF16
            device: Target device
            
        Returns:
            Dict with prepared buffers:
            - qweight: Encoded and reordered int4 weights
            - scales: FP8 packed group scales
            - chan_scales: Per-channel scales (BF16)
        """
        if self.cutlass_encode_int4 is None or self.cutlass_pack_scale is None:
            raise RuntimeError(
                f"[{self.name}] CUTLASS encoding ops not available. "
                "Please install vLLM with CUDA support."
            )
        
        try:
            # Move to device
            qweight = qweight.to(device)
            scales = scales.to(device)
            
            # Convert uint4 to signed int4 in-place
            # (This is done by the caller or we need to do it here)
            
            # Encode and reorder weights
            # Input: [K/8, N] int32 packed
            # Output: [K/8, N] int32 encoded and reordered
            qweight_encoded = self.cutlass_encode_int4(
                qweight.t().contiguous().t()
            )
            
            # Convert scales to FP8 and pack
            # scales: [K/group_size, N] BF16 -> FP8 E4M3
            scales_fp8 = scales.to(torch.float8_e4m3fn)
            scales_packed = self.cutlass_pack_scale(scales_fp8)
            
            # Compute per-channel scales (for output scaling)
            # This is derived from group scales
            chan_scales = scales.mean(dim=0, keepdim=True).expand(scales.shape[1])
            
            return {
                'qweight': qweight_encoded,
                'scales': scales_packed,
                'chan_scales': chan_scales,
            }
        except Exception as e:
            raise RuntimeError(
                f"[{self.name}] Failed to prepare CUTLASS W4A8 weights: {e}"
            ) from e
