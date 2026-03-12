"""
INT8 W8A16 Linear Strategy - vLLM-aligned high-performance implementation.

Key optimizations (from feat/kv-cache-fp8-support):
- Uses vLLM's allspark_w8a16_gemm for fused INT8 weight + BF16 activation GEMM
- allspark_repack_weight for N32K16 weight layout (optimal for Ampere+)
- Per-output-channel symmetric int8 quantization stored as uint8 (+128 bias)
- Streaming quantization to avoid OOM

No fallback - requires vLLM custom ops.
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

from ..strategy import LinearQuantizationStrategy
from ..registry import register_linear_strategy


try:
    import vllm._custom_ops as _vllm_ops
except Exception:
    _vllm_ops = None


def _allspark_is_available() -> bool:
    """Check if AllSpark W8A16 kernel is available."""
    return bool(
        _vllm_ops is not None
        and hasattr(_vllm_ops, "allspark_w8a16_gemm")
        and hasattr(_vllm_ops, "allspark_repack_weight")
    )


def _allspark_repack_weight(b_qweight_kn: torch.Tensor, scales_1xn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Repack K×N uint8 qweight + 1×N scales into (N_32,K) + (1,N_32) for AllSpark GEMM."""
    if _vllm_ops is None or not hasattr(_vllm_ops, "allspark_repack_weight"):
        raise RuntimeError("vLLM custom ops unavailable: missing allspark_repack_weight")
    q_reorder, s_reorder, _ = _vllm_ops.allspark_repack_weight(
        b_qweight_kn,
        scales_1xn,
        None,
        False,
    )
    return q_reorder, s_reorder


@register_linear_strategy("int8", "bf16")
class INT8W8A16LinearStrategy(LinearQuantizationStrategy):
    """
    INT8 W8A16 linear quantization using vLLM's AllSpark fused kernel.
    
    Weight layout: stored as [N_32, K] uint8 in N32K16 reorder format
    Scale layout: [N_32] bf16 (reordered/padded)
    """
    
    def __init__(self):
        super().__init__()
        # Cache for bf16 Parameters (dynamic quantization path)
        self._weight_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        # Cache device info
        self._sm_info_cache: Dict[int, Tuple[int, int]] = {}
        # Config
        self._quant_block_n: int = 256
        self._cublas_m_thr: int = 256
        
        # Check availability once and fail fast
        if not _allspark_is_available():
            raise RuntimeError(
                "vLLM AllSpark W8A16 fused kernel is unavailable. "
                "Please ensure vLLM custom ops are installed with allspark support."
            )
    
    @property
    def name(self) -> str:
        return "linear_int8_w8a16"
    
    @property
    def linear_weight_format(self) -> str:
        return "int8"
    
    @property
    def linear_act_format(self) -> str:
        return "bf16"
    
    def get_storage_dtype(self, device: torch.device) -> Tuple[torch.dtype, int]:
        return (torch.uint8, 1)
    
    def get_scale_shape(self, original_shape: Tuple[int, ...], **kwargs: Any) -> Tuple[int, ...]:
        """Return scale shape for weight quantization."""
        if len(original_shape) < 2:
            raise ValueError(f"Expected weight shape with at least 2 dims, got {original_shape}")
        return (original_shape[0],)  # Per-output-channel
    
    def quantize(self, weight: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, Any]:
        """
        Reference per-output-channel symmetric int8 quantization.
        
        Returns:
            quantized_int8: [N, K] int8
            scales: [N] bf16
        """
        if weight.dim() != 2:
            raise ValueError(f"Expected 2D weight [N,K], got shape={tuple(weight.shape)}")
        if weight.dtype != torch.bfloat16:
            weight = weight.to(dtype=torch.bfloat16)
        
        abs_max = torch.abs(weight).max(dim=-1, keepdim=True)[0]  # [N,1]
        scales = (abs_max.clamp(min=1e-8) / 127.0).to(dtype=torch.bfloat16)  # [N,1]
        q = torch.round(weight.to(torch.float32) / scales.to(torch.float32)).clamp(-128, 127).to(torch.int8)
        
        return q, scales.squeeze(-1)
    
    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs: Any) -> torch.Tensor:
        """Dequantize - NOT SUPPORTED to prevent slow fallback."""
        raise RuntimeError(
            "W8A16 does not provide dequantize path (avoid slow BF16 GEMM). "
            "Use allspark_w8a16_gemm only."
        )
    
    def quantize_weight_for_kernel(
        self,
        weight: torch.Tensor,
        *,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Quantize+repack bf16 weight for AllSpark fused kernel.
        
        Input:
            weight: [N, K] bf16/fp16
        Output:
            qweight_reorder: [N_32align, K] uint8 in N32K16 reorder layout
            scales_reorder: [N_32align] bf16 scales
        """
        if device is not None:
            weight = weight.to(device=device)
        
        if weight.dim() != 2:
            raise ValueError(f"Expected 2D weight [N,K], got shape={tuple(weight.shape)}")
        
        if weight.dtype != torch.bfloat16:
            weight = weight.to(dtype=torch.bfloat16)
        
        n, k = weight.shape
        
        # Per-output-channel symmetric scale
        abs_max = torch.abs(weight).max(dim=-1)[0]  # [N]
        scales = (abs_max.clamp(min=1e-8) / 127.0).to(dtype=torch.bfloat16)  # [N]
        
        # AllSpark repack expects B in (K,N) contiguous layout
        b_kn = torch.empty((k, n), device=weight.device, dtype=torch.uint8)  # [K,N]
        
        # Quantize in blocks to avoid OOM
        block_n = max(1, int(self._quant_block_n))
        for i in range(0, n, block_n):
            j = min(i + block_n, n)
            w_blk = weight[i:j, :]  # [B,K]
            s_blk = scales[i:j].unsqueeze(-1)  # [B,1]
            # Quantize to signed int
            q_i16 = torch.round(w_blk / s_blk).clamp(-128, 127).to(torch.int16)  # [B,K]
            q_u8_blk = (q_i16 + 128).to(torch.uint8)  # [B,K]
            # Write directly into [K,N] buffer
            b_kn[:, i:j] = q_u8_blk.transpose(0, 1)
        
        # Repack to N32K16 format
        q_reorder, s_reorder_1xn = _allspark_repack_weight(
            b_kn.contiguous(),
            scales.unsqueeze(0).contiguous(),
        )
        
        # Store scales as 1D
        s_1d = s_reorder_1xn.reshape(-1).to(dtype=torch.bfloat16)
        return q_reorder.contiguous(), s_1d.contiguous()
    
    def quantize_act_for_kernel(self, x: torch.Tensor,
                                cache_key: Optional[str] = None) -> Tuple[torch.Tensor, Any]:
        """No activation quantization in W8A16 - keep BF16."""
        return x, None
    
    def _get_sm_info(self, device: torch.device) -> Tuple[int, int]:
        """Get SM count and version for device (cached)."""
        try:
            idx = int(device.index) if device.index is not None else int(torch.cuda.current_device())
        except Exception:
            idx = -1
        
        cached = self._sm_info_cache.get(idx)
        if cached is not None:
            return cached
        
        try:
            props = torch.cuda.get_device_properties(device)
            sm_count = int(getattr(props, "multi_processor_count", 0))
            sm_version = int(props.major) * 10 + int(props.minor)
            self._sm_info_cache[idx] = (sm_count, sm_version)
            return sm_count, sm_version
        except Exception:
            self._sm_info_cache[idx] = (0, 0)
            return 0, 0
    
    def linear_forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        quant_kind: str = "other",
        quant_scales: Optional[torch.Tensor] = None,
        out_features: Optional[int] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        INT8 W8A16 linear forward using vLLM's allspark_w8a16_gemm.
        
        Args:
            x: Input tensor [..., K] (BF16)
            weight: Quantized weight [N_32, K] uint8, or original weight
            bias: Optional bias [N]
            quant_scales: Weight scales [N_32] bf16
            out_features: Original N (before padding)
            
        Returns:
            output: [..., N]
        """
        orig_shape = x.shape
        x2 = x.reshape(-1, x.shape[-1]) if x.dim() != 2 else x
        
        if x2.device.type != "cuda":
            raise RuntimeError("AllSpark W8A16 requires CUDA inputs")
        
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(dtype=torch.bfloat16)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        
        # Load-time quantized path
        if weight is not None and weight.dtype in (torch.uint8, torch.int8):
            if quant_scales is None:
                raise ValueError("quant_scales is required when weight is quantized")
            qweight = weight
            scales = quant_scales
        else:
            # Dynamic quantization (should be rare)
            weight_id = id(weight)
            cached = self._weight_cache.get(weight_id)
            if cached is None or cached[0].device != x2.device:
                qweight, scales = self.quantize_weight_for_kernel(weight, device=x2.device)
                self._weight_cache[weight_id] = (qweight, scales)
            else:
                qweight, scales = cached
        
        m, k = x2.shape
        n_32, k_w = qweight.shape
        
        if k_w != k or (k & 15) != 0:
            raise RuntimeError(
                f"AllSpark W8A16 requires K%16==0 and matching K. Got x.K={k}, w.K={k_w}"
            )
        
        n = int(out_features) if out_features is not None else (
            int(bias.numel()) if bias is not None else int(min(scales.numel(), n_32))
        )
        n = n_32 if (n <= 0 or n > n_32) else n
        
        scales_1xn = scales if scales.dim() == 2 else scales.view(1, -1)
        
        sm_count, sm_version = self._get_sm_info(x2.device)
        
        # Call allspark kernel
        y2 = _vllm_ops.allspark_w8a16_gemm(
            x2,
            qweight,
            scales_1xn,
            None,  # b_qzeros
            n,
            -1,  # group_size (only supports -1)
            sm_count,
            sm_version,
            self._cublas_m_thr,
            False,  # has_zp
            True,   # n32k16_reorder
        )
        
        if bias is not None:
            y2 = y2 + bias
        
        if orig_shape == x2.shape:
            return y2
        if x.dim() == 1:
            return y2.squeeze(0)
        return y2.reshape(*orig_shape[:-1], y2.shape[-1])
