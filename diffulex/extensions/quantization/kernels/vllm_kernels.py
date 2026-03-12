"""
vLLM Kernel Wrappers

Wrapper classes for vLLM optimized kernels following the BaseKernel interface.
"""

from typing import Any, Optional, Callable
import torch

from .kernel_registry import LinearKernel, KVCacheKernel
from .kernel_registry import register_kernel as _register
from .kernel_availability import check_vllm_op_available, warn_kernel_unavailable


class VllmKernelBase:
    """Base class for vLLM kernels."""
    
    _op_name: str = ""
    _op: Optional[Callable] = None
    _checked: bool = False
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if vLLM op is available."""
        if not cls._checked:
            cls._op = None
            if check_vllm_op_available(cls._op_name):
                try:
                    import vllm._custom_ops as ops
                    cls._op = getattr(ops, cls._op_name, None)
                except ImportError:
                    pass
            cls._checked = True
        return cls._op is not None
    
    @classmethod
    def get_missing_reason(cls) -> Optional[str]:
        """Get reason why kernel is unavailable."""
        if cls.is_available():
            return None
        return f"vLLM op '{cls._op_name}' not available. Install vLLM with CUDA support."
    
    def __init__(self):
        if not self.is_available():
            raise RuntimeError(f"{self.__class__.__name__} is not available")


@_register("vllm_gptq_gemm")
class VllmGPTQGemm(VllmKernelBase, LinearKernel):
    """GPTQ GEMM kernel (W2/W3/W4/W8)."""
    
    name = "vllm_gptq_gemm"
    description = "GPTQ GEMM for 2/3/4/8-bit weights"
    _op_name = "gptq_gemm"
    
    def forward(self, x: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor,
                scales: torch.Tensor, g_idx: torch.Tensor, is_shuffled: bool,
                bits: int) -> torch.Tensor:
        """Execute GPTQ GEMM."""
        return self._op(x, qweight, qzeros, scales, g_idx, is_shuffled, bits)


@_register("vllm_awq_gemm")
class VllmAWQGemm(VllmKernelBase, LinearKernel):
    """AWQ GEMM kernel (W4)."""
    
    name = "vllm_awq_gemm"
    description = "AWQ GEMM for 4-bit weights"
    _op_name = "awq_gemm"
    
    def forward(self, x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor,
                qzeros: torch.Tensor, M: int, N: int, K: int,
                group_size: int, bits: int) -> torch.Tensor:
        """Execute AWQ GEMM."""
        return self._op(x, qweight, scales, qzeros, M, N, K, group_size, bits)


@_register("vllm_marlin_gemm")
class VllmMarlinGemm(VllmKernelBase, LinearKernel):
    """Marlin GEMM kernel (W4/W8)."""
    
    name = "vllm_marlin_gemm"
    description = "Marlin GEMM for 4/8-bit weights"
    _op_name = "gptq_marlin_gemm"
    
    def forward(self, x: torch.Tensor, marlin_weight: torch.Tensor,
                marlin_scales: torch.Tensor, g_idx: torch.Tensor,
                sort_indices: torch.Tensor, workspace: torch.Tensor,
                M: int, N: int, K: int) -> torch.Tensor:
        """Execute Marlin GEMM."""
        return self._op(x, marlin_weight, marlin_scales, g_idx, sort_indices,
                       workspace, M, N, K)


@_register("vllm_cutlass_scaled_mm")
class VllmCutlassScaledMM(VllmKernelBase, LinearKernel):
    """CUTLASS scaled MM for INT8 W8A8."""
    
    name = "vllm_cutlass_scaled_mm"
    description = "CUTLASS INT8 W8A8 GEMM"
    _op_name = "cutlass_scaled_mm"
    
    def forward(self, a: torch.Tensor, b: torch.Tensor,
                scale_a: torch.Tensor, scale_b: torch.Tensor,
                out_dtype: torch.dtype) -> torch.Tensor:
        """Execute CUTLASS scaled MM."""
        return self._op(a, b, scale_a, scale_b, out_dtype)


@_register("vllm_allspark_w8a16")
class VllmAllSparkW8A16(VllmKernelBase, LinearKernel):
    """AllSpark W8A16 kernel (Ampere+)."""
    
    name = "vllm_allspark_w8a16"
    description = "AllSpark W8A16 GEMM for Ampere+ GPUs"
    _op_name = "allspark_w8a16_gemm"
    
    def forward(self, a: torch.Tensor, b_qweight: torch.Tensor,
                b_scales: torch.Tensor, b_qzeros: Optional[torch.Tensor],
                n: int, group_size: int, sm_count: int, sm_version: int,
                CUBLAS_M_THRESHOLD: int, has_zp: bool,
                n32k16_reorder: bool) -> torch.Tensor:
        """Execute AllSpark W8A16 GEMM."""
        return self._op(a, b_qweight, b_scales, b_qzeros, n, group_size,
                       sm_count, sm_version, CUBLAS_M_THRESHOLD, has_zp,
                       n32k16_reorder)


@_register("vllm_cutlass_w4a8")
class VllmCutlassW4A8(VllmKernelBase, LinearKernel):
    """CUTLASS W4A8 kernel (Hopper+)."""
    
    name = "vllm_cutlass_w4a8"
    description = "CUTLASS W4A8 GEMM for Hopper+ GPUs"
    _op_name = "cutlass_w4a8_mm"
    
    def forward(self, a: torch.Tensor, b_q: torch.Tensor,
                b_group_scales: torch.Tensor, b_group_size: int,
                a_token_scales: torch.Tensor,
                b_channel_scales: torch.Tensor) -> torch.Tensor:
        """Execute CUTLASS W4A8 GEMM."""
        return self._op(a, b_q, b_group_scales, b_group_size,
                       a_token_scales, b_channel_scales)


@_register("vllm_fp8_linear")
class VllmFp8LinearOp(VllmKernelBase, LinearKernel):
    """FP8 linear operation."""
    
    name = "vllm_fp8_linear"
    description = "vLLM FP8 linear with dynamic scaling"
    _op_name = "scaled_fp8_quant"
    
    def __init__(self, cutlass_fp8_supported: bool = True,
                 use_per_token_if_dynamic: bool = True):
        super().__init__()
        from vllm.model_executor.layers.quantization.utils.fp8_utils import Fp8LinearOp
        self._op = Fp8LinearOp(
            cutlass_fp8_supported=cutlass_fp8_supported,
            use_per_token_if_dynamic=use_per_token_if_dynamic,
        )
    
    def forward(self, x: torch.Tensor, w: torch.Tensor,
                w_scale: torch.Tensor, bias: Optional[torch.Tensor],
                act_dtype: torch.dtype) -> torch.Tensor:
        """Execute FP8 linear forward."""
        return self._op.apply(x, w, w_scale, bias, act_dtype)
