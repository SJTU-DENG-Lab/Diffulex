import os

import torch
import torch.nn as nn


def _use_reference_rmsnorm() -> bool:
    return os.getenv("DIFFULEX_REFERENCE_RMSNORM", "0") == "1"


def _get_sgl_kernel_op(name: str):
    namespace = getattr(torch.ops, "sgl_kernel", None)
    op = getattr(namespace, name, None) if namespace is not None else None
    if op is not None:
        return op
    try:
        import sgl_kernel

        return getattr(sgl_kernel, name, None)
    except Exception:
        return None


def _reshape_to_2d(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    shape = x.shape
    return x.reshape(-1, shape[-1]), shape


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _can_use_kernel(self, x: torch.Tensor) -> bool:
        return x.is_cuda and self.weight.is_cuda and not _use_reference_rmsnorm()

    def _rms_forward_kernel(self, x: torch.Tensor) -> torch.Tensor | None:
        op = _get_sgl_kernel_op("rmsnorm")
        if op is None or not self._can_use_kernel(x):
            return None
        x_2d, original_shape = _reshape_to_2d(x.contiguous())
        return op(x_2d, self.weight.data, self.eps).reshape(original_shape)

    def _add_rms_forward_kernel(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        op = _get_sgl_kernel_op("fused_add_rmsnorm")
        if op is None or not self._can_use_kernel(x) or not residual.is_cuda:
            return None
        x_2d, original_shape = _reshape_to_2d(x.contiguous())
        residual_2d = residual.contiguous().reshape_as(x_2d)
        op(x_2d, residual_2d, self.weight.data, self.eps)
        return x_2d.reshape(original_shape), residual_2d.reshape(original_shape)

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(var + self.eps)
        return x_fp32.to(orig_dtype) * self.weight

    def rms_forward_reference(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(var + self.eps)
        return x_fp32.to(orig_dtype) * self.weight

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32) + residual.to(torch.float32)
        residual_out = x_fp32.to(orig_dtype)
        var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(var + self.eps)
        return x_fp32.to(orig_dtype) * self.weight, residual_out

    def add_rms_forward_reference(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32) + residual.to(torch.float32)
        residual_out = x_fp32.to(orig_dtype)
        var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(var + self.eps)
        return x_fp32.to(orig_dtype) * self.weight, residual_out

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            kernel_out = self._rms_forward_kernel(x)
            if kernel_out is not None:
                return kernel_out
            if _use_reference_rmsnorm():
                return self.rms_forward_reference(x)
            return self.rms_forward(x)

        kernel_out = self._add_rms_forward_kernel(x, residual)
        if kernel_out is not None:
            return kernel_out
        if _use_reference_rmsnorm():
            return self.add_rms_forward_reference(x, residual)
        return self.add_rms_forward(x, residual)
