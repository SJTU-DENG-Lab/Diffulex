import os

import torch
import torch.nn as nn

from diffulex.layer.vllm_backend import get_vllm_rmsnorm_cls


def _use_reference_rmsnorm() -> bool:
    return os.getenv("DIFFULEX_REFERENCE_RMSNORM", "0") == "1"


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        has_weight: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.has_weight = bool(has_weight)
        vllm_cls = get_vllm_rmsnorm_cls()
        if vllm_cls is not None:
            try:
                impl = vllm_cls(hidden_size, eps=eps, has_weight=self.has_weight)
                object.__setattr__(self, "_vllm_impl", impl)
                if self.has_weight:
                    self.weight = impl.weight
                else:
                    self.register_parameter("weight", None)
                return
            except Exception:
                object.__setattr__(self, "_vllm_impl", None)
        else:
            object.__setattr__(self, "_vllm_impl", None)
        if self.has_weight:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype)
        if self.weight is not None:
            x = x.mul_(self.weight)
        return x

    def rms_forward_reference(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(var + self.eps)
        x_out = x_fp32.to(orig_dtype)
        if self.weight is not None:
            x_out = x_out * self.weight
        return x_out

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.to(torch.float32).add_(residual.to(torch.float32))
        residual = x.to(orig_dtype).clone()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype)
        if self.weight is not None:
            x = x.mul_(self.weight)
        return x, residual

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
        x_out = x_fp32.to(orig_dtype)
        if self.weight is not None:
            x_out = x_out * self.weight
        return x_out, residual_out

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        vllm_impl = getattr(self, "_vllm_impl", None)
        if vllm_impl is not None and not _use_reference_rmsnorm():
            return vllm_impl(x, residual)
        if _use_reference_rmsnorm():
            if residual is None:
                return self.rms_forward_reference(x)
            return self.add_rms_forward_reference(x, residual)
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
