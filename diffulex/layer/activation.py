import torch.nn.functional as F
import torch
import torch.nn as nn

from diffulex.layer.vllm_backend import get_vllm_gelu_and_mul_cls, get_vllm_silu_and_mul_cls


def _vllm_silu_and_mul_direct(x: torch.Tensor) -> torch.Tensor | None:
    if not x.is_cuda or not x.is_contiguous():
        return None
    try:
        import vllm._custom_ops  # noqa: F401
    except Exception:
        pass
    if not hasattr(torch.ops, "_C") or not hasattr(torch.ops._C, "silu_and_mul"):
        return None
    out = torch.empty(x.shape[:-1] + (x.shape[-1] // 2,), device=x.device, dtype=x.dtype)
    torch.ops._C.silu_and_mul(out, x)
    return out


def _vllm_gelu_tanh_and_mul_direct(x: torch.Tensor) -> torch.Tensor | None:
    if not x.is_cuda or not x.is_contiguous():
        return None
    try:
        import vllm._custom_ops  # noqa: F401
    except Exception:
        pass
    if not hasattr(torch.ops, "_C") or not hasattr(torch.ops._C, "gelu_tanh_and_mul"):
        return None
    out = torch.empty(x.shape[:-1] + (x.shape[-1] // 2,), device=x.device, dtype=x.dtype)
    torch.ops._C.gelu_tanh_and_mul(out, x)
    return out


@torch.compile
def _silu_and_mul_native(x: torch.Tensor) -> torch.Tensor:
    x, y = x.chunk(2, -1)
    return F.silu(x) * y


@torch.compile
def _gelu_and_mul_native(x: torch.Tensor) -> torch.Tensor:
    x, y = x.chunk(2, -1)
    return F.gelu(x, approximate="tanh") * y


class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()
        vllm_cls = get_vllm_silu_and_mul_cls()
        self._vllm_direct_enabled = vllm_cls is not None
        self._vllm_impl = None
        if vllm_cls is not None:
            try:
                self._vllm_impl = vllm_cls()
            except Exception:
                self._vllm_impl = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._vllm_direct_enabled:
            out = _vllm_silu_and_mul_direct(x)
            if out is not None:
                return out
        if self._vllm_impl is not None:
            return self._vllm_impl(x)
        return _silu_and_mul_native(x)


class GeluAndMul(nn.Module):
    def __init__(self):
        super().__init__()
        vllm_cls = get_vllm_gelu_and_mul_cls()
        self._vllm_direct_enabled = vllm_cls is not None
        self._vllm_impl = None
        if vllm_cls is not None:
            try:
                self._vllm_impl = vllm_cls(approximate="tanh")
            except Exception:
                self._vllm_impl = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._vllm_direct_enabled:
            out = _vllm_gelu_tanh_and_mul_direct(x)
            if out is not None:
                return out
        if self._vllm_impl is not None:
            return self._vllm_impl(x)
        return _gelu_and_mul_native(x)
