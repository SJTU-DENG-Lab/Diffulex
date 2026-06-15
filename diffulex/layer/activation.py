import torch.nn.functional as F
import torch
import torch.nn as nn

from diffulex.layer.vllm_backend import get_vllm_gelu_and_mul_cls, get_vllm_silu_and_mul_cls


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
        self._vllm_impl = vllm_cls() if vllm_cls is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._vllm_impl is not None:
            return self._vllm_impl(x)
        return _silu_and_mul_native(x)


class GeluAndMul(nn.Module):
    def __init__(self):
        super().__init__()
        vllm_cls = get_vllm_gelu_and_mul_cls()
        self._vllm_impl = vllm_cls(approximate="tanh") if vllm_cls is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._vllm_impl is not None:
            return self._vllm_impl(x)
        return _gelu_and_mul_native(x)
