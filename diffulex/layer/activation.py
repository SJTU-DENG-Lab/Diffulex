import torch

import torch.nn as nn
import torch.nn.functional as F


def _get_sgl_kernel_silu_and_mul():
    namespace = getattr(torch.ops, "sgl_kernel", None)
    op = getattr(namespace, "silu_and_mul", None) if namespace is not None else None
    if op is not None:
        return op
    try:
        import sgl_kernel

        return getattr(sgl_kernel, "silu_and_mul", None)
    except Exception:
        return None


class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()

    def _forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        op = _get_sgl_kernel_silu_and_mul()
        if op is None or not x.is_cuda or x.shape[-1] % 2 != 0:
            return self._forward_reference(x)
        out = torch.empty(x.shape[:-1] + (x.shape[-1] // 2,), dtype=x.dtype, device=x.device)
        op(x.contiguous(), out)
        return out
