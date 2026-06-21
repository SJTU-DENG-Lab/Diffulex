from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_add_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    base = row * n_cols + offsets

    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + base, mask=mask, other=0.0)
    variance = tl.sum(x * x, axis=0) / n_cols
    y = x * tl.rsqrt(variance + eps)
    if HAS_WEIGHT:
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        y = y * weight
    y = y + residual
    tl.store(out_ptr + base, y, mask=mask)


def rms_norm_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
) -> torch.Tensor | None:
    """Compute RMSNorm(x) + residual for contiguous CUDA tensors."""
    if not x.is_cuda or not residual.is_cuda:
        return None
    if x.shape != residual.shape or x.dim() < 1:
        return None
    if not x.is_contiguous() or not residual.is_contiguous():
        return None
    if weight is not None and (not weight.is_cuda or weight.numel() != x.shape[-1]):
        return None
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None

    orig_shape = x.shape
    n_cols = int(orig_shape[-1])
    if n_cols <= 0:
        return None
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 8192:
        return None

    x_2d = x.view(-1, n_cols)
    residual_2d = residual.view(-1, n_cols)
    out = torch.empty_like(x_2d)
    weight_arg = weight if weight is not None else x

    _rms_norm_add_kernel[(x_2d.shape[0],)](
        x_2d,
        residual_2d,
        weight_arg,
        out,
        n_cols,
        float(eps),
        HAS_WEIGHT=weight is not None,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return out.view(orig_shape)
