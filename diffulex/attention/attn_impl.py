import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from diffulex_kernel import (
    store_kv_cache_distinct_layout,
    store_kv_cache_unified_layout,
    chunked_prefill_attn_unified,
    chunked_prefill_attn_grouped_unified,
)
from diffulex.attention.metadata import AttnMetaDataBase


ATTN_IMPLS = {"naive", "triton", "triton_grouped"}


def reference_torch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
    mask: torch.Tensor | None = None,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Torch attention reference path for debugging numerical drift."""
    q_ref = q.transpose(0, 1).contiguous()  # [nh, s, hd]
    k_ref = k.transpose(0, 1).contiguous()  # [nkvh, s, hd]
    v_ref = v.transpose(0, 1).contiguous()  # [nkvh, s, hd]
    if num_kv_heads != num_heads:
        repeat_factor = num_heads // num_kv_heads
        k_ref = k_ref.repeat_interleave(repeat_factor, dim=0)
        v_ref = v_ref.repeat_interleave(repeat_factor, dim=0)
    scores = torch.matmul(
        q_ref.to(torch.float32),
        k_ref.transpose(-1, -2).to(torch.float32),
    ) * scale
    if sliding_window is not None and int(sliding_window) > 0:
        q_pos = torch.arange(q_ref.shape[1], device=q_ref.device)
        k_pos = torch.arange(k_ref.shape[1], device=k_ref.device)
        scores = scores.masked_fill((q_pos[:, None] - k_pos[None, :]) >= int(sliding_window), float("-inf"))
    if mask is not None:
        scores = scores + mask.to(scores.dtype)
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_ref.dtype)
    o = torch.matmul(probs, v_ref.to(probs.dtype))
    return o.transpose(0, 1).contiguous()


@torch.compiler.disable
def triton_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    softmax_scale: float,
    sliding_window: int = 0,
) -> torch.Tensor:
    # Keep Triton JIT/autotune state out of torch.compile; CUDA graph capture
    # still records the launched kernels.
    from diffulex.attention import fetch_attn_metadata

    attn_metadata: AttnMetaDataBase = fetch_attn_metadata()
    is_unified_layout = attn_metadata.kv_cache_layout == "unified"
    if k_cache.numel() and v_cache.numel():
        if attn_metadata.need_kv_cache_store:
            store_kv_cache = store_kv_cache_unified_layout if is_unified_layout else store_kv_cache_distinct_layout
            store_kv_cache(k, v, k_cache, v_cache, attn_metadata.slot_mapping, attn_metadata)
    return chunked_prefill_attn_unified(
        q,
        k,
        v,
        k_cache,
        v_cache,
        attn_metadata,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
    )


@torch.compiler.disable
def triton_grouped_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    softmax_scale: float,
    sliding_window: int = 0,
) -> torch.Tensor:
    from diffulex.attention import fetch_attn_metadata

    attn_metadata: AttnMetaDataBase = fetch_attn_metadata()
    if attn_metadata.kv_cache_layout != "unified":
        raise ValueError("attn_impl='triton_grouped' currently requires kv_cache_layout='unified'.")
    if k_cache.numel() and v_cache.numel() and attn_metadata.need_kv_cache_store:
        store_kv_cache_unified_layout(k, v, k_cache, v_cache, attn_metadata.slot_mapping, attn_metadata)
    return chunked_prefill_attn_grouped_unified(
        q,
        k,
        v,
        k_cache,
        v_cache,
        attn_metadata,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
    )


class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        attn_impl: str = "triton_grouped",
        sliding_window: int | None = None,
    ):
        super().__init__()
        if attn_impl not in ATTN_IMPLS:
            raise ValueError(f"attn_impl must be one of {sorted(ATTN_IMPLS)}, got: {attn_impl}")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.attn_impl = attn_impl
        self.sliding_window = int(sliding_window or 0)
        self.k_cache = self.v_cache = torch.tensor([])

        self.q_shape = {
            "nh": self.num_heads,
            "hd": self.head_dim,
        }
        self.kv_shape = {
            "nkvh": self.num_kv_heads,
            "hd": self.head_dim,
        }
        # Import the specified fetch function
        from diffulex.attention import fetch_attn_metadata

        self.fetch_attn_metadata = fetch_attn_metadata

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if q.dim() == 2:
            q = rearrange(q, "s (nh hd) -> s nh hd", **self.q_shape).contiguous()
        elif q.dim() == 3:
            q = q.contiguous()
        else:
            raise ValueError(f"Unsupported q ndim for Attention: {q.dim()}")

        if k.dim() == 2:
            k = rearrange(k, "s (nkvh hd) -> s nkvh hd", **self.kv_shape).contiguous()
        elif k.dim() == 3:
            k = k.contiguous()
        else:
            raise ValueError(f"Unsupported k ndim for Attention: {k.dim()}")

        # Some callers pass V as a strided view from packed QKV. The Triton
        # chunked prefill kernel requires contiguous V rows for this layout.
        if v.dim() == 2:
            v = rearrange(v, "s (nkvh hd) -> s nkvh hd", **self.kv_shape).contiguous()
        elif v.dim() == 3:
            v = v.contiguous()
        else:
            raise ValueError(f"Unsupported v ndim for Attention: {v.dim()}")

        if q.shape[0] == 0:
            return rearrange(q, "s nh hd -> s (nh hd)").contiguous()

        if self.attn_impl == "naive":
            o = reference_torch_attention(
                q,
                k,
                v,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                scale=self.scale,
                mask=mask,
                sliding_window=self.sliding_window,
            )
            return rearrange(o, "s nh hd -> s (nh hd)").contiguous()

        if self.attn_impl not in {"triton", "triton_grouped"}:
            raise ValueError(f"Unsupported attn_impl: {self.attn_impl}")

        k_cache, v_cache = self.k_cache, self.v_cache

        if self.attn_impl == "triton_grouped":
            o = triton_grouped_attention(q, k, v, k_cache, v_cache, self.scale, self.sliding_window)
            return rearrange(o, "s nh hd -> s (nh hd)").contiguous()

        o = triton_attention(q, k, v, k_cache, v_cache, self.scale, self.sliding_window)

        # Final reshape
        return rearrange(o, "s nh hd -> s (nh hd)").contiguous()
