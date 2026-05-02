import os
from functools import lru_cache
from typing import Any
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


_ROPE_KERNEL_SUPPORTED_HEAD_SIZES = {64, 128, 256, 512}


def _use_reference_rope() -> bool:
    return os.getenv("DIFFULEX_REFERENCE_ROPE", "0") == "1"


def _get_sgl_rope_op():
    namespace = getattr(torch.ops, "sgl_kernel", None)
    op = getattr(namespace, "apply_rope_with_cos_sin_cache_inplace", None) if namespace is not None else None
    if op is not None:
        return op
    try:
        import sgl_kernel

        return getattr(sgl_kernel, "apply_rope_with_cos_sin_cache_inplace", None)
    except Exception:
        return None


def _rope_type_from_scaling(rope_scaling: dict[str, Any] | None) -> str:
    if rope_scaling is None:
        return "default"
    return str(rope_scaling.get("rope_type", rope_scaling.get("type", "default")))


def _build_rope_config(
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: float,
    rope_scaling: dict[str, Any] | None,
):
    config = SimpleNamespace(
        rope_theta=base,
        head_dim=head_size,
        hidden_size=head_size,
        num_attention_heads=1,
        max_position_embeddings=max_position_embeddings,
        partial_rotary_factor=rotary_dim / head_size,
        rope_scaling=rope_scaling,
    )
    return config


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        assert rotary_dim == head_size
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = max_position_embeddings
        self.base = base
        self.rope_scaling = None
        self.rope_type = "default"
        self.rope_init_fn = ROPE_INIT_FUNCTIONS["default"]
        self.rope_config = _build_rope_config(head_size, rotary_dim, max_position_embeddings, base, None)
        self.attention_factor = 1.0
        cache = self._build_cos_sin_cache(max_position_embeddings)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, *, seq_len: int | None = None, device: torch.device | None = None) -> torch.Tensor:
        inv_freq, attention_factor = self.rope_init_fn(self.rope_config, device=device, seq_len=seq_len)
        self.attention_factor = attention_factor
        return inv_freq

    def _build_cos_sin_cache(self, cache_len: int, device: torch.device | None = None) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(seq_len=cache_len if self.rope_type == "dynamic" else None, device=device)
        t = torch.arange(cache_len, dtype=torch.float, device=device)
        freqs = torch.einsum("i,j -> ij", t, inv_freq.to(dtype=torch.float32))
        cos = freqs.cos() * self.attention_factor
        sin = freqs.sin() * self.attention_factor
        return torch.cat((cos, sin), dim=-1)

    def _ensure_cos_sin_cache_length(self, positions: torch.Tensor, device: torch.device) -> None:
        needed_max_pos = int(torch.max(positions).item()) + 1 if positions.numel() else 0
        if needed_max_pos <= 0:
            return

        current_len = int(self.cos_sin_cache.shape[0])
        if self.rope_type == "dynamic":
            if needed_max_pos > current_len:
                self.cos_sin_cache = self._build_cos_sin_cache(needed_max_pos, device=device)
                return
            if current_len > self.original_max_position_embeddings and needed_max_pos <= self.original_max_position_embeddings:
                self.cos_sin_cache = self._build_cos_sin_cache(self.original_max_position_embeddings, device=device)
                return

        if needed_max_pos > current_len:
            self.cos_sin_cache = self._build_cos_sin_cache(needed_max_pos, device=device)

    def _can_use_kernel(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor) -> bool:
        return (
            not _use_reference_rope()
            and query.is_cuda
            and key.is_cuda
            and positions.device == query.device == key.device
            and self.head_size in _ROPE_KERNEL_SUPPORTED_HEAD_SIZES
            and query.dtype in (torch.float16, torch.bfloat16)
            and key.dtype == query.dtype
        )

    def _reshape_for_kernel(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        original_shape = x.shape
        if x.dim() == 2:
            x = x.reshape(x.size(0), -1, self.head_size)
        elif x.dim() != 3:
            raise ValueError(f"Unsupported x ndim for RotaryEmbedding: {x.dim()}")
        return x.contiguous(), original_shape

    def _restore_from_kernel(self, x: torch.Tensor, original_shape: tuple[int, ...]) -> torch.Tensor:
        if len(original_shape) == 2:
            return x.reshape(original_shape)
        return x

    def _forward_kernel(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if _use_reference_rope():
            raise RuntimeError("RoPE kernel path is disabled by reference mode.")
        op = _get_sgl_rope_op()
        if op is None or not self._can_use_kernel(positions, query, key):
            raise RuntimeError("RoPE kernel path is unavailable.")
        self._ensure_cos_sin_cache_length(positions, query.device)
        query_3d, query_shape = self._reshape_for_kernel(query)
        key_3d, key_shape = self._reshape_for_kernel(key)
        op(
            positions=positions,
            query=query_3d,
            key=key_3d,
            head_size=self.head_size,
            cos_sin_cache=self.cos_sin_cache.to(device=query.device),
            is_neox=True,
        )
        return self._restore_from_kernel(query_3d, query_shape), self._restore_from_kernel(key_3d, key_shape)

    @torch.compile
    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_cos_sin_cache_length(positions, query.device)
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query_shape = query.shape
        key_shape = key.shape
        if query.dim() == 2:
            q_tokens = query.size(0)
            nheads_q = query_shape[-1] // self.head_size
            query = query.view(q_tokens, nheads_q, self.head_size)
            query = apply_rotary_emb(query, cos, sin).view(query_shape)
        elif query.dim() == 3:
            query = apply_rotary_emb(query, cos, sin)
        else:
            raise ValueError(f"Unsupported query ndim for RotaryEmbedding: {query.dim()}")

        if key.dim() == 2:
            k_tokens = key.size(0)
            nheads_k = key_shape[-1] // self.head_size
            key = key.view(k_tokens, nheads_k, self.head_size)
            key = apply_rotary_emb(key, cos, sin).view(key_shape)
        elif key.dim() == 3:
            key = apply_rotary_emb(key, cos, sin)
        else:
            raise ValueError(f"Unsupported key ndim for RotaryEmbedding: {key.dim()}")
        return query, key

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return self._forward_kernel(positions, query, key)
        except RuntimeError:
            return self.forward_native(positions, query, key)


class PartialRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        if rotary_dim <= 0 or rotary_dim > head_size or rotary_dim % 2 != 0:
            raise ValueError(f"Invalid rotary_dim={rotary_dim} for head_size={head_size}.")
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = max_position_embeddings
        self.base = base
        self.rope_scaling = None
        self.rope_type = "default"
        self.rope_init_fn = ROPE_INIT_FUNCTIONS["default"]
        self.rope_config = _build_rope_config(head_size, rotary_dim, max_position_embeddings, base, None)
        self.attention_factor = 1.0
        cache = self._build_cos_sin_cache(max_position_embeddings)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, *, seq_len: int | None = None, device: torch.device | None = None) -> torch.Tensor:
        inv_freq, attention_factor = self.rope_init_fn(self.rope_config, device=device, seq_len=seq_len)
        self.attention_factor = attention_factor
        return inv_freq

    def _build_cos_sin_cache(self, cache_len: int, device: torch.device | None = None) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(seq_len=cache_len if self.rope_type == "dynamic" else None, device=device)
        t = torch.arange(cache_len, dtype=torch.float, device=device)
        freqs = torch.einsum("i,j -> ij", t, inv_freq.to(dtype=torch.float32))
        cos = freqs.cos() * self.attention_factor
        sin = freqs.sin() * self.attention_factor
        return torch.cat((cos, sin), dim=-1)

    def _ensure_cos_sin_cache_length(self, positions: torch.Tensor, device: torch.device) -> None:
        needed_max_pos = int(torch.max(positions).item()) + 1 if positions.numel() else 0
        if needed_max_pos <= 0:
            return

        current_len = int(self.cos_sin_cache.shape[0])
        if self.rope_type == "dynamic":
            if needed_max_pos > current_len:
                self.cos_sin_cache = self._build_cos_sin_cache(needed_max_pos, device=device)
                return
            if current_len > self.original_max_position_embeddings and needed_max_pos <= self.original_max_position_embeddings:
                self.cos_sin_cache = self._build_cos_sin_cache(self.original_max_position_embeddings, device=device)
                return

        if needed_max_pos > current_len:
            self.cos_sin_cache = self._build_cos_sin_cache(needed_max_pos, device=device)

    def _can_use_kernel(self, positions: torch.Tensor, x: torch.Tensor) -> bool:
        return (
            not _use_reference_rope()
            and x.is_cuda
            and positions.device == x.device
            and self.head_size in _ROPE_KERNEL_SUPPORTED_HEAD_SIZES
            and x.dtype in (torch.float16, torch.bfloat16)
        )

    @torch.compile
    def _apply_rope_native(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        self._ensure_cos_sin_cache_length(positions, x.device)
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        x_shape = x.shape
        if x.dim() == 2:
            tokens = x.size(0)
            nheads = x_shape[-1] // self.head_size
            x = x.view(tokens, nheads, self.head_size)
            x_rot = x[..., : self.rotary_dim]
            x_pass = x[..., self.rotary_dim :]
            x_rot = apply_rotary_emb(x_rot, cos, sin)
            return torch.cat((x_rot, x_pass), dim=-1).view(x_shape)
        if x.dim() == 3:
            x_rot = x[..., : self.rotary_dim]
            x_pass = x[..., self.rotary_dim :]
            x_rot = apply_rotary_emb(x_rot, cos, sin)
            return torch.cat((x_rot, x_pass), dim=-1)
        raise ValueError(f"Unsupported x ndim for PartialRotaryEmbedding: {x.dim()}")

    def _apply_rope_kernel(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if _use_reference_rope():
            raise RuntimeError("Partial RoPE kernel path is disabled by reference mode.")
        op = _get_sgl_rope_op()
        if op is None or not self._can_use_kernel(positions, x):
            raise RuntimeError("Partial RoPE kernel path is unavailable.")
        self._ensure_cos_sin_cache_length(positions, x.device)
        original_shape = x.shape
        if x.dim() == 2:
            x = x.reshape(x.size(0), -1, self.head_size)
        elif x.dim() != 3:
            raise ValueError(f"Unsupported x ndim for PartialRotaryEmbedding: {x.dim()}")
        x_out = x.contiguous()
        empty = x_out.new_empty((x_out.size(0), 0, self.head_size))
        op(
            positions=positions,
            query=x_out,
            key=empty,
            head_size=self.head_size,
            cos_sin_cache=self.cos_sin_cache.to(device=x_out.device),
            is_neox=True,
        )
        if len(original_shape) == 2:
            return x_out.reshape(original_shape)
        return x_out

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return self._apply_rope_kernel(positions, query), self._apply_rope_kernel(positions, key)
        except RuntimeError:
            return self._apply_rope_native(positions, query), self._apply_rope_native(positions, key)


def _normalize_rope_scaling(
    rope_scaling: dict[str, Any] | tuple[tuple[str, Any], ...] | None,
) -> tuple[tuple[str, Any], ...] | None:
    if rope_scaling is None:
        return None
    if isinstance(rope_scaling, tuple):
        return rope_scaling
    if not isinstance(rope_scaling, dict):
        raise TypeError(f"rope_scaling must be a dict, tuple, or None, got: {type(rope_scaling)!r}")
    return tuple(sorted(rope_scaling.items()))


def _validate_rope_scaling(
    rope_scaling: tuple[tuple[str, Any], ...] | None,
) -> None:
    if rope_scaling is None:
        return

    rope_scaling_dict = dict(rope_scaling)
    rope_type = rope_scaling_dict.get("rope_type", rope_scaling_dict.get("type", "default"))

    if rope_type in ("default", None, "linear", "dynamic"):
        return

    raise NotImplementedError(
        "Diffulex RotaryEmbedding currently supports only default, linear, and dynamic rope variants, "
        f"got rope_scaling={rope_scaling_dict}."
    )


@lru_cache(1)
def _get_rope_cached(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: tuple[tuple[str, Any], ...] | None = None,
):
    _validate_rope_scaling(rope_scaling)
    rope_scaling_dict = dict(rope_scaling) if rope_scaling is not None else None
    rotary_cls = RotaryEmbedding if rotary_dim == head_size else PartialRotaryEmbedding
    rotary_emb = rotary_cls(head_size, rotary_dim, max_position, base)
    rope_type = _rope_type_from_scaling(rope_scaling_dict)
    rotary_emb.rope_scaling = rope_scaling_dict
    rotary_emb.rope_type = rope_type
    rotary_emb.rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
    rotary_emb.rope_config = _build_rope_config(head_size, rotary_dim, max_position, base, rope_scaling_dict)
    rotary_emb.cos_sin_cache = rotary_emb._build_cos_sin_cache(max_position)
    return rotary_emb


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | tuple[tuple[str, Any], ...] | None = None,
):
    return _get_rope_cached(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position=max_position,
        base=base,
        rope_scaling=_normalize_rope_scaling(rope_scaling),
    )
