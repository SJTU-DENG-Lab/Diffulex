import torch
import torch.nn as nn

from functools import lru_cache
from typing import Any

from diffulex.layer.vllm_backend import get_vllm_rope_fn


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


def apply_rotary_emb_(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x.copy_(apply_rotary_emb(x, cos, sin))
    return x


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
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query_shape = query.shape
        key_shape = key.shape
        if query.dim() == 2:
            q_tokens = query.size(0)
            nheads_q = query_shape[-1] // self.head_size
            query = query.view(q_tokens, nheads_q, self.head_size)
            query = apply_rotary_emb_(query, cos, sin).view(query_shape)
        elif query.dim() == 3:
            query = apply_rotary_emb_(query, cos, sin)
        else:
            raise ValueError(f"Unsupported query ndim for RotaryEmbedding: {query.dim()}")

        if key.dim() == 2:
            k_tokens = key.size(0)
            nheads_k = key_shape[-1] // self.head_size
            key = key.view(k_tokens, nheads_k, self.head_size)
            key = apply_rotary_emb_(key, cos, sin).view(key_shape)
        elif key.dim() == 3:
            key = apply_rotary_emb_(key, cos, sin)
        else:
            raise ValueError(f"Unsupported key ndim for RotaryEmbedding: {key.dim()}")
        return query, key


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
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def _apply_rope(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        x_shape = x.shape
        if x.dim() == 2:
            tokens = x.size(0)
            nheads = x_shape[-1] // self.head_size
            x = x.view(tokens, nheads, self.head_size)
            x_rot = x[..., : self.rotary_dim]
            apply_rotary_emb_(x_rot, cos, sin)
            return x.view(x_shape)
        if x.dim() == 3:
            x_rot = x[..., : self.rotary_dim]
            apply_rotary_emb_(x_rot, cos, sin)
            return x
        raise ValueError(f"Unsupported x ndim for PartialRotaryEmbedding: {x.dim()}")

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._apply_rope(positions, query), self._apply_rope(positions, key)


class Gemma4ProportionalRotaryEmbedding(nn.Module):
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
        self.rotary_dim = head_size
        self.rope_angles = rotary_dim // 2
        self.nope_angles = head_size // 2 - self.rope_angles
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / head_size))
        if self.nope_angles > 0:
            inv_freq = torch.cat((inv_freq, torch.zeros(self.nope_angles, dtype=torch.float)))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query_shape = query.shape
        key_shape = key.shape
        if query.dim() == 2:
            q_tokens = query.size(0)
            nheads_q = query_shape[-1] // self.head_size
            query = query.view(q_tokens, nheads_q, self.head_size)
            query = apply_rotary_emb_(query, cos, sin).view(query_shape)
        elif query.dim() == 3:
            query = apply_rotary_emb_(query, cos, sin)
        else:
            raise ValueError(f"Unsupported query ndim for Gemma4ProportionalRotaryEmbedding: {query.dim()}")

        if key.dim() == 2:
            k_tokens = key.size(0)
            nheads_k = key_shape[-1] // self.head_size
            key = key.view(k_tokens, nheads_k, self.head_size)
            key = apply_rotary_emb_(key, cos, sin).view(key_shape)
        elif key.dim() == 3:
            key = apply_rotary_emb_(key, cos, sin)
        else:
            raise ValueError(f"Unsupported key ndim for Gemma4ProportionalRotaryEmbedding: {key.dim()}")
        return query, key


class VllmRotaryEmbeddingAdapter(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        rope_type: str = "default",
    ) -> None:
        super().__init__()
        vllm_get_rope = get_vllm_rope_fn()
        if vllm_get_rope is None:
            raise RuntimeError("vLLM RoPE is unavailable.")
        rope_parameters = {
            "rope_type": rope_type,
            "rope_theta": base,
            "rope_dim": rotary_dim,
        }
        self.rotary_emb = vllm_get_rope(
            head_size=head_size,
            max_position=max_position_embeddings,
            is_neox_style=True,
            rope_parameters=rope_parameters,
            dtype=torch.get_default_dtype(),
        )

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query, key = self.rotary_emb(positions, query, key)
        if key is None:
            raise RuntimeError("Diffulex RoPE expects key output, got None.")
        return query, key


def _make_vllm_rope(
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: float,
    rope_type: str = "default",
) -> nn.Module | None:
    if get_vllm_rope_fn() is None:
        return None
    try:
        return VllmRotaryEmbeddingAdapter(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            rope_type=rope_type,
        )
    except Exception:
        return None


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

    # HF configs may standardize default rope into a dict form. We accept that and
    # treat it identically to rope_scaling=None because Diffulex currently only
    # implements the default rotary embedding path here.
    if rope_type in ("default", None):
        return

    raise NotImplementedError(
        "Diffulex RotaryEmbedding currently supports only the default rope variant, "
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
    rotary_emb = _make_vllm_rope(head_size, rotary_dim, max_position, base)
    if rotary_emb is not None:
        return rotary_emb
    rotary_cls = RotaryEmbedding if rotary_dim == head_size else PartialRotaryEmbedding
    rotary_emb = rotary_cls(head_size, rotary_dim, max_position, base)
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


@lru_cache(8)
def get_gemma4_proportional_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = _make_vllm_rope(head_size, rotary_dim, max_position, base, rope_type="proportional")
    if rotary_emb is not None:
        return rotary_emb
    return Gemma4ProportionalRotaryEmbedding(head_size, rotary_dim, max_position, base)
