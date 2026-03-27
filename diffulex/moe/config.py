from __future__ import annotations

from typing import Any


_MISSING = object()


def _get_attr(config: Any, names: tuple[str, ...], default: Any = _MISSING) -> Any:
    for name in names:
        value = getattr(config, name, _MISSING)
        if value is not _MISSING and value is not None:
            return value

    if default is _MISSING:
        joined = ", ".join(names)
        raise AttributeError(f"Config does not define any of: {joined}.")
    return default


def get_num_experts(config: Any) -> int:
    value = _get_attr(
        config,
        ("num_experts", "n_routed_experts", "num_local_experts"),
        0,
    )
    return int(value or 0)


def get_num_experts_per_tok(config: Any) -> int:
    value = _get_attr(
        config,
        ("num_experts_per_tok", "moe_top_k", "top_k"),
        2,
    )
    return int(value)


def get_moe_intermediate_size(config: Any) -> int:
    value = _get_attr(
        config,
        ("moe_intermediate_size", "intermediate_size"),
    )
    return int(value)


def get_norm_topk_prob(config: Any) -> bool:
    return bool(_get_attr(config, ("norm_topk_prob",), True))


def get_mlp_only_layers(config: Any) -> tuple[int, ...]:
    value = getattr(config, "mlp_only_layers", []) or []
    return tuple(int(layer_idx) for layer_idx in value)


def is_moe_layer(config: Any, layer_idx: int) -> bool:
    if get_num_experts(config) <= 0:
        return False

    if layer_idx in get_mlp_only_layers(config):
        return False

    decoder_sparse_step = int(getattr(config, "decoder_sparse_step", 1) or 1)
    if decoder_sparse_step <= 0:
        return False

    return (layer_idx + 1) % decoder_sparse_step == 0


__all__ = [
    "get_mlp_only_layers",
    "get_moe_intermediate_size",
    "get_num_experts",
    "get_num_experts_per_tok",
    "get_norm_topk_prob",
    "is_moe_layer",
]
