from __future__ import annotations

from typing import Any


_MISSING = object()


def _read_external_attr(config: Any, names: tuple[str, ...], default: Any = _MISSING) -> Any:
    for name in names:
        value = getattr(config, name, _MISSING)
        if value is not _MISSING and value is not None:
            return value

    if default is _MISSING:
        joined = ", ".join(names)
        raise AttributeError(f"Config does not define any of: {joined}.")
    return default


def _read_external_int(config: Any, names: tuple[str, ...], default: int | object = _MISSING) -> int:
    return int(_read_external_attr(config, names, default))


def _read_external_float(config: Any, names: tuple[str, ...], default: float | object = _MISSING) -> float:
    return float(_read_external_attr(config, names, default))


def _read_external_bool(config: Any, names: tuple[str, ...], default: bool | object = _MISSING) -> bool:
    return bool(_read_external_attr(config, names, default))


def _read_external_int_tuple(config: Any, names: tuple[str, ...], default: tuple[int, ...] = ()) -> tuple[int, ...]:
    value = _read_external_attr(config, names, default)
    return tuple(int(layer_idx) for layer_idx in value)


def get_num_experts(config: Any) -> int:
    value = _read_external_attr(
        config,
        ("num_experts", "n_routed_experts", "num_local_experts"),
        0,
    )
    return int(value or 0)


def get_num_experts_per_tok(config: Any) -> int:
    value = _read_external_attr(
        config,
        ("num_experts_per_tok", "moe_top_k", "top_k"),
        2,
    )
    return int(value)


def get_moe_intermediate_size(config: Any) -> int:
    value = _read_external_attr(
        config,
        ("moe_intermediate_size", "expert_intermediate_size", "intermediate_size"),
    )
    return int(value)


def get_norm_topk_prob(config: Any) -> bool:
    return _read_external_bool(config, ("norm_topk_prob",), True)


def get_mlp_only_layers(config: Any) -> tuple[int, ...]:
    return _read_external_int_tuple(config, ("mlp_only_layers",), ())


def get_first_k_dense_replace(config: Any) -> int | None:
    value = _read_external_attr(config, ("first_k_dense_replace",), None)
    if value is None:
        return None
    return int(value)


def get_decoder_sparse_step(config: Any) -> int:
    return _read_external_int(config, ("decoder_sparse_step",), 1)


def is_moe_layer(config: Any, layer_idx: int) -> bool:
    if get_num_experts(config) <= 0:
        return False

    if layer_idx in get_mlp_only_layers(config):
        return False

    first_k_dense_replace = get_first_k_dense_replace(config)
    if first_k_dense_replace is not None:
        return layer_idx >= first_k_dense_replace

    decoder_sparse_step = get_decoder_sparse_step(config)
    if decoder_sparse_step <= 0:
        return False

    return (layer_idx + 1) % decoder_sparse_step == 0


__all__ = [
    "get_mlp_only_layers",
    "get_moe_intermediate_size",
    "get_num_experts",
    "get_num_experts_per_tok",
    "get_norm_topk_prob",
    "get_first_k_dense_replace",
    "get_decoder_sparse_step",
    "is_moe_layer",
]
