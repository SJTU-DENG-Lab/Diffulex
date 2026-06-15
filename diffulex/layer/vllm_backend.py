from __future__ import annotations

from functools import lru_cache


_VLLM_LAYERS_ENABLED = True


def set_vllm_layers_enabled(enabled: bool) -> None:
    global _VLLM_LAYERS_ENABLED
    enabled = bool(enabled)
    if _VLLM_LAYERS_ENABLED == enabled:
        return
    _VLLM_LAYERS_ENABLED = enabled
    clear_vllm_layer_caches()


def is_vllm_layers_enabled() -> bool:
    return _VLLM_LAYERS_ENABLED


def clear_vllm_layer_caches() -> None:
    get_vllm_silu_and_mul_cls.cache_clear()
    get_vllm_gelu_and_mul_cls.cache_clear()
    get_vllm_rmsnorm_cls.cache_clear()
    get_vllm_rope_fn.cache_clear()


@lru_cache(1)
def get_vllm_silu_and_mul_cls():
    if not is_vllm_layers_enabled():
        return None
    try:
        from vllm.model_executor.layers.activation import SiluAndMul

        return SiluAndMul
    except Exception:
        return None


@lru_cache(1)
def get_vllm_gelu_and_mul_cls():
    if not is_vllm_layers_enabled():
        return None
    try:
        from vllm.model_executor.layers.activation import GeluAndMul

        return GeluAndMul
    except Exception:
        return None


@lru_cache(1)
def get_vllm_rmsnorm_cls():
    if not is_vllm_layers_enabled():
        return None
    try:
        from vllm.model_executor.layers.layernorm import RMSNorm

        return RMSNorm
    except Exception:
        return None


@lru_cache(1)
def get_vllm_rope_fn():
    if not is_vllm_layers_enabled():
        return None
    try:
        from vllm.model_executor.layers.rotary_embedding import get_rope

        return get_rope
    except Exception:
        return None
