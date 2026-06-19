from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from diffulex.utils.registry import fetch_factory_name

_NOT_PROVIDED = object()

StrategyNormalizer = Callable[[Any], object | None]


@dataclass
class DefaultStrategyConfig:
    name: str


def default_strategy_normalizer(config: Any) -> DefaultStrategyConfig:
    return DefaultStrategyConfig(name=config.decoding_strategy)


class StrategyConfigRegistry:
    _DEFAULT_KEY = "__default__"
    _MAPPING: dict[str, StrategyNormalizer] = {}

    @classmethod
    def register(
        cls,
        strategy_name: str,
        normalizer: StrategyNormalizer | object = _NOT_PROVIDED,
        *,
        aliases: Iterable[str] = (),
        is_default: bool = False,
        exist_ok: bool = False,
    ):
        if not isinstance(strategy_name, str) or not strategy_name:
            raise ValueError("strategy_name must be a non-empty string.")
        if isinstance(aliases, str):
            raise TypeError("aliases must be an iterable of strings, not a single string.")

        def decorator(fn: StrategyNormalizer):
            cls._register(strategy_name, fn, exist_ok=exist_ok)
            for alias in dict.fromkeys(aliases):
                cls._register(alias, fn, exist_ok=exist_ok)
            if is_default:
                cls._register(cls._DEFAULT_KEY, fn, exist_ok=True)
            return fn

        if normalizer is _NOT_PROVIDED:
            return decorator
        return decorator(normalizer)

    @classmethod
    def _register(cls, key: str, normalizer: StrategyNormalizer, *, exist_ok: bool) -> None:
        if key in cls._MAPPING:
            existing = cls._MAPPING[key]
            existing_name = fetch_factory_name(existing)
            new_name = fetch_factory_name(normalizer)
            if existing is normalizer or existing_name == new_name:
                return
            if not exist_ok:
                raise ValueError(
                    f"Strategy config normalizer '{key}: {new_name}' is already "
                    f"registered as '{existing_name}'."
                )
        cls._MAPPING[key] = normalizer

    @classmethod
    def _ensure_strategies_loaded(cls) -> None:
        if not cls._MAPPING:
            from diffulex import strategy as _  # noqa: F401
        if cls._DEFAULT_KEY not in cls._MAPPING:
            cls._MAPPING[cls._DEFAULT_KEY] = default_strategy_normalizer

    @classmethod
    def normalize(cls, config: Any) -> object | None:
        cls._ensure_strategies_loaded()
        normalizer = cls._MAPPING.get(config.decoding_strategy) or cls._MAPPING[cls._DEFAULT_KEY]
        return normalizer(config)
