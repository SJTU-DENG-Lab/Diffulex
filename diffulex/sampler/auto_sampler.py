from __future__ import annotations

from typing import Any, Callable

from diffulex.config import Config


_NOT_PROVIDED = object()
RegistryEntry = tuple[Callable[[Any], Any] | type | None, bool]


class AutoSampler:
    """Factory and registry for diffusion language model samplers."""

    SAMPLER_MAPPING: dict[str, RegistryEntry] = {}

    @classmethod
    def register(
        cls,
        sampler_name: str,
        sampler_class: Callable[[Any], Any] | type | None = _NOT_PROVIDED,
        *,
        use_full_config: bool = False,
        exist_ok: bool = False,
    ):
        """Register a sampler factory or class under ``sampler_name``.

        When ``sampler_class`` is omitted this method returns a decorator.

        Args:
            sampler_name: Key used to retrieve the sampler.
            sampler_class: Callable or class that builds the sampler instance.
            use_full_config: Pass the entire :class:`Config` to the factory
                instead of ``config.hf_config``.
            exist_ok: Allow overriding an existing registration.
        """

        if not isinstance(sampler_name, str) or not sampler_name:
            raise ValueError("sampler_name must be a non-empty string.")

        if sampler_class is _NOT_PROVIDED:
            def decorator(sampler_cls):
                cls._register(sampler_name, sampler_cls, use_full_config=use_full_config, exist_ok=exist_ok)
                return sampler_cls

            return decorator

        cls._register(sampler_name, sampler_class, use_full_config=use_full_config, exist_ok=exist_ok)
        return sampler_class

    @classmethod
    def _register(
        cls,
        sampler_name: str,
        sampler_class: Callable[[Any], Any] | type | None,
        *,
        use_full_config: bool,
        exist_ok: bool,
    ) -> None:
        if not exist_ok and sampler_name in cls.SAMPLER_MAPPING:
            raise ValueError(f"Sampler '{sampler_name}' is already registered.")
        cls.SAMPLER_MAPPING[sampler_name] = (sampler_class, use_full_config)

    @classmethod
    def unregister(cls, sampler_name: str) -> None:
        cls.SAMPLER_MAPPING.pop(sampler_name, None)

    @classmethod
    def available_samplers(cls) -> tuple[str, ...]:
        return tuple(sorted(cls.SAMPLER_MAPPING))

    @classmethod
    def from_config(cls, config: Config):
        if not hasattr(config, "model_name"):
            raise AttributeError("Config must define 'model_name' to build a sampler.")

        try:
            factory, use_full_config = cls.SAMPLER_MAPPING[config.model_name]
        except KeyError as err:
            available = ", ".join(cls.available_samplers()) or "<none>"
            raise ValueError(
                f"Sampler '{config.model_name}' is not registered. Available samplers: {available}."
            ) from err

        if factory is None:
            raise ValueError(f"Sampler '{config.model_name}' is reserved but not implemented yet.")

        # Samplers don't require initialization arguments, they are nn.Module subclasses
        sampler = factory()
        return sampler