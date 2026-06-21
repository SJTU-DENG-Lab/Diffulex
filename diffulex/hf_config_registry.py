from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Callable

from transformers import AutoConfig

from diffulex.utils.registry import fetch_factory_name

ConfigDict = dict[str, Any]
ConfigLoader = Callable[[str, ConfigDict], Any]
ConfigPredicate = Callable[[ConfigDict], bool]
ConfigPostprocessor = Callable[[Any, Any, ConfigDict], None]

_NOT_PROVIDED = object()


def read_checkpoint_config(model: str) -> ConfigDict:
    config_path = Path(model) / "config.json"
    try:
        with config_path.open("r", encoding="utf-8") as f:
            value = json.load(f)
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected {config_path} to contain a JSON object.")
    return value


class HFConfigRegistry:
    _model_type_loaders: dict[str, ConfigLoader] = {}
    _predicate_loaders: list[tuple[ConfigPredicate, ConfigLoader]] = []
    _postprocessors: dict[str, list[ConfigPostprocessor]] = {}
    _hooks_loaded = False

    @classmethod
    def register_model_type(
        cls,
        model_type: str,
        loader: ConfigLoader | object = _NOT_PROVIDED,
        *,
        exist_ok: bool = False,
    ):
        if not isinstance(model_type, str) or not model_type:
            raise ValueError("model_type must be a non-empty string.")

        def decorator(fn: ConfigLoader):
            cls._register_model_type(model_type, fn, exist_ok=exist_ok)
            return fn

        if loader is _NOT_PROVIDED:
            return decorator
        return decorator(loader)

    @classmethod
    def _register_model_type(cls, model_type: str, loader: ConfigLoader, *, exist_ok: bool) -> None:
        if model_type in cls._model_type_loaders:
            existing = cls._model_type_loaders[model_type]
            if existing is loader or fetch_factory_name(existing) == fetch_factory_name(loader):
                return
            if not exist_ok:
                raise ValueError(
                    f"HF config loader for model_type='{model_type}' is already registered "
                    f"as {fetch_factory_name(existing)}."
                )
        cls._model_type_loaders[model_type] = loader

    @classmethod
    def register_predicate(
        cls,
        predicate: ConfigPredicate,
        loader: ConfigLoader | object = _NOT_PROVIDED,
        *,
        exist_ok: bool = False,
    ):
        if not callable(predicate):
            raise TypeError("predicate must be callable.")

        def decorator(fn: ConfigLoader):
            cls._register_predicate(predicate, fn, exist_ok=exist_ok)
            return fn

        if loader is _NOT_PROVIDED:
            return decorator
        return decorator(loader)

    @classmethod
    def _register_predicate(
        cls,
        predicate: ConfigPredicate,
        loader: ConfigLoader,
        *,
        exist_ok: bool,
    ) -> None:
        for existing_predicate, existing_loader in cls._predicate_loaders:
            if (
                fetch_factory_name(existing_predicate) == fetch_factory_name(predicate)
                and fetch_factory_name(existing_loader) == fetch_factory_name(loader)
            ):
                return
            if fetch_factory_name(existing_predicate) == fetch_factory_name(predicate) and not exist_ok:
                raise ValueError(
                    "HF config predicate loader is already registered for "
                    f"{fetch_factory_name(predicate)}."
                )
        cls._predicate_loaders.append((predicate, loader))

    @classmethod
    def register_postprocessor(
        cls,
        model_type: str,
        postprocessor: ConfigPostprocessor | object = _NOT_PROVIDED,
    ):
        if not isinstance(model_type, str) or not model_type:
            raise ValueError("model_type must be a non-empty string.")

        def decorator(fn: ConfigPostprocessor):
            processors = cls._postprocessors.setdefault(model_type, [])
            for existing in processors:
                if existing is fn or fetch_factory_name(existing) == fetch_factory_name(fn):
                    return fn
            processors.append(fn)
            return fn

        if postprocessor is _NOT_PROVIDED:
            return decorator
        return decorator(postprocessor)

    @classmethod
    def _ensure_hooks_loaded(cls) -> None:
        if cls._hooks_loaded:
            return

        root = Path(__file__).parent / "model" / "config"
        for path in sorted(root.rglob("configuration_*.py")):
            rel = path.relative_to(root).with_suffix("")
            module = ".".join(("diffulex.model.config", *rel.parts))
            importlib.import_module(module)

        cls._hooks_loaded = True

    @classmethod
    def load(cls, model: str):
        cls._ensure_hooks_loaded()
        config_dict = read_checkpoint_config(model)
        model_type = config_dict.get("model_type")

        if isinstance(model_type, str) and model_type in cls._model_type_loaders:
            return cls._model_type_loaders[model_type](model, config_dict)

        for predicate, loader in cls._predicate_loaders:
            if predicate(config_dict):
                return loader(model, config_dict)

        return AutoConfig.from_pretrained(model, trust_remote_code=True)

    @classmethod
    def postprocess(cls, hf_config: Any, engine_config: Any, config_dict: ConfigDict | None = None) -> None:
        cls._ensure_hooks_loaded()
        if config_dict is None:
            config_dict = read_checkpoint_config(engine_config.model)
        cls._normalize_common_fields(hf_config)
        model_type = getattr(hf_config, "model_type", None)
        if not isinstance(model_type, str):
            model_type = config_dict.get("model_type")
        for postprocessor in cls._postprocessors.get(str(model_type), []):
            postprocessor(hf_config, engine_config, config_dict)

    @staticmethod
    def _normalize_common_fields(hf_config: Any) -> None:
        head_dim = getattr(hf_config, "head_dim", None)
        if head_dim is None and hasattr(hf_config, "hidden_size") and hasattr(hf_config, "num_attention_heads"):
            hidden_size = int(hf_config.hidden_size)
            num_attention_heads = int(hf_config.num_attention_heads)
            if hidden_size % num_attention_heads != 0:
                raise ValueError(
                    "Cannot infer head_dim because hidden_size is not divisible by num_attention_heads: "
                    f"hidden_size={hidden_size}, num_attention_heads={num_attention_heads}."
                )
            hf_config.head_dim = hidden_size // num_attention_heads
