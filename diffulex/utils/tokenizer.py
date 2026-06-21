import json
import os
import re
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerFast


def _special_token_name(token: str, index: int, used: set[str]) -> str:
    name = token.strip("<>|").replace("-", "_")
    name = re.sub(r"\W+", "_", name).strip("_").lower()
    if not name:
        name = f"extra_special_{index}"
    if not name.endswith("_token"):
        name = f"{name}_token"
    base = name
    suffix = 1
    while name in used:
        suffix += 1
        name = f"{base}_{suffix}"
    used.add(name)
    return name


def _coerce_extra_special_tokens(tokenizer_path: str) -> dict[str, str] | None:
    config_path = os.path.join(tokenizer_path, "tokenizer_config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, encoding="utf-8") as f:
            tokenizer_config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    extra_special_tokens = tokenizer_config.get("extra_special_tokens")
    if not isinstance(extra_special_tokens, list):
        return None

    used: set[str] = set()
    return {
        _special_token_name(str(token), index, used): str(token)
        for index, token in enumerate(extra_special_tokens)
    }


def _tokenizer_class_name(tokenizer_path: str) -> str | None:
    config_path = os.path.join(tokenizer_path, "tokenizer_config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, encoding="utf-8") as f:
            tokenizer_config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    tokenizer_class = tokenizer_config.get("tokenizer_class")
    return tokenizer_class if isinstance(tokenizer_class, str) else None


def auto_tokenizer_from_pretrained(tokenizer_path: str, **kwargs: Any):
    try:
        return AutoTokenizer.from_pretrained(tokenizer_path, **kwargs)
    except AttributeError as exc:
        if "'list' object has no attribute 'keys'" not in str(exc) or "extra_special_tokens" in kwargs:
            raise
        coerced_extra_special_tokens = _coerce_extra_special_tokens(tokenizer_path)
        if coerced_extra_special_tokens is None:
            raise
        return AutoTokenizer.from_pretrained(
            tokenizer_path,
            **kwargs,
            extra_special_tokens=coerced_extra_special_tokens,
        )
    except Exception:
        if _tokenizer_class_name(tokenizer_path) != "PreTrainedTokenizerFast":
            raise
        return PreTrainedTokenizerFast.from_pretrained(tokenizer_path, **kwargs)
