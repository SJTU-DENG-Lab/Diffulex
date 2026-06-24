from __future__ import annotations

from types import SimpleNamespace

from diffulex.engine.engine import maybe_override_mask_token_id


def test_mask_token_override_uses_tokenizer_when_config_is_default():
    config = SimpleNamespace(mask_token_id=151666)
    tokenizer = SimpleNamespace(mask_token_id=126336)

    maybe_override_mask_token_id(config, tokenizer)

    assert config.mask_token_id == 126336


def test_mask_token_override_preserves_explicit_value():
    config = SimpleNamespace(mask_token_id=151666)
    tokenizer = SimpleNamespace(mask_token_id=126336)

    maybe_override_mask_token_id(config, tokenizer, mask_token_id_explicit=True)

    assert config.mask_token_id == 151666


def test_mask_token_override_preserves_non_default_value():
    config = SimpleNamespace(mask_token_id=151665)
    tokenizer = SimpleNamespace(mask_token_id=126336)

    maybe_override_mask_token_id(config, tokenizer)

    assert config.mask_token_id == 151665
