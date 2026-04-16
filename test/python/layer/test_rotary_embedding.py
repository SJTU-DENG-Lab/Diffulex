from diffulex.layer.rotary_embedding import get_rope


def test_get_rope_accepts_default_rope_scaling_dict() -> None:
    rope = get_rope(
        head_size=8,
        rotary_dim=8,
        max_position=128,
        base=10000.0,
        rope_scaling={"rope_type": "default"},
    )

    assert rope.head_size == 8


def test_get_rope_rejects_unsupported_rope_scaling_variant() -> None:
    try:
        get_rope(
            head_size=8,
            rotary_dim=8,
            max_position=128,
            base=10000.0,
            rope_scaling={"rope_type": "linear", "factor": 2.0},
        )
    except NotImplementedError as exc:
        assert "supports only the default rope variant" in str(exc)
    else:
        raise AssertionError("Expected NotImplementedError for unsupported rope scaling.")
