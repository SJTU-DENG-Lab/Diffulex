import pytest
import torch

from diffulex.layer.rotary_embedding import PartialRotaryEmbedding, RotaryEmbedding, get_rope


def test_get_rope_accepts_default_rope_scaling_dict() -> None:
    rope = get_rope(
        head_size=8,
        rotary_dim=8,
        max_position=128,
        base=10000.0,
        rope_scaling={"rope_type": "default"},
    )

    assert rope.head_size == 8
    assert rope.rope_type == "default"


def test_get_rope_accepts_linear_rope_scaling_dict() -> None:
    rope = get_rope(
        head_size=8,
        rotary_dim=8,
        max_position=128,
        base=10000.0,
        rope_scaling={"rope_type": "linear", "factor": 2.0},
    )

    assert rope.rope_type == "linear"


def test_get_rope_accepts_dynamic_rope_scaling_dict() -> None:
    rope = get_rope(
        head_size=8,
        rotary_dim=8,
        max_position=128,
        base=10000.0,
        rope_scaling={"rope_type": "dynamic", "factor": 2.0},
    )

    assert rope.rope_type == "dynamic"


def test_get_rope_rejects_unsupported_rope_scaling_variant() -> None:
    try:
        get_rope(
            head_size=8,
            rotary_dim=8,
            max_position=128,
            base=10000.0,
            rope_scaling={"rope_type": "yarn", "factor": 2.0},
        )
    except NotImplementedError as exc:
        assert "supports only default, linear, and dynamic rope variants" in str(exc)
    else:
        raise AssertionError("Expected NotImplementedError for unsupported rope scaling.")


def test_dynamic_rope_scaling_extends_cache_for_longer_positions() -> None:
    rope = get_rope(
        head_size=8,
        rotary_dim=8,
        max_position=8,
        base=10000.0,
        rope_scaling={"rope_type": "dynamic", "factor": 2.0},
    )
    positions = torch.tensor([0, 1, 2, 15], dtype=torch.long)
    query = torch.randn(4, 8)
    key = torch.randn(4, 8)

    rope(positions, query, key)

    assert rope.cos_sin_cache.shape[0] >= 16


def test_rotary_embedding_falls_back_to_reference_when_forced(monkeypatch) -> None:
    monkeypatch.setenv("DIFFULEX_REFERENCE_ROPE", "1")
    monkeypatch.setattr("diffulex.layer.rotary_embedding._get_sgl_rope_op", lambda: (_ for _ in ()).throw(AssertionError("kernel should not be called")))

    rope = RotaryEmbedding(head_size=8, rotary_dim=8, max_position_embeddings=16, base=10000.0)
    positions = torch.arange(4, dtype=torch.long)
    query = torch.randn(4, 8)
    key = torch.randn(4, 8)

    out_q, out_k = rope(positions, query, key)

    assert out_q.shape == query.shape
    assert out_k.shape == key.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rotary_embedding_uses_kernel_dispatch_when_available(monkeypatch) -> None:
    calls = {"kernel": 0}

    def fake_kernel(*, positions, query, key, head_size, cos_sin_cache, is_neox):
        calls["kernel"] += 1
        query.add_(1)
        key.add_(2)

    monkeypatch.delenv("DIFFULEX_REFERENCE_ROPE", raising=False)
    monkeypatch.setattr("diffulex.layer.rotary_embedding._get_sgl_rope_op", lambda: fake_kernel)

    rope = RotaryEmbedding(head_size=64, rotary_dim=64, max_position_embeddings=16, base=10000.0)
    positions = torch.arange(4, device="cuda", dtype=torch.long)
    query = torch.zeros(4, 2 * 64, device="cuda", dtype=torch.bfloat16)
    key = torch.zeros(4, 2 * 64, device="cuda", dtype=torch.bfloat16)

    out_q, out_k = rope(positions, query, key)

    assert calls["kernel"] == 1
    assert torch.equal(out_q, torch.ones_like(out_q))
    assert torch.equal(out_k, torch.full_like(out_k, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_partial_rotary_embedding_uses_kernel_dispatch_when_available(monkeypatch) -> None:
    calls = {"kernel": 0}

    def fake_kernel(*, positions, query, key, head_size, cos_sin_cache, is_neox):
        calls["kernel"] += 1
        query[..., : cos_sin_cache.shape[-1]].add_(3)

    monkeypatch.delenv("DIFFULEX_REFERENCE_ROPE", raising=False)
    monkeypatch.setattr("diffulex.layer.rotary_embedding._get_sgl_rope_op", lambda: fake_kernel)

    rope = PartialRotaryEmbedding(head_size=64, rotary_dim=32, max_position_embeddings=16, base=10000.0)
    positions = torch.arange(4, device="cuda", dtype=torch.long)
    query = torch.zeros(4, 2 * 64, device="cuda", dtype=torch.bfloat16)
    key = torch.zeros(4, 2 * 64, device="cuda", dtype=torch.bfloat16)

    out_q, out_k = rope(positions, query, key)
    out_q = out_q.view(4, 2, 64)
    out_k = out_k.view(4, 2, 64)

    assert calls["kernel"] == 2
    assert torch.all(out_q[..., :32] == 3)
    assert torch.all(out_q[..., 32:] == 0)
    assert torch.all(out_k[..., :32] == 3)
    assert torch.all(out_k[..., 32:] == 0)
