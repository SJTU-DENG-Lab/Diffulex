from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from diffulex.layer.activation import GeluAndMul, SiluAndMul
from diffulex.layer.layernorm import RMSNorm
from diffulex.layer.rotary_embedding import get_gemma4_proportional_rope, get_rope
from diffulex.layer.vllm_backend import set_vllm_layers_enabled


@pytest.fixture(autouse=True)
def _force_diffulex_fallback_layers():
    set_vllm_layers_enabled(False)
    yield
    set_vllm_layers_enabled(True)


def _reference_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


def _default_cos_sin(
    positions: torch.Tensor,
    *,
    rotary_dim: int,
    base: float,
    denom: int | None = None,
    zero_pad_angles: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    denom = int(denom or rotary_dim)
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / denom))
    if zero_pad_angles > 0:
        inv_freq = torch.cat((inv_freq, torch.zeros(zero_pad_angles, dtype=torch.float32)))
    freqs = torch.einsum("i,j->ij", positions.float(), inv_freq)
    return freqs.cos(), freqs.sin()


def test_silu_and_mul_matches_reference() -> None:
    torch.manual_seed(0)
    x = torch.randn(17, 64, dtype=torch.float32)

    actual = SiluAndMul()(x)
    gate, up = x.chunk(2, dim=-1)
    expected = F.silu(gate) * up

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_gelu_and_mul_matches_reference() -> None:
    torch.manual_seed(1)
    x = torch.randn(17, 64, dtype=torch.float32)

    actual = GeluAndMul()(x)
    gate, up = x.chunk(2, dim=-1)
    expected = F.gelu(gate, approximate="tanh") * up

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("has_weight", [True, False])
@pytest.mark.parametrize("with_residual", [True, False])
def test_rmsnorm_matches_reference(has_weight: bool, with_residual: bool) -> None:
    torch.manual_seed(2)
    x = torch.randn(11, 32, dtype=torch.float32)
    residual = torch.randn_like(x) if with_residual else None
    norm = RMSNorm(32, eps=1e-6, has_weight=has_weight)
    if has_weight:
        with torch.no_grad():
            norm.weight.copy_(torch.randn_like(norm.weight))

    actual = norm(x.clone(), residual.clone() if residual is not None else None)
    ref_input = x.float() if residual is None else x.float() + residual.float()
    ref = ref_input * torch.rsqrt(ref_input.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    ref = ref.to(x.dtype)
    if has_weight:
        ref = ref * norm.weight

    if residual is None:
        torch.testing.assert_close(actual, ref, rtol=1e-6, atol=1e-6)
    else:
        actual_hidden, actual_residual = actual
        torch.testing.assert_close(actual_hidden, ref, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(actual_residual, ref_input.to(x.dtype), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("shape", [(9, 32), (9, 4, 8)])
def test_full_rope_matches_reference_and_is_inplace(shape: tuple[int, ...]) -> None:
    torch.manual_seed(3)
    positions = torch.arange(shape[0], dtype=torch.long)
    q = torch.randn(*shape, dtype=torch.float32)
    k = torch.randn(*shape, dtype=torch.float32)
    q_ref = q.clone()
    k_ref = k.clone()
    q_ptr = q.data_ptr()
    k_ptr = k.data_ptr()

    rope = get_rope(head_size=8, rotary_dim=8, max_position=64, base=10000.0)
    q_out, k_out = rope(positions, q, k)

    cos, sin = _default_cos_sin(positions, rotary_dim=8, base=10000.0)
    if len(shape) == 2:
        q_ref = q_ref.view(shape[0], -1, 8)
        k_ref = k_ref.view(shape[0], -1, 8)
    expected_q = _reference_rope(q_ref, cos, sin).view(shape)
    expected_k = _reference_rope(k_ref, cos, sin).view(shape)

    assert q_out.data_ptr() == q_ptr
    assert k_out.data_ptr() == k_ptr
    torch.testing.assert_close(q_out, expected_q, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(k_out, expected_k, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("shape", [(9, 32), (9, 4, 8)])
def test_partial_rope_matches_reference_keeps_tail_and_is_inplace(shape: tuple[int, ...]) -> None:
    torch.manual_seed(4)
    positions = torch.arange(shape[0], dtype=torch.long)
    q = torch.randn(*shape, dtype=torch.float32)
    k = torch.randn(*shape, dtype=torch.float32)
    q_ref = q.clone()
    k_ref = k.clone()
    q_tail = (q.view(shape[0], -1, 8) if len(shape) == 2 else q)[..., 4:].clone()
    k_tail = (k.view(shape[0], -1, 8) if len(shape) == 2 else k)[..., 4:].clone()
    q_ptr = q.data_ptr()
    k_ptr = k.data_ptr()

    rope = get_rope(head_size=8, rotary_dim=4, max_position=64, base=10000.0)
    q_out, k_out = rope(positions, q, k)

    cos, sin = _default_cos_sin(positions, rotary_dim=4, base=10000.0)
    q_view = q_ref.view(shape[0], -1, 8) if len(shape) == 2 else q_ref
    k_view = k_ref.view(shape[0], -1, 8) if len(shape) == 2 else k_ref
    expected_q = q_view.clone()
    expected_k = k_view.clone()
    expected_q[..., :4] = _reference_rope(expected_q[..., :4], cos, sin)
    expected_k[..., :4] = _reference_rope(expected_k[..., :4], cos, sin)
    expected_q = expected_q.view(shape)
    expected_k = expected_k.view(shape)

    assert q_out.data_ptr() == q_ptr
    assert k_out.data_ptr() == k_ptr
    torch.testing.assert_close(q_out, expected_q, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(k_out, expected_k, rtol=1e-6, atol=1e-6)
    q_out_tail = (q_out.view(shape[0], -1, 8) if len(shape) == 2 else q_out)[..., 4:]
    k_out_tail = (k_out.view(shape[0], -1, 8) if len(shape) == 2 else k_out)[..., 4:]
    torch.testing.assert_close(q_out_tail, q_tail, rtol=0, atol=0)
    torch.testing.assert_close(k_out_tail, k_tail, rtol=0, atol=0)


def test_gemma4_proportional_rope_matches_reference_and_is_inplace() -> None:
    torch.manual_seed(5)
    positions = torch.arange(9, dtype=torch.long)
    q = torch.randn(9, 4, 8, dtype=torch.float32)
    k = torch.randn(9, 4, 8, dtype=torch.float32)
    q_ref = q.clone()
    k_ref = k.clone()
    q_ptr = q.data_ptr()
    k_ptr = k.data_ptr()

    rope = get_gemma4_proportional_rope(head_size=8, rotary_dim=4, max_position=64, base=10000.0)
    q_out, k_out = rope(positions, q, k)

    zero_pad_angles = (8 // 2) - (4 // 2)
    cos, sin = _default_cos_sin(
        positions,
        rotary_dim=4,
        base=10000.0,
        denom=8,
        zero_pad_angles=zero_pad_angles,
    )
    expected_q = _reference_rope(q_ref, cos, sin)
    expected_k = _reference_rope(k_ref, cos, sin)

    assert q_out.data_ptr() == q_ptr
    assert k_out.data_ptr() == k_ptr
    torch.testing.assert_close(q_out, expected_q, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(k_out, expected_k, rtol=1e-6, atol=1e-6)
