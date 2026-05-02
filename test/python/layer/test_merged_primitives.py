from __future__ import annotations

import torch

import diffulex.distributed.parallel_state as parallel_state
from diffulex.layer.activation import SiluAndMul
from diffulex.layer.layernorm import RMSNorm
from diffulex.layer.linear import MergedColumnParallelLinear, QKVParallelLinear


def _mock_tp(*, tp_size: int = 1, global_rank: int = 0):
    parallel_state.reset_parallel_state()
    parallel_state.PARALLEL_STATE = parallel_state.build_parallel_state_for_test(
        tp_size=tp_size,
        ep_size=1,
        dp_size=1,
        world_size=tp_size,
        global_rank=global_rank,
    )


def test_rmsnorm_kernel_dispatch_supports_3d_residual(monkeypatch):
    calls = {"rms": 0, "add": 0}

    def fake_rms(x, weight, eps):
        calls["rms"] += 1
        return x + weight

    def fake_add(x, residual, weight, eps):
        calls["add"] += 1
        residual.add_(x)
        x.copy_(residual + weight)

    monkeypatch.setattr("diffulex.layer.layernorm._get_sgl_kernel_op", lambda name: {"rmsnorm": fake_rms, "fused_add_rmsnorm": fake_add}.get(name))

    norm = RMSNorm(4).cuda()
    x = torch.ones(2, 3, 4, device="cuda")
    residual = torch.full_like(x, 2)

    out = norm(x)
    out_with_residual, residual_out = norm(x, residual)

    assert calls == {"rms": 1, "add": 1}
    assert out.shape == x.shape
    assert out_with_residual.shape == x.shape
    assert residual_out.shape == x.shape


def test_silu_and_mul_kernel_dispatch(monkeypatch):
    called = {"kernel": 0}

    def fake_op(x, out):
        called["kernel"] += 1
        out.copy_(x[..., : x.shape[-1] // 2] + x[..., x.shape[-1] // 2 :])

    monkeypatch.setattr("diffulex.layer.activation._get_sgl_kernel_silu_and_mul", lambda: fake_op)

    act = SiluAndMul()
    x = torch.arange(12, device="cuda", dtype=torch.float32).reshape(2, 6)
    out = act(x)

    assert called["kernel"] == 1
    assert out.shape == (2, 3)


def test_qkv_parallel_linear_loader_supports_kv_replication():
    _mock_tp(tp_size=4, global_rank=2)
    layer = QKVParallelLinear(hidden_size=5, head_size=3, total_num_heads=8, total_num_kv_heads=2, bias=True)

    q_total = 8 * 3
    kv_total = 2 * 3
    loaded = torch.arange((q_total + kv_total + kv_total) * 5, dtype=torch.float32).reshape(q_total + kv_total + kv_total, 5)
    layer.weight.weight_loader(layer.weight, loaded)

    q, k, v = loaded.split((q_total, kv_total, kv_total), dim=0)
    expected = torch.cat((q.chunk(4, dim=0)[2], k.chunk(2, dim=0)[1], v.chunk(2, dim=0)[1]), dim=0)
    assert torch.equal(layer.weight, expected)

    loaded_bias = torch.arange(q_total + kv_total + kv_total, dtype=torch.float32)
    layer.bias.weight_loader(layer.bias, loaded_bias)
    q_bias, k_bias, v_bias = loaded_bias.split((q_total, kv_total, kv_total), dim=0)
    expected_bias = torch.cat((q_bias.chunk(4, dim=0)[2], k_bias.chunk(2, dim=0)[1], v_bias.chunk(2, dim=0)[1]), dim=0)
    assert torch.equal(layer.bias, expected_bias)


def test_merged_column_parallel_linear_loader_supports_split_and_fused_weights():
    _mock_tp(tp_size=2, global_rank=1)
    layer = MergedColumnParallelLinear(4, [6, 4], bias=True)

    fused = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    layer.weight.weight_loader(layer.weight, fused)
    expected_weight = torch.cat((fused[:6].chunk(2, dim=0)[1], fused[6:].chunk(2, dim=0)[1]), dim=0)
    assert torch.equal(layer.weight, expected_weight)

    split_bias = torch.arange(6, dtype=torch.float32)
    layer.bias.weight_loader(layer.bias, split_bias, 0)
    assert torch.equal(layer.bias[:3], split_bias.chunk(2, dim=0)[1])


def test_rmsnorm_supports_split_views_without_inplace_errors():
    norm = RMSNorm(4)
    packed = torch.randn(3, 12)
    q, k, _v = packed.split((4, 4, 4), dim=-1)

    q_out = norm(q.reshape(3, 1, 4))
    k_out = norm(k.reshape(3, 1, 4))

    assert q_out.shape == (3, 1, 4)
    assert k_out.shape == (3, 1, 4)
