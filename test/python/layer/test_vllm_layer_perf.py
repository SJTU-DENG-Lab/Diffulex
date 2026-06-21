from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

import pytest
import torch
import torch.nn.functional as F

from diffulex.layer.activation import GeluAndMul, SiluAndMul
from diffulex.layer.layernorm import RMSNorm
from diffulex.layer.rotary_embedding import apply_rotary_emb, get_rope
from diffulex.layer.vllm_backend import set_vllm_layers_enabled


pytestmark = [
    pytest.mark.vllm_layer_perf,
    pytest.mark.skipif(
        os.getenv("DIFFULEX_RUN_VLLM_LAYER_PERF", "0") != "1",
        reason="set DIFFULEX_RUN_VLLM_LAYER_PERF=1 to run vLLM layer microbenchmarks",
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@dataclass(frozen=True)
class BenchResult:
    name: str
    shape: str
    wrapper_ms: float
    vllm_ms: float
    compiled_ms: float
    eager_ms: float


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def _env_dtype(name: str, default: torch.dtype) -> torch.dtype:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.lower()
    if normalized in ("bf16", "bfloat16"):
        return torch.bfloat16
    if normalized in ("fp16", "float16", "half"):
        return torch.float16
    if normalized in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype for {name}: {raw}")


def _env_shape_cases(name: str, defaults: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    raw = os.getenv(name)
    if not raw:
        return defaults
    cases = []
    for case in raw.split(";"):
        case = case.strip()
        if not case:
            continue
        cases.append(tuple(int(part) for part in case.split(",")))
    return cases


def _bench_cuda(fn: Callable[[], object], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iters


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


def _vllm_config_context():
    try:
        from vllm.config.vllm import VllmConfig, set_current_vllm_config

        return set_current_vllm_config(VllmConfig())
    except Exception:
        return nullcontext()


def _make_wrapper_module(
    cls: type[torch.nn.Module],
    device: torch.device,
    dtype: torch.dtype,
    *args,
    **kwargs,
):
    set_vllm_layers_enabled(True)
    with _vllm_config_context():
        module = cls(*args, **kwargs).to(device=device, dtype=dtype)
    if getattr(module, "_vllm_impl", None) is None:
        pytest.skip(f"{cls.__name__} did not construct a vLLM backend")
    return module


def _make_compiled_module(
    cls: type[torch.nn.Module],
    device: torch.device,
    dtype: torch.dtype,
    *args,
    **kwargs,
):
    set_vllm_layers_enabled(False)
    return cls(*args, **kwargs).to(device=device, dtype=dtype)


def _vllm_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    import vllm._custom_ops  # noqa: F401

    out = torch.empty(x.shape[:-1] + (x.shape[-1] // 2,), device=x.device, dtype=x.dtype)
    torch.ops._C.silu_and_mul(out, x)
    return out


def _vllm_gelu_tanh_and_mul(x: torch.Tensor) -> torch.Tensor:
    import vllm._custom_ops  # noqa: F401

    out = torch.empty(x.shape[:-1] + (x.shape[-1] // 2,), device=x.device, dtype=x.dtype)
    torch.ops._C.gelu_tanh_and_mul(out, x)
    return out


def _bench_activation(
    *,
    name: str,
    cls: type[torch.nn.Module],
    vllm_fn: Callable[[torch.Tensor], torch.Tensor],
    eager_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    shape: str,
    warmup: int,
    iters: int,
) -> BenchResult:
    wrapper_mod = _make_wrapper_module(cls, x.device, x.dtype)
    compiled_mod = _make_compiled_module(cls, x.device, x.dtype)

    expected = eager_fn(x)
    _assert_close(vllm_fn(x), expected)
    _assert_close(wrapper_mod(x), expected)
    _assert_close(compiled_mod(x), expected)

    return BenchResult(
        name=name,
        shape=shape,
        wrapper_ms=_bench_cuda(lambda: wrapper_mod(x), warmup=warmup, iters=iters),
        vllm_ms=_bench_cuda(lambda: vllm_fn(x), warmup=warmup, iters=iters),
        compiled_ms=_bench_cuda(lambda: compiled_mod(x), warmup=warmup, iters=iters),
        eager_ms=_bench_cuda(lambda: eager_fn(x), warmup=warmup, iters=iters),
    )


def _make_rmsnorm_pair(hidden_size: int, device: torch.device, dtype: torch.dtype):
    wrapper_mod = _make_wrapper_module(RMSNorm, device, dtype, hidden_size)
    compiled_mod = _make_compiled_module(RMSNorm, device, dtype, hidden_size)
    with torch.no_grad():
        weight = torch.randn(hidden_size, device=device, dtype=dtype)
        wrapper_mod.weight.copy_(weight)
        compiled_mod.weight.copy_(weight)
    return wrapper_mod, compiled_mod, weight


def _vllm_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])
    out = torch.empty_like(x_2d)
    from vllm import _custom_ops as ops

    ops.rms_norm(out, x_2d, weight, eps)
    return out.view(orig_shape)


def _vllm_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])
    residual_2d = residual.view(-1, residual.shape[-1])
    from vllm import _custom_ops as ops

    ops.fused_add_rms_norm(x_2d, residual_2d, weight, eps)
    return x_2d.view(orig_shape), residual_2d.view(orig_shape)


def _rmsnorm_eager(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_fp32 = x.to(torch.float32)
    out = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return out.to(x.dtype) * weight


def _rmsnorm_add_eager(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_fp32 = x.to(torch.float32) + residual.to(torch.float32)
    residual_out = x_fp32.to(x.dtype)
    out = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return out.to(x.dtype) * weight, residual_out


def _bench_rmsnorm(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    shape: str,
    warmup: int,
    iters: int,
) -> list[BenchResult]:
    hidden_size = int(x.shape[-1])
    wrapper_mod, compiled_mod, weight = _make_rmsnorm_pair(hidden_size, x.device, x.dtype)

    expected = _rmsnorm_eager(x, weight)
    _assert_close(_vllm_rmsnorm(x, weight), expected)
    _assert_close(wrapper_mod(x), expected)
    _assert_close(compiled_mod(x), expected)

    expected_add, expected_residual = _rmsnorm_add_eager(x, residual, weight)
    vllm_add, vllm_residual = _vllm_fused_add_rmsnorm(x.clone(), residual.clone(), weight)
    wrapper_add, wrapper_residual = wrapper_mod(x.clone(), residual.clone())
    compiled_add, compiled_residual = compiled_mod(x.clone(), residual.clone())
    _assert_close(vllm_add, expected_add)
    _assert_close(vllm_residual, expected_residual)
    _assert_close(wrapper_add, expected_add)
    _assert_close(wrapper_residual, expected_residual)
    _assert_close(compiled_add, expected_add)
    _assert_close(compiled_residual, expected_residual)

    return [
        BenchResult(
            name="RMSNorm",
            shape=shape,
            wrapper_ms=_bench_cuda(lambda: wrapper_mod(x), warmup=warmup, iters=iters),
            vllm_ms=_bench_cuda(lambda: _vllm_rmsnorm(x, weight), warmup=warmup, iters=iters),
            compiled_ms=_bench_cuda(lambda: compiled_mod(x), warmup=warmup, iters=iters),
            eager_ms=_bench_cuda(lambda: _rmsnorm_eager(x, weight), warmup=warmup, iters=iters),
        ),
        BenchResult(
            name="RMSNorm + residual",
            shape=shape,
            wrapper_ms=_bench_cuda(
                lambda: wrapper_mod(x.clone(), residual.clone()),
                warmup=warmup,
                iters=iters,
            ),
            vllm_ms=_bench_cuda(
                lambda: _vllm_fused_add_rmsnorm(x.clone(), residual.clone(), weight),
                warmup=warmup,
                iters=iters,
            ),
            compiled_ms=_bench_cuda(
                lambda: compiled_mod(x.clone(), residual.clone()),
                warmup=warmup,
                iters=iters,
            ),
            eager_ms=_bench_cuda(
                lambda: _rmsnorm_add_eager(x.clone(), residual.clone(), weight),
                warmup=warmup,
                iters=iters,
            ),
        ),
    ]


class EagerRotaryEmbedding(torch.nn.Module):
    def __init__(self, head_size: int, rotary_dim: int, max_position: int, base: float) -> None:
        super().__init__()
        self.head_size = int(head_size)
        self.rotary_dim = int(rotary_dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        t = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos_sin_cache", torch.cat((freqs.cos(), freqs.sin()), dim=-1), persistent=False)

    def _apply_one(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        cos, sin = self.cos_sin_cache[positions].chunk(2, dim=-1)
        x_shape = x.shape
        x_view = x.view(x.shape[0], -1, self.head_size) if x.dim() == 2 else x
        out = x_view.clone()
        out[..., : self.rotary_dim] = apply_rotary_emb(out[..., : self.rotary_dim], cos, sin)
        return out.view(x_shape)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._apply_one(positions, query), self._apply_one(positions, key)


def _make_rope_pair(
    *,
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    device: torch.device,
):
    set_vllm_layers_enabled(True)
    with _vllm_config_context():
        vllm_rope = get_rope(head_size, rotary_dim, max_position, base).to(device=device)
    if type(vllm_rope).__name__ != "VllmRotaryEmbeddingAdapter":
        pytest.skip(f"RoPE did not construct a vLLM backend, got {type(vllm_rope).__name__}")
    wrapper_rope = vllm_rope

    set_vllm_layers_enabled(False)
    compiled_rope = get_rope(head_size, rotary_dim, max_position, base).to(device=device)
    eager_rope = EagerRotaryEmbedding(head_size, rotary_dim, max_position, base).to(device=device)
    return wrapper_rope, compiled_rope, eager_rope


def _bench_rope(
    *,
    name: str,
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    shape: str,
    warmup: int,
    iters: int,
) -> BenchResult:
    wrapper_rope, compiled_rope, eager_rope = _make_rope_pair(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position=max(int(positions.numel()) * 2, 2048),
        base=10000.0,
        device=q.device,
    )

    expected_q, expected_k = eager_rope(positions, q.clone(), k.clone())
    vllm_q, vllm_k = wrapper_rope.rotary_emb.forward_cuda(positions, q.clone(), k.clone())
    wrapper_q, wrapper_k = wrapper_rope(positions, q.clone(), k.clone())
    compiled_q, compiled_k = compiled_rope(positions, q.clone(), k.clone())
    _assert_close(vllm_q, expected_q)
    _assert_close(vllm_k, expected_k)
    _assert_close(wrapper_q, expected_q)
    _assert_close(wrapper_k, expected_k)
    _assert_close(compiled_q, expected_q)
    _assert_close(compiled_k, expected_k)

    return BenchResult(
        name=name,
        shape=shape,
        wrapper_ms=_bench_cuda(
            lambda: wrapper_rope(positions, q.clone(), k.clone()),
            warmup=warmup,
            iters=iters,
        ),
        vllm_ms=_bench_cuda(
            lambda: wrapper_rope.rotary_emb.forward_cuda(positions, q.clone(), k.clone()),
            warmup=warmup,
            iters=iters,
        ),
        compiled_ms=_bench_cuda(
            lambda: compiled_rope(positions, q.clone(), k.clone()),
            warmup=warmup,
            iters=iters,
        ),
        eager_ms=_bench_cuda(
            lambda: eager_rope(positions, q.clone(), k.clone()),
            warmup=warmup,
            iters=iters,
        ),
    )


def _print_results(results: list[BenchResult], *, device: torch.device, dtype: torch.dtype, iters: int) -> None:
    print(f"\nvLLM layer microbench: device={device}, dtype={dtype}, iters={iters}")
    print(
        f"{'op':<24} {'shape':<38} {'wrapper ms':>11} {'vllm ms':>10} {'compile ms':>12} "
        f"{'eager ms':>10} {'wrap/vllm':>10} {'wrap/compile':>13}"
    )
    for item in results:
        print(
            f"{item.name:<24} "
            f"{item.shape:<38} "
            f"{item.wrapper_ms:>11.4f} "
            f"{item.vllm_ms:>10.4f} "
            f"{item.compiled_ms:>12.4f} "
            f"{item.eager_ms:>10.4f} "
            f"{item.wrapper_ms / item.vllm_ms:>10.3f} "
            f"{item.wrapper_ms / item.compiled_ms:>13.3f}"
        )


def test_vllm_layer_operator_microbenchmarks() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = _env_dtype("DIFFULEX_VLLM_LAYER_PERF_DTYPE", torch.bfloat16)
    warmup = _env_int("DIFFULEX_VLLM_LAYER_PERF_WARMUP", 20)
    iters = _env_int("DIFFULEX_VLLM_LAYER_PERF_ITERS", 200)

    activation_cases = _env_shape_cases(
        "DIFFULEX_VLLM_LAYER_PERF_ACT_CASES",
        [
            (1, 2112),
            (32, 2112),
            (256, 2112),
            (1024, 2112),
            (2048, 2112),
        ],
    )
    block_norm_cases = _env_shape_cases(
        "DIFFULEX_VLLM_LAYER_PERF_BLOCK_NORM_CASES",
        [
            (1, 2816),
            (32, 2816),
            (256, 2816),
            (1024, 2816),
            (2048, 2816),
        ],
    )
    head_norm_cases = _env_shape_cases(
        "DIFFULEX_VLLM_LAYER_PERF_HEAD_NORM_CASES",
        [
            (1, 16, 512),
            (32, 16, 512),
            (256, 16, 512),
            (1024, 16, 512),
            (2048, 16, 512),
            (256, 8, 512),
            (256, 2, 512),
        ],
    )
    rope_cases = _env_shape_cases(
        "DIFFULEX_VLLM_LAYER_PERF_ROPE_CASES",
        [
            (1, 16, 8, 512, 128),
            (32, 16, 8, 512, 128),
            (256, 16, 8, 512, 128),
            (1024, 16, 8, 512, 128),
            (2048, 16, 8, 512, 128),
            (256, 16, 2, 512, 128),
        ],
    )

    results: list[BenchResult] = []
    for num_tokens, intermediate_size in activation_cases:
        shape = f"T={num_tokens},I={intermediate_size},x=[T,2I]"
        x_act = torch.randn(num_tokens, intermediate_size * 2, device=device, dtype=dtype)
        results.extend(
            [
                _bench_activation(
                    name="SiluAndMul",
                    cls=SiluAndMul,
                    vllm_fn=_vllm_silu_and_mul,
                    eager_fn=lambda x: F.silu(x.chunk(2, dim=-1)[0]) * x.chunk(2, dim=-1)[1],
                    x=x_act,
                    shape=shape,
                    warmup=warmup,
                    iters=iters,
                ),
                _bench_activation(
                    name="GeluAndMul",
                    cls=GeluAndMul,
                    vllm_fn=_vllm_gelu_tanh_and_mul,
                    eager_fn=lambda x: F.gelu(x.chunk(2, dim=-1)[0], approximate="tanh") * x.chunk(2, dim=-1)[1],
                    x=x_act,
                    shape=shape,
                    warmup=warmup,
                    iters=iters,
                ),
            ]
        )

    for num_tokens, hidden_size in block_norm_cases:
        shape = f"T={num_tokens},H={hidden_size},x=[T,H]"
        x_norm = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
        residual = torch.randn_like(x_norm)
        results.extend(_bench_rmsnorm(x=x_norm, residual=residual, shape=shape, warmup=warmup, iters=iters))

    for num_tokens, num_heads, head_size in head_norm_cases:
        shape = f"T={num_tokens},heads={num_heads},D={head_size},x=[T,heads,D]"
        x_norm = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=dtype)
        residual = torch.randn_like(x_norm)
        results.extend(_bench_rmsnorm(x=x_norm, residual=residual, shape=shape, warmup=warmup, iters=iters))

    for num_tokens, q_heads, kv_heads, head_size, rotary_dim in rope_cases:
        shape = f"T={num_tokens},QH={q_heads},KVH={kv_heads},D={head_size},RD={rotary_dim}"
        positions = torch.arange(num_tokens, device=device, dtype=torch.long)
        q = torch.randn(num_tokens, q_heads, head_size, device=device, dtype=dtype)
        k = torch.randn(num_tokens, kv_heads, head_size, device=device, dtype=dtype)
        results.append(
            _bench_rope(
                name="RoPE dgemma",
                q=q,
                k=k,
                positions=positions,
                head_size=head_size,
                rotary_dim=rotary_dim,
                shape=shape,
                warmup=warmup,
                iters=iters,
            )
        )
        results.append(
            _bench_rope(
                name="RoPE full-head",
                q=q,
                k=k,
                positions=positions,
                head_size=head_size,
                rotary_dim=head_size,
                shape=shape,
                warmup=warmup,
                iters=iters,
            )
        )

    _print_results(results, device=device, dtype=dtype, iters=iters)
