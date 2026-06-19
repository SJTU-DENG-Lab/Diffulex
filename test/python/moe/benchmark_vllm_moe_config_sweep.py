from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Iterable

import torch

import vllm.model_executor.layers.fused_moe as vllm_fused_moe_pkg
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts


def _parse_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def _candidate_configs() -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    seen: set[tuple[tuple[str, int], ...]] = set()

    def add(**kwargs: int) -> None:
        cfg = {
            "BLOCK_SIZE_M": kwargs["block_m"],
            "BLOCK_SIZE_N": kwargs["block_n"],
            "BLOCK_SIZE_K": kwargs["block_k"],
            "GROUP_SIZE_M": kwargs["group_m"],
            "SPLIT_K": 1,
            "num_warps": kwargs["num_warps"],
            "num_stages": kwargs["num_stages"],
        }
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            configs.append(cfg)

    for block_m in (16, 32, 64):
        for block_n in (32, 64, 128):
            for block_k in (32, 64, 128):
                for group_m in (1, 8):
                    add(
                        block_m=block_m,
                        block_n=block_n,
                        block_k=block_k,
                        group_m=group_m,
                        num_warps=4,
                        num_stages=4 if block_m <= 16 else 3,
                    )

    for block_m in (64, 128):
        for block_n in (64, 128, 256):
            for block_k in (32, 64):
                for group_m in (1, 8, 16):
                    add(
                        block_m=block_m,
                        block_n=block_n,
                        block_k=block_k,
                        group_m=group_m,
                        num_warps=8,
                        num_stages=3,
                    )

    return configs


def _make_inputs(
    *,
    m: int,
    hidden: int,
    intermediate: int,
    experts: int,
    top_k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_states = torch.randn((m, hidden), device=device, dtype=torch.bfloat16)
    w1 = torch.randn((experts, intermediate * 2, hidden), device=device, dtype=torch.bfloat16)
    w2 = torch.randn((experts, hidden, intermediate), device=device, dtype=torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, top_k), device=device, dtype=torch.int64)
    topk_weights = torch.rand((m, top_k), device=device, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return hidden_states, w1, w2, topk_weights, topk_ids


def _bench_one(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    config: dict[str, int] | None,
    *,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    hidden_states, w1, w2, topk_weights, topk_ids = inputs
    torch.cuda.synchronize()

    context = _override_config_safe(config) if config is not None else _nullcontext()
    with context:
        for _ in range(warmup):
            fused_experts(hidden_states, w1, w2, topk_weights, topk_ids)
        torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fused_experts(hidden_states, w1, w2, topk_weights, topk_ids)
            end.record()
            end.synchronize()
            times.append(float(start.elapsed_time(end)))
    return statistics.median(times), min(times)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _override_config_safe:
    def __init__(self, config: dict[str, int]) -> None:
        self.config = config
        self.old_config = None

    def __enter__(self):
        self.old_config = vllm_fused_moe_pkg.get_config()
        vllm_fused_moe_pkg._config = self.config
        return None

    def __exit__(self, exc_type, exc, tb):
        vllm_fused_moe_pkg._config = self.old_config
        return False


def _iter_subset(configs: list[dict[str, int]], limit: int | None) -> Iterable[tuple[int, dict[str, int]]]:
    if limit is None or limit >= len(configs):
        yield from enumerate(configs)
        return
    step = max(1, len(configs) // limit)
    emitted = 0
    for idx in range(0, len(configs), step):
        yield idx, configs[idx]
        emitted += 1
        if emitted >= limit:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep vLLM fused MoE Triton configs for LLaDA2-style shapes.")
    parser.add_argument("--m-values", default="1,2,4,8,16,32,64,128,256,512,1024,4096")
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--limit-configs", type=int, default=None)
    parser.add_argument("--skip-default", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    torch.manual_seed(0)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    configs = _candidate_configs()
    results = []
    for m in _parse_ints(args.m_values):
        inputs = _make_inputs(
            m=m,
            hidden=args.hidden,
            intermediate=args.intermediate,
            experts=args.experts,
            top_k=args.top_k,
            device=device,
        )
        best = None
        if not args.skip_default:
            median_ms, min_ms = _bench_one(inputs, None, warmup=args.warmup, iters=args.iters)
            default_row = {
                "M": m,
                "config_index": "default",
                "median_ms": median_ms,
                "min_ms": min_ms,
                "config": None,
            }
            results.append(default_row)
            best = default_row
            print(f"M={m:<5} default median={median_ms:.4f} ms min={min_ms:.4f} ms")

        for config_idx, config in _iter_subset(configs, args.limit_configs):
            try:
                median_ms, min_ms = _bench_one(inputs, config, warmup=args.warmup, iters=args.iters)
            except Exception as exc:
                print(f"M={m:<5} config#{config_idx:<3} failed: {exc}")
                continue
            row = {
                "M": m,
                "config_index": config_idx,
                "median_ms": median_ms,
                "min_ms": min_ms,
                "config": config,
            }
            results.append(row)
            if best is None or median_ms < best["median_ms"]:
                best = row

        if best is None:
            print(f"M={m:<5} no valid configs")
        else:
            print(
                f"M={m:<5} best median={best['median_ms']:.4f} ms "
                f"min={best['min_ms']:.4f} ms config#{best['config_index']} {best['config']}"
            )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
