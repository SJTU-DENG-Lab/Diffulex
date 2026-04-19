from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffulex.moe.dispatcher.deepep_dispatcher import DeepEPDispatcher
from diffulex.moe.metadata import DeepEPDispatchMetadata
from diffulex_kernel import fused_expert_packed
from diffulex_kernel.python.fused_moe_triton import fused_moe


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _pick_free_port() -> int:
    configured = os.environ.get("DIFFULEX_DEEPEP_SMOKE_PORT")
    if configured:
        return int(configured)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(frozen=True)
class SmokeConfig:
    world_size: int
    num_tokens: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    warmup: int
    iters: int
    deepep_mode: str
    num_max_dispatch_tokens_per_rank: int
    master_port: int


def _make_inputs(config: SmokeConfig, device: torch.device):
    generator = torch.Generator(device=device)
    generator.manual_seed(20260419)
    hidden_states = torch.randn(
        (config.num_tokens, config.hidden_size),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ) * 0.1
    w13 = torch.randn(
        (config.num_experts, config.intermediate_size * 2, config.hidden_size),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ) * 0.02
    w2 = torch.randn(
        (config.num_experts, config.hidden_size, config.intermediate_size),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ) * 0.02
    topk_ids = torch.randint(
        0,
        config.num_experts,
        (config.num_tokens, config.top_k),
        device=device,
        dtype=torch.int32,
        generator=generator,
    )
    topk_weights = torch.rand(
        (config.num_tokens, config.top_k),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return hidden_states, w13, w2, topk_ids, topk_weights


def _broadcast_inputs(tensors) -> None:
    for tensor in tensors:
        dist.broadcast(tensor, src=0)


def _reference_forward(hidden_states, w13, w2, topk_ids, topk_weights):
    return fused_moe(
        hidden_states=hidden_states,
        w13=w13,
        w2=w2,
        topk_ids=topk_ids,
        topk_weights=topk_weights.to(hidden_states.dtype),
        local_expert_start=0,
        hidden_act="silu",
    )


def _deepep_forward(config: SmokeConfig, rank: int, dispatcher, hidden_states, w13, w2, topk_ids, topk_weights):
    num_local_experts = config.num_experts // config.world_size
    local_start = rank * num_local_experts
    local_end = local_start + num_local_experts
    dispatched = dispatcher.dispatch(hidden_states, topk_ids, topk_weights)
    metadata = dispatched.metadata
    assert isinstance(metadata, DeepEPDispatchMetadata)
    if metadata.total_recv_slots == 0:
        recv_slot_outputs = torch.empty(
            (0, config.hidden_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    else:
        recv_slot_outputs = fused_expert_packed(
            hidden_states=dispatched.recv_hidden_states,
            w13=w13[local_start:local_end].contiguous(),
            w2=w2[local_start:local_end].contiguous(),
            execution_metadata=metadata.to_expert_execution_metadata(),
            hidden_act="silu",
        ).contiguous()
        recv_slot_outputs.mul_(metadata.recv_weights.to(recv_slot_outputs.dtype).unsqueeze(-1))
    return dispatcher.combine(recv_slot_outputs, metadata)


def _time_cuda(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def _worker(rank: int, config: SmokeConfig, return_dict):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{config.master_port}",
        rank=rank,
        world_size=config.world_size,
    )
    device = torch.device("cuda", rank)
    try:
        hidden_states, w13, w2, topk_ids, topk_weights = _make_inputs(config, device)
        _broadcast_inputs((hidden_states, w13, w2, topk_ids, topk_weights))
        num_local_experts = config.num_experts // config.world_size
        dispatcher = DeepEPDispatcher(
            ep_group=dist.group.WORLD,
            ep_size=config.world_size,
            num_local_experts=num_local_experts,
            top_k=config.top_k,
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            deepep_mode=config.deepep_mode,
            num_max_dispatch_tokens_per_rank=config.num_max_dispatch_tokens_per_rank,
            async_finish=False,
        )

        reference = _reference_forward(hidden_states, w13, w2, topk_ids, topk_weights)
        deepep_output = _deepep_forward(config, rank, dispatcher, hidden_states, w13, w2, topk_ids, topk_weights)
        torch.cuda.synchronize()

        max_abs = (deepep_output - reference).abs().max().item()
        rel = max_abs / reference.abs().max().clamp_min(1e-12).item()
        ref_ms = _time_cuda(
            lambda: _reference_forward(hidden_states, w13, w2, topk_ids, topk_weights),
            warmup=config.warmup,
            iters=config.iters,
        )
        deepep_ms = _time_cuda(
            lambda: _deepep_forward(config, rank, dispatcher, hidden_states, w13, w2, topk_ids, topk_weights),
            warmup=config.warmup,
            iters=config.iters,
        )
        gathered = [None for _ in range(config.world_size)]
        dist.all_gather_object(
            gathered,
            {
                "rank": rank,
                "max_abs": max_abs,
                "rel": rel,
                "ref_ms": ref_ms,
                "deepep_ms": deepep_ms,
            },
        )
        if rank == 0:
            return_dict["results"] = gathered
    finally:
        dist.destroy_process_group()


def main() -> None:
    world_size = _env_int("DIFFULEX_DEEPEP_SMOKE_WORLD_SIZE", 2)
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(f"Need at least {world_size} CUDA devices, got {torch.cuda.device_count()}.")
    num_experts = _env_int("DIFFULEX_DEEPEP_SMOKE_NUM_EXPERTS", 8)
    if num_experts % world_size != 0:
        raise ValueError(f"num_experts must be divisible by world_size, got {num_experts=} {world_size=}.")
    config = SmokeConfig(
        world_size=world_size,
        num_tokens=_env_int("DIFFULEX_DEEPEP_SMOKE_NUM_TOKENS", 128),
        hidden_size=_env_int("DIFFULEX_DEEPEP_SMOKE_HIDDEN_SIZE", 256),
        intermediate_size=_env_int("DIFFULEX_DEEPEP_SMOKE_INTERMEDIATE_SIZE", 512),
        num_experts=num_experts,
        top_k=_env_int("DIFFULEX_DEEPEP_SMOKE_TOP_K", 2),
        warmup=_env_int("DIFFULEX_DEEPEP_SMOKE_WARMUP", 5),
        iters=_env_int("DIFFULEX_DEEPEP_SMOKE_ITERS", 20),
        deepep_mode=os.environ.get("DIFFULEX_DEEPEP_SMOKE_MODE", "auto"),
        num_max_dispatch_tokens_per_rank=_env_int(
            "DIFFULEX_DEEPEP_SMOKE_NUM_MAX_DISPATCH_TOKENS_PER_RANK",
            256,
        ),
        master_port=_pick_free_port(),
    )
    manager = mp.Manager()
    return_dict = manager.dict()
    mp.spawn(_worker, args=(config, return_dict), nprocs=config.world_size, join=True)
    results = list(return_dict["results"])
    print("config", config, flush=True)
    for item in results:
        print(item, flush=True)
    print(
        "summary",
        {
            "max_abs": max(item["max_abs"] for item in results),
            "rel": max(item["rel"] for item in results),
            "ref_ms_max_rank": max(item["ref_ms"] for item in results),
            "deepep_ms_max_rank": max(item["deepep_ms"] for item in results),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
