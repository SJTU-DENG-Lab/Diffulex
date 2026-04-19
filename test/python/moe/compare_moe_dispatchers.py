from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffulex.moe.dispatcher.naive_dispatcher import NaiveA2ADispatcher
from diffulex_kernel.python.fused_moe_triton import fused_moe


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _pick_free_port() -> int:
    configured = os.environ.get("DIFFULEX_MOE_COMPARE_PORT")
    if configured:
        return int(configured)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(frozen=True)
class CompareConfig:
    world_size: int
    num_tokens: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    warmup: int
    iters: int
    check_serial: bool
    dtype: torch.dtype
    master_port: int


def _make_inputs(config: CompareConfig, device: torch.device):
    generator = torch.Generator(device=device)
    generator.manual_seed(1234)
    hidden_states = torch.randn(
        (config.num_tokens, config.hidden_size),
        device=device,
        dtype=config.dtype,
        generator=generator,
    ) * 0.1
    w13 = torch.randn(
        (config.num_experts, config.intermediate_size * 2, config.hidden_size),
        device=device,
        dtype=config.dtype,
        generator=generator,
    ) * 0.02
    w2 = torch.randn(
        (config.num_experts, config.hidden_size, config.intermediate_size),
        device=device,
        dtype=config.dtype,
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
        dtype=config.dtype,
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
        topk_weights=topk_weights,
        local_expert_start=0,
        hidden_act="silu",
    )


def _serial_reference_forward(hidden_states, w13, w2, topk_ids, topk_weights):
    num_tokens, hidden_size = hidden_states.shape
    intermediate_size = w2.shape[-1]
    output = hidden_states.new_zeros((num_tokens, hidden_size))
    for token_idx in range(num_tokens):
        token_output = torch.zeros((hidden_size,), device=hidden_states.device, dtype=torch.float32)
        token_hidden = hidden_states[token_idx]
        for slot_idx in range(topk_ids.shape[1]):
            expert_idx = int(topk_ids[token_idx, slot_idx].item())
            expert_w13 = w13[expert_idx]
            gate_proj = expert_w13[:intermediate_size]
            up_proj = expert_w13[intermediate_size:]
            gate = torch.matmul(token_hidden, gate_proj.transpose(0, 1))
            up = torch.matmul(token_hidden, up_proj.transpose(0, 1))
            activated = torch.nn.functional.silu(gate) * up
            expert_output = torch.matmul(activated, w2[expert_idx].transpose(0, 1))
            token_output += expert_output.float() * topk_weights[token_idx, slot_idx].float()
        output[token_idx] = token_output.to(output.dtype)
    return output


def _should_check_serial(config: CompareConfig) -> bool:
    if not config.check_serial:
        return False
    return config.num_tokens <= 128 and config.hidden_size <= 512 and config.intermediate_size <= 1024


def _tp_forward(config: CompareConfig, rank: int, hidden_states, w13, w2, topk_ids, topk_weights):
    num_local_experts = config.num_experts // config.world_size
    local_start = rank * num_local_experts
    local_end = local_start + num_local_experts
    partial = fused_moe(
        hidden_states=hidden_states,
        w13=w13[local_start:local_end].contiguous(),
        w2=w2[local_start:local_end].contiguous(),
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        local_expert_start=local_start,
        hidden_act="silu",
    )
    dist.all_reduce(partial)
    return partial


def _naive_forward(config: CompareConfig, rank: int, hidden_states, w13, w2, topk_ids, topk_weights):
    num_local_experts = config.num_experts // config.world_size
    dispatcher = NaiveA2ADispatcher(
        ep_group=dist.group.WORLD,
        ep_size=config.world_size,
        num_local_experts=num_local_experts,
        top_k=config.top_k,
    )
    local_start = rank * num_local_experts
    local_end = local_start + num_local_experts
    dispatched = dispatcher.dispatch(hidden_states, topk_ids, topk_weights)
    if dispatched.metadata.total_recv_slots == 0:
        recv_slot_outputs = torch.empty(
            (0, config.hidden_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    else:
        recv_topk_ids = dispatched.recv_local_expert_ids[:, None].contiguous()
        recv_topk_weights = torch.ones(
            (dispatched.metadata.total_recv_slots, 1),
            device=hidden_states.device,
            dtype=topk_weights.dtype,
        )
        recv_slot_outputs = fused_moe(
            hidden_states=dispatched.recv_hidden_states,
            w13=w13[local_start:local_end].contiguous(),
            w2=w2[local_start:local_end].contiguous(),
            topk_ids=recv_topk_ids,
            topk_weights=recv_topk_weights,
            local_expert_start=0,
            hidden_act="silu",
        ).contiguous()
        recv_slot_outputs.mul_(dispatched.recv_weights.to(recv_slot_outputs.dtype).unsqueeze(-1))
    return dispatcher.combine(recv_slot_outputs, dispatched.metadata)


def _time_cuda(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def _worker(rank: int, config: CompareConfig, return_dict):
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
        reference = _reference_forward(hidden_states, w13, w2, topk_ids, topk_weights)
        serial_reference = (
            _serial_reference_forward(hidden_states, w13, w2, topk_ids, topk_weights)
            if _should_check_serial(config)
            else None
        )
        tp_output = _tp_forward(config, rank, hidden_states, w13, w2, topk_ids, topk_weights)
        naive_output = _naive_forward(config, rank, hidden_states, w13, w2, topk_ids, topk_weights)
        torch.cuda.synchronize()

        serial_max_abs = None
        if serial_reference is not None:
            serial_max_abs = (reference - serial_reference).abs().max().item()
        tp_max_abs = (tp_output - reference).abs().max().item()
        naive_max_abs = (naive_output - reference).abs().max().item()
        naive_vs_tp_max_abs = (naive_output - tp_output).abs().max().item()
        reference_norm = reference.abs().max().clamp_min(1e-12).item()

        ref_ms = _time_cuda(
            lambda: _reference_forward(hidden_states, w13, w2, topk_ids, topk_weights),
            warmup=config.warmup,
            iters=config.iters,
        )
        tp_ms = _time_cuda(
            lambda: _tp_forward(config, rank, hidden_states, w13, w2, topk_ids, topk_weights),
            warmup=config.warmup,
            iters=config.iters,
        )
        naive_ms = _time_cuda(
            lambda: _naive_forward(config, rank, hidden_states, w13, w2, topk_ids, topk_weights),
            warmup=config.warmup,
            iters=config.iters,
        )

        gathered = [None for _ in range(config.world_size)]
        dist.all_gather_object(
            gathered,
            {
                "rank": rank,
                "serial_max_abs": serial_max_abs,
                "tp_max_abs": tp_max_abs,
                "naive_max_abs": naive_max_abs,
                "naive_vs_tp_max_abs": naive_vs_tp_max_abs,
                "tp_rel": tp_max_abs / reference_norm,
                "naive_rel": naive_max_abs / reference_norm,
                "naive_vs_tp_rel": naive_vs_tp_max_abs / reference_norm,
                "ref_ms": ref_ms,
                "tp_ms": tp_ms,
                "naive_ms": naive_ms,
            },
        )
        if rank == 0:
            return_dict["results"] = gathered
    finally:
        dist.destroy_process_group()


def main() -> None:
    world_size = _env_int("DIFFULEX_MOE_COMPARE_WORLD_SIZE", 2)
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(f"Need at least {world_size} CUDA devices, got {torch.cuda.device_count()}.")
    num_experts = _env_int("DIFFULEX_MOE_COMPARE_NUM_EXPERTS", 8)
    if num_experts % world_size != 0:
        raise ValueError(f"num_experts must be divisible by world_size, got {num_experts=} {world_size=}.")
    config = CompareConfig(
        world_size=world_size,
        num_tokens=_env_int("DIFFULEX_MOE_COMPARE_NUM_TOKENS", 256),
        hidden_size=_env_int("DIFFULEX_MOE_COMPARE_HIDDEN_SIZE", 256),
        intermediate_size=_env_int("DIFFULEX_MOE_COMPARE_INTERMEDIATE_SIZE", 512),
        num_experts=num_experts,
        top_k=_env_int("DIFFULEX_MOE_COMPARE_TOP_K", 2),
        warmup=_env_int("DIFFULEX_MOE_COMPARE_WARMUP", 5),
        iters=_env_int("DIFFULEX_MOE_COMPARE_ITERS", 20),
        check_serial=bool(_env_int("DIFFULEX_MOE_COMPARE_CHECK_SERIAL", 1)),
        dtype=torch.float16,
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
            "tp_max_abs": max(item["tp_max_abs"] for item in results),
            "serial_max_abs": (
                max(item["serial_max_abs"] for item in results)
                if results and results[0]["serial_max_abs"] is not None
                else None
            ),
            "naive_max_abs": max(item["naive_max_abs"] for item in results),
            "naive_vs_tp_max_abs": max(item["naive_vs_tp_max_abs"] for item in results),
            "tp_rel": max(item["tp_rel"] for item in results),
            "naive_rel": max(item["naive_rel"] for item in results),
            "naive_vs_tp_rel": max(item["naive_vs_tp_rel"] for item in results),
            "ref_ms_max_rank": max(item["ref_ms"] for item in results),
            "tp_ms_max_rank": max(item["tp_ms"] for item in results),
            "naive_ms_max_rank": max(item["naive_ms"] for item in results),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
