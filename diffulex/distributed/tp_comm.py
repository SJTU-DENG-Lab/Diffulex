from __future__ import annotations

import torch
import torch.distributed as dist

from diffulex.distributed.parallel_state import fetch_parallel_state


def tp_all_reduce(x: torch.Tensor, group) -> torch.Tensor:
    try:
        sglang_tp_group = fetch_parallel_state().get_sglang_tp_group()
    except RuntimeError:
        sglang_tp_group = None
    if sglang_tp_group is not None:
        return sglang_tp_group.all_reduce(x)
    dist.all_reduce(x, group=group)
    return x


def tp_gather_to_rank0(x: torch.Tensor, group, tp_size: int, tp_rank: int) -> torch.Tensor | None:
    try:
        sglang_tp_group = fetch_parallel_state().get_sglang_tp_group()
    except RuntimeError:
        sglang_tp_group = None
    if sglang_tp_group is not None:
        return sglang_tp_group.gather(x, dst=0, dim=-1)
    gathered = [torch.empty_like(x) for _ in range(tp_size)]
    dist.all_gather(gathered, x, group=group)
    return torch.cat(gathered, dim=-1) if tp_rank == 0 else None
