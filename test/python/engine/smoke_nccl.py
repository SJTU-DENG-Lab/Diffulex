from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    x = torch.ones(1, device=f"cuda:{local_rank}") * (rank + 1)
    dist.all_reduce(x)
    torch.cuda.synchronize()
    if rank == 0:
        print(f"all_reduce={x.item()}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
