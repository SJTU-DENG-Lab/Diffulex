from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist

try:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.distributed import parallel_state as sglang_parallel_state
except Exception:
    GroupCoordinator = None
    sglang_parallel_state = None


SGLANG_TP_GROUP_COORDINATOR = None
SGLANG_SINGLETON_GROUP_COORDINATOR = None


def _current_cuda_rank() -> int:
    if torch.cuda.is_available():
        return int(torch.cuda.current_device())
    return 0


class ParallelStateSGLangMixin:
    def get_sglang_tp_group_ranks(self) -> list[list[int]]:
        return [list(self.base_model.tp_ranks)]

    def can_use_sglang_tp(self) -> bool:
        return (
            GroupCoordinator is not None
            and dist.is_available()
            and dist.is_initialized()
            and self.get_tp_world_size() > 1
        )

    def get_sglang_tp_group(self):
        global SGLANG_TP_GROUP_COORDINATOR

        if not self.can_use_sglang_tp():
            return None

        coordinator = SGLANG_TP_GROUP_COORDINATOR
        if coordinator is not None:
            return coordinator

        coordinator = GroupCoordinator(
            group_ranks=self.get_sglang_tp_group_ranks(),
            local_rank=_current_cuda_rank(),
            torch_distributed_backend=dist.get_backend(),
            use_pynccl=False,
            use_pymscclpp=False,
            use_custom_allreduce=True,
            use_torch_symm_mem_all_reduce=False,
            use_hpu_communicator=False,
            use_xpu_communicator=False,
            use_npu_communicator=False,
            group_name="diffulex_tp",
        )
        SGLANG_TP_GROUP_COORDINATOR = coordinator
        return coordinator

    @contextmanager
    def sglang_graph_capture(self, stream: torch.cuda.Stream | None = None):
        coordinator = self.get_sglang_tp_group()
        if coordinator is None:
            with nullcontext():
                yield None
            return
        with coordinator.graph_capture(stream=stream) as capture_context:
            yield capture_context


def reset_sglang_backend_state() -> None:
    global SGLANG_SINGLETON_GROUP_COORDINATOR, SGLANG_TP_GROUP_COORDINATOR

    coordinator = SGLANG_TP_GROUP_COORDINATOR
    SGLANG_TP_GROUP_COORDINATOR = None
    if coordinator is not None:
        try:
            coordinator.destroy()
        except Exception:
            pass

    singleton = SGLANG_SINGLETON_GROUP_COORDINATOR
    SGLANG_SINGLETON_GROUP_COORDINATOR = None
    if singleton is not None:
        try:
            singleton.destroy()
        except Exception:
            pass

    if sglang_parallel_state is not None:
        sglang_parallel_state._TP = None
        sglang_parallel_state._MOE_TP = None
        sglang_parallel_state._MOE_EP = None
        sglang_parallel_state._MOE_DP = None


class _LocalGroupCoordinator:
    def __init__(self, *, rank_in_group: int = 0, world_size: int = 1):
        self.rank_in_group = rank_in_group
        self.world_size = world_size
        self.device_group = None

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def gather(self, x: torch.Tensor, dst: int = 0, dim: int = -1) -> torch.Tensor | None:
        return x if dst == 0 else None

    @contextmanager
    def graph_capture(self, stream: torch.cuda.Stream | None = None):
        with nullcontext():
            yield None

    def destroy(self) -> None:
        return None


def _build_singleton_group(rank: int):
    global SGLANG_SINGLETON_GROUP_COORDINATOR

    coordinator = SGLANG_SINGLETON_GROUP_COORDINATOR
    if coordinator is not None:
        return coordinator

    if GroupCoordinator is not None and dist.is_available() and dist.is_initialized():
        coordinator = GroupCoordinator(
            group_ranks=[[dist.get_rank()]],
            local_rank=_current_cuda_rank(),
            torch_distributed_backend=dist.get_backend(),
            use_pynccl=False,
            use_pymscclpp=False,
            use_custom_allreduce=False,
            use_torch_symm_mem_all_reduce=False,
            use_hpu_communicator=False,
            use_xpu_communicator=False,
            use_npu_communicator=False,
            group_name=f"diffulex_singleton_{rank}",
        )
    else:
        coordinator = _LocalGroupCoordinator(rank_in_group=0, world_size=1)

    SGLANG_SINGLETON_GROUP_COORDINATOR = coordinator
    return coordinator


def ensure_sglang_moe_parallel_state(parallel_state) -> None:
    if sglang_parallel_state is None:
        return

    tp_group = parallel_state.get_sglang_tp_group()
    if tp_group is None:
        tp_group = _LocalGroupCoordinator(
            rank_in_group=parallel_state.get_tp_rank(),
            world_size=parallel_state.get_tp_world_size(),
        )

    singleton_group = _build_singleton_group(parallel_state.global_rank)
    sglang_parallel_state._TP = tp_group
    sglang_parallel_state._MOE_EP = tp_group
    sglang_parallel_state._MOE_TP = singleton_group
    sglang_parallel_state._MOE_DP = singleton_group
