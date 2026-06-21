from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Iterator

import os
import torch
import torch.distributed as dist

from diffulex.config import Config
from diffulex.distributed.parallel_state import fetch_parallel_state
from diffulex.logger import get_logger


logger = get_logger(__name__)

_VLLM_TP_GROUP = None
_VLLM_TP_GROUP_FAILED = False


def _tp_group_ranks_for_vllm() -> list[list[int]]:
    state = fetch_parallel_state()
    tp_size = int(state.tp_size)
    if tp_size <= 1:
        return [[int(state.global_rank)]]
    dp_size = int(state.dp_size)
    return [
        list(range(dp_rank * tp_size, (dp_rank + 1) * tp_size))
        for dp_rank in range(dp_size)
    ]


def get_vllm_tp_group():
    """Return a vLLM GroupCoordinator matching Diffulex TP ranks.

    dInfer routes custom all-reduce and CUDA graph capture through the vLLM /
    sglang group coordinator. Diffulex owns process-group initialization, so we
    build the coordinator on top of the already initialized torch distributed
    group and fall back silently when vLLM is unavailable.
    """
    global _VLLM_TP_GROUP, _VLLM_TP_GROUP_FAILED
    if os.environ.get("DIFFULEX_DISABLE_VLLM_TP_GROUP", "0") == "1":
        return None
    if _VLLM_TP_GROUP is not None:
        return _VLLM_TP_GROUP
    if _VLLM_TP_GROUP_FAILED:
        return None

    try:
        from vllm.distributed.parallel_state import GroupCoordinator, set_custom_all_reduce

        set_custom_all_reduce(True)
        _VLLM_TP_GROUP = GroupCoordinator(
            group_ranks=_tp_group_ranks_for_vllm(),
            local_rank=int(torch.cuda.current_device()),
            torch_distributed_backend=dist.get_backend(),
            use_device_communicator=True,
            group_name="tp",
        )
        return _VLLM_TP_GROUP
    except Exception:
        _VLLM_TP_GROUP_FAILED = True
        logger.debug("Failed to initialize vLLM TP coordinator.", exc_info=True)
        return None


@contextmanager
def vllm_current_config(config: Config) -> Iterator[None]:
    """Temporarily install a minimal vLLM config during module construction."""
    try:
        from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

        parallel_config = ParallelConfig(
            tensor_parallel_size=int(config.tensor_parallel_size),
            pipeline_parallel_size=1,
            data_parallel_size=int(config.data_parallel_size),
            enable_expert_parallel=bool(int(config.expert_parallel_size) > 1),
            disable_custom_all_reduce=False,
            distributed_timeout_seconds=int(config.distributed_timeout_seconds),
        )
    except Exception:
        logger.debug("Using Diffulex model init without vLLM current config.", exc_info=True)
        yield
        return

    with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
        yield


@contextmanager
def vllm_graph_capture(stream: torch.cuda.Stream, pool) -> Iterator[torch.cuda.Stream]:
    """Enter vLLM graph-capture side contexts when available."""
    try:
        from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id

        set_graph_pool_id(pool)
    except Exception:
        logger.debug("Failed to set vLLM graph pool id.", exc_info=True)

    group = get_vllm_tp_group()
    if group is None or not hasattr(group, "graph_capture"):
        with torch.cuda.stream(stream):
            yield stream
        return

    try:
        context = getattr(group, "graph_capture")()
        with context as graph_context:
            yield getattr(graph_context, "stream", stream)
    except Exception:
        logger.debug("vLLM graph_capture context failed; using raw CUDA stream.", exc_info=True)
        with torch.cuda.stream(stream):
            yield stream


def reset_vllm_compat_state() -> None:
    global _VLLM_TP_GROUP, _VLLM_TP_GROUP_FAILED
    group = _VLLM_TP_GROUP
    _VLLM_TP_GROUP = None
    _VLLM_TP_GROUP_FAILED = False
    if group is not None and hasattr(group, "destroy"):
        try:
            group.destroy()
        except Exception:
            logger.debug("Failed to destroy vLLM TP coordinator.", exc_info=True)
