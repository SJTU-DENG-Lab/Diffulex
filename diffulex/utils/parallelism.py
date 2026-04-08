from __future__ import annotations

from dataclasses import dataclass

import torch.distributed as dist


@dataclass(frozen=True)
class ModelParallelismMetadata:
    """
    for example, if tp is off and ep is on, world size != tp_size, and linears should not shard weights
    so they need to ref this class to determine whether to shard weights or not

    Examples:
    - pure EP:
        tp_size = 1
        ep_size = world_size
        tp_rank = 0
        ep_rank = global_rank
    - tp == ep:
        tp_size = ep_size = world_size
        tp_rank = global_rank
        ep_rank = global_rank
    - TP only:
        ep_size = 1
        tp_size = world_size
        tp_rank = global_rank
        ep_rank = 0
    """

    tp_size: int
    ep_size: int
    world_size: int
    global_rank: int
    tp_rank: int
    ep_rank: int

    @property
    def is_pure_ep(self) -> bool:
        return self.tp_size == 1 and self.ep_size > 1

    @property
    def is_tp_eq_ep(self) -> bool:
        return self.tp_size > 1 and self.tp_size == self.ep_size

    @property
    def is_tp_only(self) -> bool:
        return self.tp_size > 1 and self.ep_size == 1

    @classmethod
    def from_world(
        cls,
        tp_size: int,
        ep_size: int,
        world_size: int,
        global_rank: int,
    ) -> "ModelParallelismMetadata":
        if tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {tp_size}.")
        if ep_size < 1:
            raise ValueError(f"ep_size must be >= 1, got {ep_size}.")
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}.")
        if not (0 <= global_rank < world_size):
            raise ValueError(
                f"global_rank must be in [0, world_size), got global_rank={global_rank}, world_size={world_size}."
            )

        if tp_size == 1 and ep_size == 1:
            if world_size != 1:
                raise ValueError(
                    "Single-device layout requires world_size == 1, "
                    f"got tp_size={tp_size}, ep_size={ep_size}, world_size={world_size}."
                )
            return cls(tp_size, ep_size, world_size, global_rank, 0, 0)

        if tp_size == 1:
            if ep_size != world_size:
                raise ValueError(
                    "Pure EP layout requires ep_size == world_size when tp_size == 1, "
                    f"got tp_size={tp_size}, ep_size={ep_size}, world_size={world_size}."
                )
            return cls(tp_size, ep_size, world_size, global_rank, 0, global_rank)

        if ep_size == 1:
            if tp_size != world_size:
                raise ValueError(
                    "TP-only layout requires tp_size == world_size when ep_size == 1, "
                    f"got tp_size={tp_size}, ep_size={ep_size}, world_size={world_size}."
                )
            return cls(tp_size, ep_size, world_size, global_rank, global_rank, 0)

        if tp_size == ep_size:
            if world_size != tp_size:
                raise ValueError(
                    "tp == ep layout requires world_size == tp_size == ep_size, "
                    f"got tp_size={tp_size}, ep_size={ep_size}, world_size={world_size}."
                )
            return cls(tp_size, ep_size, world_size, global_rank, global_rank, global_rank)

        raise NotImplementedError(
            "Only TP-only, pure EP, and tp == ep layouts are supported right now, "
            f"got tp_size={tp_size}, ep_size={ep_size}, world_size={world_size}."
        )


_MODEL_PARALLELISM_METADATA: ModelParallelismMetadata | None = None


def get_world_size(tp_size: int, ep_size: int) -> int:
    if tp_size < 1:
        raise ValueError(f"tp_size must be >= 1, got {tp_size}.")
    if ep_size < 1:
        raise ValueError(f"ep_size must be >= 1, got {ep_size}.")

    if tp_size == 1:
        return ep_size
    if ep_size == 1:
        return tp_size
    if tp_size == ep_size:
        return tp_size

    raise NotImplementedError(
        "Only TP-only, pure EP, and tp == ep layouts are supported right now, "
        f"got tp_size={tp_size}, ep_size={ep_size}."
    )


def init_model_parallelism_metadata(tp_size: int, ep_size: int) -> ModelParallelismMetadata:
    global _MODEL_PARALLELISM_METADATA

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed process group has not been initialized.")

    layout = ModelParallelismMetadata.from_world(
        tp_size=tp_size,
        ep_size=ep_size,
        world_size=dist.get_world_size(),
        global_rank=dist.get_rank(),
    )
    if _MODEL_PARALLELISM_METADATA is not None and _MODEL_PARALLELISM_METADATA != layout:
        raise RuntimeError(
            "Model parallelism layout has already been initialized in this process, "
            f"existing={_MODEL_PARALLELISM_METADATA}, new={layout}."
        )
    _MODEL_PARALLELISM_METADATA = layout
    return layout


def reset_model_parallelism_metadata() -> None:
    global _MODEL_PARALLELISM_METADATA
    _MODEL_PARALLELISM_METADATA = None


def get_model_parallelism_metadata() -> ModelParallelismMetadata:
    if _MODEL_PARALLELISM_METADATA is None:
        raise RuntimeError("Model parallelism layout has not been initialized for this process.")
    return _MODEL_PARALLELISM_METADATA


def get_tp_rank() -> int:
    return get_model_parallelism_metadata().tp_rank


def get_tp_world_size() -> int:
    return get_model_parallelism_metadata().tp_size


def is_tp_enabled() -> bool:
    return get_model_parallelism_metadata().tp_size > 1


def get_ep_rank() -> int:
    return get_model_parallelism_metadata().ep_rank


def get_ep_world_size() -> int:
    return get_model_parallelism_metadata().ep_size


def is_ep_enabled() -> bool:
    return get_model_parallelism_metadata().ep_size > 1


def init_process_group(
    *,
    tp_size: int,
    ep_size: int,
    rank: int,
    init_method: str,
    device_id: int | None = None,
    backend: str = "nccl",
) -> None:
    world_size = get_world_size(tp_size, ep_size)

    if dist.is_available() and dist.is_initialized():
        current_world_size = dist.get_world_size()
        current_rank = dist.get_rank()
        if current_world_size != world_size or current_rank != rank:
            raise RuntimeError(
                "torch.distributed process group is already initialized with a different layout, "
                f"existing world_size={current_world_size}, rank={current_rank}; "
                f"requested world_size={world_size}, rank={rank}."
            )
    else:
        kwargs = {
            "backend": backend,
            "init_method": init_method,
            "world_size": world_size,
            "rank": rank,
        }
        if device_id is not None:
            kwargs["device_id"] = device_id
        dist.init_process_group(**kwargs)
