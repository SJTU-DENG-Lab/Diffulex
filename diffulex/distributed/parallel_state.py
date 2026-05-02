from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import torch.distributed as dist

from diffulex.distributed.sglang_backend import ParallelStateSGLangMixin, reset_sglang_backend_state


@dataclass(frozen=True)
class WorldMesh:
    world_size: int
    global_rank: int
    local_rank: int
    node_rank: int
    num_nodes: int


@dataclass(frozen=True)
class BaseModelParallelLayout:
    model_parallel_size: int
    model_parallel_rank: int
    model_parallel_ranks: tuple[int, ...]
    model_parallel_group: dist.ProcessGroup | None

    tp_size: int
    tp_rank: int
    tp_ranks: tuple[int, ...]
    tp_group: dist.ProcessGroup | None

    dp_size: int = 1
    dp_rank: int = 0
    dp_ranks: tuple[int, ...] = (0,)
    dp_group: dist.ProcessGroup | None = None

    sp_size: int = 1
    sp_rank: int = 0
    sp_ranks: tuple[int, ...] = (0,)
    sp_group: dist.ProcessGroup | None = None


@dataclass(frozen=True)
class MoEParallelLayout:
    ep_size: int
    ep_rank: int
    ep_ranks: tuple[int, ...]
    ep_group: dist.ProcessGroup | None


@dataclass(frozen=True)
class ParallelState(ParallelStateSGLangMixin):
    world: WorldMesh
    base_model: BaseModelParallelLayout
    moe: MoEParallelLayout | None
    topology: str

    @property
    def world_size(self) -> int:
        return self.world.world_size

    @property
    def global_rank(self) -> int:
        return self.world.global_rank

    @property
    def model_parallel_size(self) -> int:
        return self.base_model.model_parallel_size

    @property
    def model_parallel_rank(self) -> int:
        return self.base_model.model_parallel_rank

    @property
    def tp_size(self) -> int:
        return self.base_model.tp_size

    @property
    def tp_rank(self) -> int:
        return self.base_model.tp_rank

    @property
    def dp_size(self) -> int:
        return self.base_model.dp_size

    @property
    def dp_rank(self) -> int:
        return self.base_model.dp_rank

    @property
    def ep_size(self) -> int:
        return 1 if self.moe is None else self.moe.ep_size

    @property
    def ep_rank(self) -> int:
        return 0 if self.moe is None else self.moe.ep_rank

    @property
    def has_moe(self) -> bool:
        return self.moe is not None and self.moe.ep_size > 1

    @property
    def is_cross_dp_ep(self) -> bool:
        return self.has_moe and self.dp_size > 1 and self.ep_size > self.tp_size

    def get_model_parallel_rank(self) -> int:
        return self.base_model.model_parallel_rank

    def get_model_parallel_world_size(self) -> int:
        return self.base_model.model_parallel_size

    def get_model_parallel_group(self) -> dist.ProcessGroup | None:
        return self.base_model.model_parallel_group

    def get_base_model_tp_rank(self) -> int:
        return self.base_model.tp_rank

    def get_base_model_tp_world_size(self) -> int:
        return self.base_model.tp_size

    def get_base_model_tp_group(self) -> dist.ProcessGroup | None:
        return self.base_model.tp_group

    def get_tp_rank(self) -> int:
        return self.get_base_model_tp_rank()

    def get_tp_world_size(self) -> int:
        return self.get_base_model_tp_world_size()

    def get_tp_group(self) -> dist.ProcessGroup | None:
        return self.get_base_model_tp_group()

    def get_dp_rank(self) -> int:
        return self.base_model.dp_rank

    def get_dp_world_size(self) -> int:
        return self.base_model.dp_size

    def get_dp_group(self) -> dist.ProcessGroup | None:
        return self.base_model.dp_group

    def get_moe_ep_rank(self) -> int:
        return 0 if self.moe is None else self.moe.ep_rank

    def get_moe_ep_world_size(self) -> int:
        return 1 if self.moe is None else self.moe.ep_size

    def get_moe_ep_group(self) -> dist.ProcessGroup | None:
        return None if self.moe is None else self.moe.ep_group

    def get_ep_rank(self) -> int:
        return self.get_moe_ep_rank()

    def get_ep_world_size(self) -> int:
        return self.get_moe_ep_world_size()

    def get_ep_group(self) -> dist.ProcessGroup | None:
        return self.get_moe_ep_group()

    def is_tp_enabled(self) -> bool:
        return self.tp_size > 1

    def is_dp_enabled(self) -> bool:
        return self.dp_size > 1

    def is_ep_enabled(self) -> bool:
        return self.ep_size > 1

    def is_cross_dp_ep_enabled(self) -> bool:
        return self.is_cross_dp_ep


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")


def _resolve_topology(tp_size: int, ep_size: int, dp_size: int = 1, sp_size: int = 1) -> tuple[str, int]:
    _validate_positive("tp_size", tp_size)
    _validate_positive("ep_size", ep_size)
    _validate_positive("dp_size", dp_size)
    _validate_positive("sp_size", sp_size)

    if sp_size != 1:
        raise NotImplementedError(f"Sequence parallel is not wired yet, got sp_size={sp_size}.")

    base_world_size = tp_size * dp_size

    if ep_size == 1:
        return "base_model_only", base_world_size
    if tp_size == 1 and dp_size == 1:
        return "pure_ep", ep_size
    if ep_size == tp_size:
        return "ep_per_dp_shard", base_world_size
    if ep_size == base_world_size:
        return "global_ep", base_world_size

    raise NotImplementedError(
        "Unsupported tp/dp/ep topology. Supported layouts are: "
        "(1) base-model only, ep_size == 1; "
        "(2) pure EP, tp_size == dp_size == 1; "
        "(3) per-DP-shard EP, ep_size == tp_size; "
        "(4) global EP, ep_size == tp_size * dp_size. "
        f"got tp_size={tp_size}, ep_size={ep_size}, dp_size={dp_size}, sp_size={sp_size}."
    )


def get_world_size(tp_size: int, ep_size: int, dp_size: int = 1, sp_size: int = 1) -> int:
    _topology, world_size = _resolve_topology(tp_size, ep_size, dp_size, sp_size)
    return world_size


def init_process_group(
    *,
    tp_size: int,
    ep_size: int,
    rank: int,
    init_method: str,
    device_id: int,
    backend: str,
    dp_size: int = 1,
    sp_size: int = 1,
    timeout_seconds: int = 600,
) -> None:
    kwargs = dict(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=get_world_size(tp_size=tp_size, ep_size=ep_size, dp_size=dp_size, sp_size=sp_size),
        timeout=timedelta(seconds=timeout_seconds),
    )
    try:
        dist.init_process_group(device_id=device_id, **kwargs)
    except TypeError:
        dist.init_process_group(**kwargs)


def _build_group(
    *,
    group_ranks_list: list[tuple[int, ...]],
    global_rank: int,
    backend: str,
) -> tuple[dist.ProcessGroup | None, tuple[int, ...], int]:
    local_group: dist.ProcessGroup | None = None
    local_ranks: tuple[int, ...] = (global_rank,)
    local_rank_in_group = 0

    for ranks in group_ranks_list:
        group = dist.new_group(ranks=list(ranks), backend=backend)
        if global_rank in ranks:
            local_group = group if len(ranks) > 1 else None
            local_ranks = ranks
            local_rank_in_group = ranks.index(global_rank)

    return local_group, local_ranks, local_rank_in_group


def _select_group_for_rank(
    *,
    group_ranks_list: list[tuple[int, ...]],
    global_rank: int,
) -> tuple[tuple[int, ...], int]:
    for ranks in group_ranks_list:
        if global_rank in ranks:
            return ranks, ranks.index(global_rank)
    return (global_rank,), 0


def _compute_base_model_groups(
    topology: str,
    *,
    tp_size: int,
    dp_size: int,
    world_size: int,
) -> list[tuple[int, ...]]:
    if topology == "pure_ep":
        return [tuple(range(world_size))]
    return [
        tuple(range(dp_rank * tp_size, (dp_rank + 1) * tp_size))
        for dp_rank in range(dp_size)
    ]


def _compute_dp_groups(base_model_groups: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    tp_size = len(base_model_groups[0])
    return [
        tuple(group[tp_rank] for group in base_model_groups)
        for tp_rank in range(tp_size)
    ]


def _compute_tp_groups(
    topology: str,
    tp_size: int,
    base_model_groups: list[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    if topology == "pure_ep" or tp_size == 1:
        return [tuple([rank]) for group in base_model_groups for rank in group]
    return base_model_groups


def _compute_ep_groups(
    topology: str,
    ep_size: int,
    base_model_groups: list[tuple[int, ...]],
    world_size: int,
) -> list[tuple[int, ...]]:
    if ep_size == 1:
        return [tuple([rank]) for group in base_model_groups for rank in group]
    if topology in {"pure_ep", "global_ep"}:
        return [tuple(range(world_size))]
    if topology == "ep_per_dp_shard":
        return base_model_groups
    raise AssertionError(f"Unsupported topology for ep groups: {topology}")


def _compute_sp_groups(base_model_groups: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    return [tuple([rank]) for group in base_model_groups for rank in group]


def _build_parallel_state(
    *,
    tp_size: int,
    ep_size: int,
    dp_size: int,
    sp_size: int,
    world_size: int,
    global_rank: int,
    backend: str,
) -> ParallelState:
    expected_world_size = get_world_size(
        tp_size=tp_size,
        ep_size=ep_size,
        dp_size=dp_size,
        sp_size=sp_size,
    )
    if world_size != expected_world_size:
        raise ValueError(
            "Distributed world size does not match the requested topology, "
            f"got world_size={world_size}, expected={expected_world_size}, "
            f"tp_size={tp_size}, ep_size={ep_size}, dp_size={dp_size}, sp_size={sp_size}."
        )
    if not (0 <= global_rank < world_size):
        raise ValueError(
            f"global_rank must be in [0, world_size), got global_rank={global_rank}, world_size={world_size}."
        )

    topology, _ = _resolve_topology(tp_size, ep_size, dp_size, sp_size)
    base_model_groups = _compute_base_model_groups(
        topology,
        tp_size=tp_size,
        dp_size=dp_size,
        world_size=world_size,
    )
    model_parallel_size = len(base_model_groups[0])
    model_parallel_group, model_parallel_ranks, model_parallel_rank = _build_group(
        group_ranks_list=base_model_groups,
        global_rank=global_rank,
        backend=backend,
    )

    tp_group, tp_ranks, tp_rank = _build_group(
        group_ranks_list=_compute_tp_groups(topology, tp_size, base_model_groups),
        global_rank=global_rank,
        backend=backend,
    )
    dp_group, dp_ranks, dp_rank = _build_group(
        group_ranks_list=_compute_dp_groups(base_model_groups),
        global_rank=global_rank,
        backend=backend,
    )
    sp_group, sp_ranks, sp_rank = _build_group(
        group_ranks_list=_compute_sp_groups(base_model_groups),
        global_rank=global_rank,
        backend=backend,
    )

    base_model = BaseModelParallelLayout(
        model_parallel_size=model_parallel_size,
        model_parallel_rank=model_parallel_rank,
        model_parallel_ranks=model_parallel_ranks,
        model_parallel_group=model_parallel_group,
        tp_size=tp_size,
        tp_rank=tp_rank,
        tp_ranks=tp_ranks,
        tp_group=tp_group,
        dp_size=dp_size,
        dp_rank=dp_rank,
        dp_ranks=dp_ranks,
        dp_group=dp_group,
        sp_size=sp_size,
        sp_rank=sp_rank,
        sp_ranks=sp_ranks,
        sp_group=sp_group,
    )

    moe: MoEParallelLayout | None = None
    if ep_size > 1:
        ep_group, ep_ranks, ep_rank = _build_group(
            group_ranks_list=_compute_ep_groups(topology, ep_size, base_model_groups, world_size),
            global_rank=global_rank,
            backend=backend,
        )
        moe = MoEParallelLayout(
            ep_size=ep_size,
            ep_rank=ep_rank,
            ep_ranks=ep_ranks,
            ep_group=ep_group,
        )

    return ParallelState(
        world=WorldMesh(
            world_size=world_size,
            global_rank=global_rank,
            local_rank=global_rank,
            node_rank=0,
            num_nodes=1,
        ),
        base_model=base_model,
        moe=moe,
        topology=topology,
    )


PARALLEL_STATE: ParallelState | None = None


def _state_matches_request(
    state: ParallelState,
    *,
    tp_size: int,
    ep_size: int,
    dp_size: int,
    sp_size: int,
) -> bool:
    return (
        state.base_model.tp_size == tp_size
        and state.ep_size == ep_size
        and state.base_model.dp_size == dp_size
        and state.base_model.sp_size == sp_size
    )


def init_parallel_state(
    *,
    tp_size: int,
    ep_size: int,
    dp_size: int = 1,
    sp_size: int = 1,
) -> ParallelState:
    global PARALLEL_STATE

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed process group has not been initialized.")

    current_state = PARALLEL_STATE
    if current_state is not None:
        if not _state_matches_request(
            current_state,
            tp_size=tp_size,
            ep_size=ep_size,
            dp_size=dp_size,
            sp_size=sp_size,
        ):
            raise RuntimeError(
                "Parallel state has already been initialized in this process, "
                f"existing={current_state}, requested="
                f"(tp_size={tp_size}, ep_size={ep_size}, dp_size={dp_size}, sp_size={sp_size})."
            )
        PARALLEL_STATE = current_state
        return current_state

    state = _build_parallel_state(
        tp_size=tp_size,
        ep_size=ep_size,
        dp_size=dp_size,
        sp_size=sp_size,
        world_size=dist.get_world_size(),
        global_rank=dist.get_rank(),
        backend=dist.get_backend(),
    )
    PARALLEL_STATE = state
    return state


def reset_parallel_state() -> None:
    global PARALLEL_STATE
    reset_sglang_backend_state()
    PARALLEL_STATE = None


def fetch_parallel_state() -> ParallelState:
    current_state = PARALLEL_STATE
    if current_state is None:
        raise RuntimeError("Parallel state has not been initialized for this process.")
    return current_state


def build_parallel_state_for_test(
    *,
    tp_size: int,
    ep_size: int,
    dp_size: int = 1,
    sp_size: int = 1,
    global_rank: int = 0,
    world_size: int | None = None,
) -> ParallelState:
    topology, expected_world_size = _resolve_topology(tp_size, ep_size, dp_size, sp_size)
    world_size = expected_world_size if world_size is None else world_size
    if world_size != expected_world_size:
        raise ValueError(
            "Test parallel state world_size mismatch, "
            f"got world_size={world_size}, expected={expected_world_size}."
        )
    if not (0 <= global_rank < world_size):
        raise ValueError(
            f"global_rank must be in [0, world_size), got global_rank={global_rank}, world_size={world_size}."
        )

    base_model_groups = _compute_base_model_groups(
        topology,
        tp_size=tp_size,
        dp_size=dp_size,
        world_size=world_size,
    )
    model_parallel_ranks, model_parallel_rank = _select_group_for_rank(
        group_ranks_list=base_model_groups,
        global_rank=global_rank,
    )
    tp_ranks, tp_rank = _select_group_for_rank(
        group_ranks_list=_compute_tp_groups(topology, tp_size, base_model_groups),
        global_rank=global_rank,
    )
    dp_ranks, dp_rank = _select_group_for_rank(
        group_ranks_list=_compute_dp_groups(base_model_groups),
        global_rank=global_rank,
    )
    sp_ranks, sp_rank = _select_group_for_rank(
        group_ranks_list=_compute_sp_groups(base_model_groups),
        global_rank=global_rank,
    )

    moe: MoEParallelLayout | None = None
    if ep_size > 1:
        ep_ranks, ep_rank = _select_group_for_rank(
            group_ranks_list=_compute_ep_groups(topology, ep_size, base_model_groups, world_size),
            global_rank=global_rank,
        )
        moe = MoEParallelLayout(
            ep_size=ep_size,
            ep_rank=ep_rank,
            ep_ranks=ep_ranks,
            ep_group=None,
        )

    return ParallelState(
        world=WorldMesh(
            world_size=world_size,
            global_rank=global_rank,
            local_rank=global_rank,
            node_rank=0,
            num_nodes=1,
        ),
        base_model=BaseModelParallelLayout(
            model_parallel_size=len(model_parallel_ranks),
            model_parallel_rank=model_parallel_rank,
            model_parallel_ranks=model_parallel_ranks,
            model_parallel_group=None,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_ranks=tp_ranks,
            tp_group=None,
            dp_size=dp_size,
            dp_rank=dp_rank,
            dp_ranks=dp_ranks,
            dp_group=None,
            sp_size=sp_size,
            sp_rank=sp_rank,
            sp_ranks=sp_ranks,
            sp_group=None,
        ),
        moe=moe,
        topology=topology,
    )


ModelParallelismMetadata = ParallelState


__all__ = [
    "BaseModelParallelLayout",
    "ModelParallelismMetadata",
    "MoEParallelLayout",
    "ParallelState",
    "WorldMesh",
    "build_parallel_state_for_test",
    "fetch_parallel_state",
    "get_world_size",
    "init_parallel_state",
    "init_process_group",
    "reset_parallel_state",
]
