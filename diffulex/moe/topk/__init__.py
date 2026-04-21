from diffulex.moe.topk.output import TopKOutput
from diffulex.moe.topk.base import TopKRouter
from diffulex.moe.topk.naive import NaiveTopKRouter
from diffulex.moe.topk.group_limited import GroupLimitedTopKRouter


def build_topk_router(
        impl: str,
        *args,
        **kwargs,
) -> TopKRouter:
    if impl in {"naive", "triton"}:
        return NaiveTopKRouter(*args, **kwargs)
    elif impl == "group_limited":
        return GroupLimitedTopKRouter(*args, **kwargs)
    else:
        raise NotImplementedError(f"Unsupported topk router impl: {impl!r}.")


__all__ = [
    "build_topk_router",
    "TopKRouter",
    "TopKOutput",
    "NaiveTopKRouter",
    "GroupLimitedTopKRouter",
]
