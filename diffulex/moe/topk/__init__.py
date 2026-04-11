from diffulex.moe.topk.datatype import TopKOutput
from diffulex.moe.topk.base import TopKRouter
from diffulex.moe.topk.triton import TritonFusedTopKRouter
from diffulex.moe.topk.trivial import TrivialTopKRouter


def build_topk_router(
        impl: str,
        *args,
        **kwargs,
) -> TopKRouter:
    if impl == "trivial":
        return TrivialTopKRouter(*args, **kwargs)
    elif impl == "triton":
        return TritonFusedTopKRouter(*args, **kwargs)
    else:
        raise NotImplementedError


__all__ = [
    "build_topk_router",
    "TopKRouter",
    "TopKOutput",

    "TrivialTopKRouter",
    "TritonFusedTopKRouter",
]
