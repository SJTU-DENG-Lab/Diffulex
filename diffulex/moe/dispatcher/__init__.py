from diffulex.moe.dispatcher.a2a import A2ADispatchContext, A2ATokenDispatcher
from diffulex.moe.dispatcher.datatype import CombineInput, DispatchOutput
from diffulex.moe.dispatcher.base import TokenDispatcher
from diffulex.moe.dispatcher.trivial import TrivialTokenDispatcher

TrivialMoEDispatcher = TrivialTokenDispatcher


def build_dispatcher(
        impl: str,
        *args,
        **kwargs,
) -> TokenDispatcher:
    if impl == "a2a":
        return A2ATokenDispatcher(*args, **kwargs)
    if impl == "trivial":
        return TrivialTokenDispatcher(*args, **kwargs)
    raise NotImplementedError(f"Unsupported dispatcher backend: {impl!r}")


__all__ = [
    "A2ADispatchContext",
    "A2ATokenDispatcher",
    "CombineInput",
    "DispatchOutput",
    "TokenDispatcher",
    "TrivialTokenDispatcher",
    "build_dispatcher",
]
