from diffulex.moe.dispatcher.datatype import CombineInput, DispatchOutput
from diffulex.moe.dispatcher.base import TokenDispatcher
from diffulex.moe.dispatcher.trivial import TrivialTokenDispatcher


def build_dispatcher(
        impl: str,
        *args,
        **kwargs,
) -> TokenDispatcher:
    if impl == "trivial":
        return TrivialTokenDispatcher(*args, **kwargs)
    raise NotImplementedError(f"Unsupported dispatcher backend: {impl!r}")


__all__ = [
    "CombineInput",
    "DispatchOutput",
    "TokenDispatcher",
    "TrivialTokenDispatcher",
    "build_dispatcher",
]
