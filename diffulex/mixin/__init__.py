__all__ = [
    "EditSamplerMixin",
    "EditSchedulerMixin",
    "TokenMergeSamplerMixin",
]


def __getattr__(name):
    if name in {"EditSamplerMixin", "EditSchedulerMixin"}:
        from .edit import EditSamplerMixin, EditSchedulerMixin

        return {
            "EditSamplerMixin": EditSamplerMixin,
            "EditSchedulerMixin": EditSchedulerMixin,
        }[name]
    if name == "TokenMergeSamplerMixin":
        from .token_merge import TokenMergeSamplerMixin

        return TokenMergeSamplerMixin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
