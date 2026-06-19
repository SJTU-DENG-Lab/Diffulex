__all__ = [
    "BlockRewriteSamplerMixin",
    "BlockRewriteSchedulerMixin",
    "TokenMergeSamplerMixin",
]


def __getattr__(name):
    if name in {"BlockRewriteSamplerMixin", "BlockRewriteSchedulerMixin"}:
        from .block_rewrite import BlockRewriteSamplerMixin, BlockRewriteSchedulerMixin

        return {
            "BlockRewriteSamplerMixin": BlockRewriteSamplerMixin,
            "BlockRewriteSchedulerMixin": BlockRewriteSchedulerMixin,
        }[name]
    if name == "TokenMergeSamplerMixin":
        from .token_merge import TokenMergeSamplerMixin

        return TokenMergeSamplerMixin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
