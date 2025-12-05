"""Diffulex sampler package that imports built-in samplers to trigger registration."""
from __future__ import annotations

# Import built-in samplers so their registrations run at import time.
from . import dream  # noqa: F401
from . import llada  # noqa: F401

__all__ = ["dream", "llada"]

from .auto_sampler import AutoSampler