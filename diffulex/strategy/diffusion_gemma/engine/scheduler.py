from __future__ import annotations

from diffulex.config import Config
from diffulex.engine.scheduler import AutoScheduler, SchedulerBase


@AutoScheduler.register("diffusion_gemma")
class DiffusionGemmaScheduler(SchedulerBase):
    def __init__(self, config: Config):
        super().__init__(config)
