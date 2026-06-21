from __future__ import annotations

from diffulex.config import Config
from diffulex.engine.scheduler import AutoScheduler, SchedulerBase


@AutoScheduler.register("d2f", is_default=True)
class D2fScheduler(SchedulerBase):
    def __init__(self, config: Config):
        super().__init__(config)
