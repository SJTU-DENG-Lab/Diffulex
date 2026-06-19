from __future__ import annotations

from diffulex.config import Config
from diffulex.engine.scheduler import AutoScheduler, SchedulerBase


@AutoScheduler.register("multi_bd", is_default=True)
class MultiBDScheduler(SchedulerBase):
    def __init__(self, config: Config):
        super().__init__(config)
