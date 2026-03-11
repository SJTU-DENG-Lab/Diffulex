from __future__ import annotations

from diffulex.config import Config
from diffulex.engine.scheduler import AutoScheduler, SchedulerBase
from diffulex.engine.request import DllmReq


@AutoScheduler.register("multi_bd", is_default=True)
class MultiBDScheduler(SchedulerBase):
    def __init__(self, config: Config):
        super().__init__(config)
        self.init_multi_block()

    def add(self, req: DllmReq) -> None:
        self.add_multi_block(req)

    def schedule(self) -> tuple[list[DllmReq], bool]:
        return self.schedule_multi_block()

    def preempt(self, req: DllmReq) -> None:
        self.preempt_multi_block(req)

    def postprocess(
        self,
        reqs: list[DllmReq],
        sample_output,
    ) -> None:
        self.postprocess_multi_block(reqs, sample_output)
