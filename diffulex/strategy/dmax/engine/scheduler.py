from __future__ import annotations

from diffulex.config import Config
from diffulex.engine.scheduler import AutoScheduler
from diffulex.strategy.dmax.engine.request import DMaxReq
from diffulex.strategy_template.token_merging_multi_block.engine.scheduler import (
    TokenMergingMultiBlockSchedulerTemplate,
)


@AutoScheduler.register("dmax")
class DMaxScheduler(TokenMergingMultiBlockSchedulerTemplate):
    def __init__(self, config: Config):
        super().__init__(config)
        self.init_multi_block()

    def add(self, req: DMaxReq) -> None:
        req.init_token_merging_multi_block(self.config)
        self.waiting_reqs.append(req)

    def schedule(self) -> tuple[list[DMaxReq], bool]:
        return self.schedule_multi_block()

    def preempt(self, req: DMaxReq) -> None:
        self.preempt_multi_block(req)

    def postprocess(
        self,
        reqs: list[DMaxReq],
        sample_output,
    ) -> None:
        self.postprocess_token_merging_multi_block(reqs, sample_output)
