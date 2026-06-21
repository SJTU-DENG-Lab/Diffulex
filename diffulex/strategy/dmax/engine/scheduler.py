from __future__ import annotations

from diffulex.config import Config
from diffulex.engine.scheduler import AutoScheduler
from diffulex.strategy.dmax.engine.request import DMaxReq
from diffulex.strategy.templates.token_merge.engine.scheduler import (
    TokenMergeSchedulerTemplate,
)


@AutoScheduler.register("dmax")
class DMaxScheduler(TokenMergeSchedulerTemplate):
    def __init__(self, config: Config):
        super().__init__(config)

    def add(self, req: DMaxReq) -> None:
        req.init_token_merge(self.config)
        self.waiting_reqs.append(req)

    def postprocess(
        self,
        reqs: list[DMaxReq],
        sample_output,
    ) -> None:
        self.postprocess_token_merge(reqs, sample_output)
