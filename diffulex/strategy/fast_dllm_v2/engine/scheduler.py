from __future__ import annotations

from diffulex.config import Config
from diffulex.engine.scheduler import AutoScheduler, SchedulerBase
from diffulex.engine.status import DllmReqStatus
from diffulex.strategy.fast_dllm_v2.engine.request import FastDLLMV2Mode


@AutoScheduler.register("fast_dllm_v2")
class FastDLLMV2Scheduler(SchedulerBase):
    def __init__(self, config: Config):
        super().__init__(config)

    @staticmethod
    def _req_mode(req) -> FastDLLMV2Mode:
        return getattr(req, "fdv2_mode", FastDLLMV2Mode.FULL_BUFFER_INIT)

    def schedule(self):
        scheduled, is_prefill = super().schedule()
        if is_prefill or len(scheduled) <= 1:
            return scheduled, is_prefill

        target_mode = self._req_mode(scheduled[0])
        kept = []
        deferred = []
        for req in scheduled:
            if self._req_mode(req) == target_mode:
                kept.append(req)
            else:
                deferred.append(req)

        if not deferred:
            return scheduled, is_prefill

        for req in deferred:
            if req in self.running_reqs:
                self.running_reqs.remove(req)
            req.status = DllmReqStatus.DECODING
            self.running_reqs.append(req)

        return kept, is_prefill
