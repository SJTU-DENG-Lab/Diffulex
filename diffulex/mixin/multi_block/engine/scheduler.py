from __future__ import annotations

from typing import TYPE_CHECKING

from diffulex.engine.request import DllmReq, DllmReqStatus

if TYPE_CHECKING:
    from diffulex.engine.scheduler import SchedulerBase


class SchedulerMultiBlockMixin:
    def init_multi_block(self: SchedulerBase) -> None:
        self.block_size = self.config.block_size

    def add_multi_block(self: SchedulerBase, req: DllmReq) -> None:
        req.init_multi_block(self.config)
        self.waiting_reqs.append(req)

    def schedule_multi_block(self: SchedulerBase) -> tuple[list[DllmReq], bool]:
        scheduled: list[DllmReq] = []
        num_reqs = 0
        num_batched_tokens = 0

        while self.waiting_reqs and num_reqs < self.max_num_reqs:
            req = self.waiting_reqs[0]
            
            projected = len(req) + self.block_size
            if (
                num_batched_tokens + projected > self.max_num_batched_tokens 
                or not self.kv_cache_manager.can_allocate(req)
            ):
                break
            
            num_reqs += 1
            self.kv_cache_manager.allocate(req)
            if req.is_preempted:
                self.kv_cache_manager.may_append(req)
            
            num_batched_tokens += projected - req.num_cached_tokens
            req.make_pending()
            self.waiting_reqs.popleft()
            self.running_reqs.append(req)
            scheduled.append(req)
            
        if scheduled:
            return scheduled, True

        while self.running_reqs and num_reqs < self.max_num_reqs:
            req = self.running_reqs.popleft()
            while not self.kv_cache_manager.can_append(req):
                if self.running_reqs:
                    self.preempt(self.running_reqs.pop())
                else:
                    self.preempt(req)
                    break
            else:
                num_reqs += 1
                self.kv_cache_manager.may_append(req)
                scheduled.append(req)

        if not scheduled:
            diag = dict(
                phase="decode",
                waiting=len(self.waiting_reqs),
                running=len(self.running_reqs),
                max_num_reqs=self.max_num_reqs,
                max_num_batched_tokens=self.max_num_batched_tokens,
                block_size=self.block_size,
            )
            candidates = list(self.running_reqs)[:3] + list(self.waiting_reqs)[:2]
            details = []
            for idx, candidate in enumerate(candidates):
                try:
                    can_append = self.kv_cache_manager.can_append(candidate)
                except Exception:
                    can_append = "error"
                details.append(
                    f"[{idx}] status={candidate.status.name}, len={len(candidate)}, "
                    f"block_size={self.block_size}, "
                    f"new_tokens={getattr(candidate, 'new_tokens', '?')}, "
                    f"cached={getattr(candidate, 'num_cached_tokens', '?')}, "
                    f"can_append={can_append}"
                )
            raise RuntimeError(
                "MultiBlockScheduler: unable to schedule any req in decode; "
                f"state={diag}; details={' | '.join(details)}"
            )
            
        self.running_reqs.extendleft(reversed(scheduled))
        return scheduled, False

    def preempt_multi_block(self, req: DllmReq) -> None:
        req.preempt()
        self.kv_cache_manager.free(req)
        self.waiting_reqs.appendleft(req)

    def postprocess_multi_block(
        self,
        reqs: list[DllmReq],
        sample_output,
    ) -> None:
        for req in reqs:
            req.reset_new_tokens()

            req_id_str = str(req.req_id)
            true_ids_map = sample_output.true_local_ids_map.get(req_id_str, {})
            accepted_ids_map = sample_output.accepted_ids_map.get(req_id_str, {})
            sampled_tokens_map = sample_output.sampled_tokens_map.get(req_id_str, {})
            for block_id, accepted_ids in accepted_ids_map.items():
                if not accepted_ids:
                    continue
                dllm_block = req.dllm_blocks[int(block_id)]
                sampled_tokens = sampled_tokens_map.get(block_id, [])
                true_local_ids = true_ids_map.get(block_id, [])
                for true_local_id, accepted_id in zip(true_local_ids, accepted_ids):
                    token = sampled_tokens[accepted_id]
                    dllm_block.write_token(token, true_local_id)
                req.new_tokens += len(accepted_ids)

            req.postprocess()
            if req.is_completed:
                req.status = DllmReqStatus.FINISHED
                self.kv_cache_manager.free(req)
                if req in self.running_reqs:
                    self.running_reqs.remove(req)
