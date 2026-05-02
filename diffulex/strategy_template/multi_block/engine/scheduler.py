from __future__ import annotations

from diffulex.engine.request import DllmReq
from diffulex.engine.scheduler import SchedulerBase
from diffulex.engine.status import DllmReqStatus
from diffulex.logger import get_logger
from diffulex.mixin.edit.scheduler import EditSchedulerMixin


class MultiBlockSchedulerTemplate(EditSchedulerMixin, SchedulerBase):
    _logger = get_logger(__name__)

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
            if num_batched_tokens + projected > self.max_num_batched_tokens or not self.kv_cache_manager.can_allocate(
                req
            ):
                break

            num_reqs += 1
            self.kv_cache_manager.allocate(req)
            req.apply_cached_prefix_pages()
            if req.is_preempted:
                if not self.kv_cache_manager.can_append(req):
                    self.kv_cache_manager.free(req)
                    num_reqs -= 1
                    break
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
                    f"new_tokens={candidate.new_tokens}, "
                    f"cached={candidate.num_cached_tokens}, "
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
            true_ids_map = sample_output.true_local_ids_map[req_id_str]
            accepted_ids_map = sample_output.accepted_ids_map[req_id_str]
            sampled_tokens_map = sample_output.sampled_tokens_map[req_id_str]
            for block_id, accepted_ids in accepted_ids_map.items():
                if not accepted_ids:
                    continue

                dllm_block = req.dllm_blocks[int(block_id)]
                sampled_tokens = sampled_tokens_map[block_id]
                true_local_ids = true_ids_map[block_id]
                for true_local_id, accepted_id in zip(true_local_ids, accepted_ids):
                    token = sampled_tokens[accepted_id]
                    dllm_block.write_token(token, true_local_id)
                req.new_tokens += len(accepted_ids)

            self.apply_edit_writes_map(req, sample_output)
            block_state_map = getattr(sample_output, "block_state_map", {}).get(req_id_str, {})
            for block in req.dllm_blocks:
                if not getattr(block, "is_active", False):
                    continue
                state = block_state_map.get(str(block.block_id))
                observe_edit_state = getattr(block, "observe_edit_state", None)
                if callable(observe_edit_state) and state is not None:
                    observe_edit_state(
                        token_ids=state["token_ids"],
                        confidences=state.get("confidences"),
                    )
                elif callable(observe_edit_state) and block.is_complete:
                    observe_edit_state(
                        token_ids=block.token_ids,
                        confidences=[1.0] * int(block.block_size),
                    )

            req.postprocess()
            req.nfe += 1
            update_auto_max_nfe = getattr(req, "update_auto_max_nfe", None)
            if callable(update_auto_max_nfe):
                update_auto_max_nfe()
            if (
                req.max_new_tokens_reached
                or req.max_model_len_reached
                or req.max_nfe_reached
                or req.max_repetition_run_reached
            ):
                if req.max_new_tokens_reached:
                    reason = "max_new_tokens_reached"
                elif req.max_model_len_reached:
                    reason = "max_model_len_reached"
                elif req.max_nfe_reached:
                    reason = "max_nfe_reached"
                else:
                    reason = "max_repetition_run_reached"
                req.force_deactivate(reason=reason)
            if req.is_completed:
                if req.completion_reason is None:
                    req.completion_reason = "completed_without_reason"
                self._logger.info(
                    "Req %s marked FINISHED (reason=%s, eos=%s, max_new=%s, max_model_len=%s, max_nfe=%s, max_repeat=%s, nfe=%s, gen_tokens=%s, auto_max_nfe=%s, avg_tpf=%s)",
                    req.req_id,
                    req.completion_reason,
                    req.eos_token_generated,
                    req.max_new_tokens_reached,
                    req.max_model_len_reached,
                    req.max_nfe_reached,
                    req.max_repetition_run_reached,
                    req.nfe,
                    len(req.truncated_response) if req.truncated_response is not None else -1,
                    getattr(req, "auto_max_nfe_value", None),
                    (
                        f"{getattr(req, 'auto_max_nfe_avg_tpf', 0.0):.2f}"
                        if getattr(req, "auto_max_nfe_avg_tpf", None) is not None
                        else "n/a"
                    ),
                )
                req.status = DllmReqStatus.FINISHED
                self.kv_cache_manager.free(req)
                if req in self.running_reqs:
                    self.running_reqs.remove(req)
