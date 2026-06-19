from __future__ import annotations

from diffulex.engine.scheduler import MultiBlockSchedulerTemplate
from diffulex.strategy.templates.token_merge.engine.request import TokenMergeReqTemplate


class TokenMergeSchedulerTemplate(MultiBlockSchedulerTemplate):
    def postprocess_token_merge(
        self,
        reqs: list[TokenMergeReqTemplate],
        sample_output,
    ) -> None:
        super().postprocess(reqs, sample_output)

        token_merge_map = sample_output.token_merge_map
        for req in reqs:
            if req.is_running:
                req.prune_token_merge_descriptors_to_running_sequence()

            req_merge_map = token_merge_map[str(req.req_id)]
            if not req_merge_map:
                continue

            for position, descriptor in req_merge_map.items():
                if descriptor is None:
                    req.clear_token_merge_descriptor(int(position))
                    continue
                req.set_token_merge_descriptor(
                    int(position),
                    descriptor["topk_ids"],
                    descriptor["topk_probs"],
                    descriptor["residual_prob"],
                )
            if req.is_running:
                req.prune_token_merge_descriptors_to_running_sequence()
