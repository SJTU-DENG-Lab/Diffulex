from __future__ import annotations

from diffulex.strategy_template.multi_block.engine.scheduler import MultiBlockSchedulerTemplate
from diffulex.strategy_template.token_merging_multi_block.engine.request import TokenMergingMultiBlockReqTemplate


class TokenMergingMultiBlockSchedulerTemplate(MultiBlockSchedulerTemplate):
    def postprocess_token_merging_multi_block(
        self,
        reqs: list[TokenMergingMultiBlockReqTemplate],
        sample_output,
    ) -> None:
        self.postprocess_multi_block(reqs, sample_output)

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
