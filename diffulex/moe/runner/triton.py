from __future__ import annotations

from diffulex.moe.dispatcher.datatype import CombineInput, DispatchOutput
from diffulex.moe.runner.base import MoERunner
from diffulex_kernel import fused_moe


class TritonFusedMoERunner(MoERunner):

    def forward(self, dispatch_output: DispatchOutput) -> CombineInput:
        topk_ids = dispatch_output.topk_output.ids
        topk_weights = dispatch_output.topk_output.weights
        if topk_ids is None or topk_weights is None:
            raise RuntimeError("TritonFusedMoERunner requires dense top-k ids and weights.")

        combined_hidden_states = fused_moe(
            hidden_states=dispatch_output.hidden_states,
            w13=self.w13,
            w2=self.w2,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            local_expert_start=self.local_expert_start,
            hidden_act=self.hidden_act,
        )
        self._all_reduce_output_if_needed(combined_hidden_states)

        return CombineInput(
            hidden_states=combined_hidden_states,
            context=dispatch_output.context,
        )
