from __future__ import annotations
import torch.nn.functional as F

from diffulex.moe.dispatcher.datatype import CombineInput, DispatchOutput
from diffulex.moe.runner.base import MoERunner


class TrivialMoERunner(MoERunner):
    def forward(self, dispatch_output: DispatchOutput) -> CombineInput:
        topk_weights = dispatch_output.topk_output.weights
        if topk_weights is None:
            raise RuntimeError("TrivialMoERunner requires dense top-k weights.")

        final_hidden_states = dispatch_output.hidden_states.new_zeros(
            (dispatch_output.num_tokens, self.hidden_size)
        )

        for expert_idx, (token_idx, topk_slot_idx) in enumerate(
            zip(
                dispatch_output.expert_token_indices,
                dispatch_output.expert_topk_slot_indices,
                strict=True,
            )
        ):
            if token_idx.numel() == 0:
                continue

            expert_input = dispatch_output.hidden_states[token_idx]
            gate_up = F.linear(expert_input, self.w13[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)

            if self.hidden_act != "silu":
                raise ValueError(f"Only silu experts are supported right now, got {self.hidden_act!r}.")

            activated = F.silu(gate) * up
            expert_hidden_states = F.linear(activated, self.w2[expert_idx])
            routing_weights = topk_weights[token_idx, topk_slot_idx].to(expert_hidden_states.dtype)
            final_hidden_states.index_add_(
                0,
                token_idx,
                expert_hidden_states * routing_weights.unsqueeze(-1),
            )

        self._all_reduce_output_if_needed(final_hidden_states)

        return CombineInput(
            hidden_states=final_hidden_states,
            context=dispatch_output.context,
        )
