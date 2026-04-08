from __future__ import annotations

import torch.nn.functional as F

from diffulex.moe.dispatcher.datatype import CombineInput, DispatchOutput
from diffulex.moe.runner.base import MoERunner


class TrivialMoERunner(MoERunner):
    def forward(self, dispatch_output: DispatchOutput) -> CombineInput:
        expert_hidden_states = []

        for expert_idx, token_idx in enumerate(dispatch_output.expert_token_indices):
            if token_idx.numel() == 0:
                expert_hidden_states.append(
                    dispatch_output.hidden_states.new_empty((0, self.hidden_size))
                )
                continue

            expert_input = dispatch_output.hidden_states[token_idx]
            gate_up = F.linear(expert_input, self.w13[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)

            if self.hidden_act != "silu":
                raise ValueError(f"Only silu experts are supported right now, got {self.hidden_act!r}.")

            activated = F.silu(gate) * up
            expert_hidden_states.append(F.linear(activated, self.w2[expert_idx]))

        return CombineInput(
            expert_hidden_states=tuple(expert_hidden_states),
            expert_token_indices=dispatch_output.expert_token_indices,
            expert_topk_slot_indices=dispatch_output.expert_topk_slot_indices,
            topk_weights=dispatch_output.topk_output.weights,
            num_tokens=dispatch_output.num_tokens,
            hidden_size=self.hidden_size,
            context=dispatch_output.context
        )
