from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from diffulex.moe.dispatcher.base import TokenDispatcher
from diffulex.moe.dispatcher.datatype import CombineInput, DispatchOutput
from diffulex.moe.topk import TopKOutput
from diffulex.utils.parallelism import get_model_parallelism_metadata


class TrivialTokenDispatcher(TokenDispatcher):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        *,
        num_local_experts: int | None = None,
        local_expert_start: int = 0,
        **_: object,
    ) -> None:
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_local_experts,
            local_expert_start=local_expert_start,
        )
        self.layout = get_model_parallelism_metadata()
        self._validate_layout()

    def _validate_layout(self) -> None:
        if self.layout.world_size == 1 or self.layout.is_tp_only:
            if self.num_local_experts != self.num_experts:
                raise ValueError(
                    "TrivialTokenDispatcher expects all experts to be local for single-device or TP-only layout, "
                    f"got num_experts={self.num_experts}, num_local_experts={self.num_local_experts}."
                )
            if self.local_expert_start != 0:
                raise ValueError(
                    "TrivialTokenDispatcher expects local_expert_start == 0 for single-device or TP-only layout, "
                    f"got {self.local_expert_start}."
                )
            return

        if self.layout.is_pure_ep or self.layout.is_tp_eq_ep:
            return

        raise ValueError(f"Unsupported layout for TrivialTokenDispatcher: {self.layout}.")

    def _expected_expert_ids(self) -> list[int]:
        if self.layout.world_size == 1 or self.layout.is_tp_only:
            return list(range(self.num_experts))
        return list(range(self.local_expert_start, self.local_expert_end))

    def _resolve_expert_ids(
        self,
        active_expert_ids: Sequence[int] | None,
        *,
        expected_expert_ids: Sequence[int],
    ) -> list[int]:
        expected = list(expected_expert_ids)
        if active_expert_ids is None:
            return expected

        expert_ids = list(active_expert_ids)
        if expert_ids != expected:
            raise ValueError(f"Expected active_expert_ids={expected}, got {expert_ids}.")
        return expert_ids

    def _dispatch_trivial(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        *,
        expert_ids: Sequence[int],
        hidden_states_scale: torch.Tensor | None = None,
    ) -> DispatchOutput:
        if topk_output.ids is None or topk_output.weights is None:
            raise RuntimeError("Token dispatch requires top-k ids and weights.")
        if ((topk_output.ids < 0) | (topk_output.ids >= self.num_experts)).any():
            invalid_ids = topk_output.ids[
                ((topk_output.ids < 0) | (topk_output.ids >= self.num_experts))
            ][:16].detach().cpu().tolist()
            raise RuntimeError(
                "Top-k routing ids are out of range before dispatch. "
                f"num_experts={self.num_experts}, sample_invalid_ids={invalid_ids}."
            )

        expert_mask = F.one_hot(topk_output.ids.to(torch.long), num_classes=self.num_experts).permute(2, 1, 0)
        expert_token_indices = []
        expert_topk_slot_indices = []

        for expert_idx in expert_ids:
            topk_slot_idx, token_idx = torch.where(expert_mask[expert_idx])
            expert_token_indices.append(token_idx)
            expert_topk_slot_indices.append(topk_slot_idx)

        return DispatchOutput(
            hidden_states=hidden_states,
            topk_output=topk_output,
            num_tokens=hidden_states.shape[0],
            expert_token_indices=tuple(expert_token_indices),
            expert_topk_slot_indices=tuple(expert_topk_slot_indices),
            hidden_states_scale=hidden_states_scale,
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        *,
        active_expert_ids: Sequence[int] | None = None,
        hidden_states_scale: torch.Tensor | None = None,
    ) -> DispatchOutput:
        expert_ids = self._resolve_expert_ids(
            active_expert_ids,
            expected_expert_ids=self._expected_expert_ids(),
        )
        return self._dispatch_trivial(
            hidden_states,
            topk_output,
            expert_ids=expert_ids,
            hidden_states_scale=hidden_states_scale,
        )

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        return combine_input.hidden_states
