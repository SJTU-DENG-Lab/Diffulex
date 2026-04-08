"""
Note on the current engine semantics:

The current model-parallel engine uses replicated-token semantics when entering
the MoE layer. In practice this means EP ranks see the same token hidden states,
instead of each rank owning a disjoint token subset.

Under this assumption, the trivial EP fallback (each rank computes only its
local experts and then all-reduces the final hidden states) is often simpler
and can be comparatively better than A2A, because it avoids explicit token
movement between EP ranks.

This A2A dispatcher is meant for the standard EP routing pattern where tokens
have a unique source owner inside the EP communication domain, are dispatched
to expert owners with all-to-all, and are then sent back for combine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist

from diffulex.moe.dispatcher.base import TokenDispatcher
from diffulex.moe.dispatcher.datatype import CombineInput, DispatchOutput
from diffulex.moe.topk import TopKOutput
from diffulex.utils.parallelism import get_model_parallelism_metadata


@dataclass(frozen=True)
class A2ADispatchContext:
    send_counts: tuple[int, ...]
    recv_counts: tuple[int, ...]
    sort_indices: torch.Tensor
    flat_token_indices: torch.Tensor
    flat_topk_slot_indices: torch.Tensor


def _all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    *,
    output_split_sizes: Sequence[int],
    input_split_sizes: Sequence[int],
    group: dist.ProcessGroup | None = None,
) -> None:
    kwargs = {
        "output_split_sizes": list(output_split_sizes),
        "input_split_sizes": list(input_split_sizes),
    }
    if group is None:
        dist.all_to_all_single(output, input, **kwargs)
    else:
        dist.all_to_all_single(output, input, group=group, **kwargs)


def _all_to_all_tensor(
    tensor: torch.Tensor,
    *,
    recv_counts: Sequence[int],
    send_counts: Sequence[int],
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    per_token_shape = tensor.shape[1:]
    total_recv = sum(recv_counts)
    output = torch.empty(
        (total_recv, *per_token_shape),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    _all_to_all_single(
        output,
        tensor.contiguous(),
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
        group=group,
    )
    return output


class A2ATokenDispatcher(TokenDispatcher):
    """EP token dispatcher using all-to-all communication on the default EP world."""

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        *,
        num_local_experts: int,
        local_expert_start: int = 0,
        ep_size: int | None = None,
        ep_group: dist.ProcessGroup | None = None,
        **_: object,
    ) -> None:
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_local_experts,
            local_expert_start=local_expert_start,
        )
        self.layout = get_model_parallelism_metadata()
        self.ep_size = ep_size if ep_size is not None else self.layout.ep_size
        self.ep_group = ep_group
        self._validate_layout()

    def _validate_layout(self) -> None:
        if not (self.layout.is_pure_ep or self.layout.is_tp_eq_ep):
            raise ValueError(
                "A2ATokenDispatcher requires pure EP or tp == ep layout, "
                f"got {self.layout}."
            )
        if self.ep_size != self.layout.ep_size:
            raise ValueError(
                "A2ATokenDispatcher ep_size does not match initialized model parallel layout, "
                f"dispatcher ep_size={self.ep_size}, layout={self.layout}."
            )
        expected_local_experts = self.num_experts // self.ep_size
        if self.num_local_experts != expected_local_experts:
            raise ValueError(
                "A2ATokenDispatcher expects contiguous full experts partitioned by EP rank, "
                f"got num_experts={self.num_experts}, ep_size={self.ep_size}, "
                f"num_local_experts={self.num_local_experts}, expected={expected_local_experts}."
            )
        expected_local_expert_start = self.layout.ep_rank * self.num_local_experts
        if self.local_expert_start != expected_local_expert_start:
            raise ValueError(
                "A2ATokenDispatcher expects local experts to match current EP rank ownership, "
                f"got local_expert_start={self.local_expert_start}, expected={expected_local_expert_start}."
            )

    def _expected_expert_ids(self) -> list[int]:
        return list(range(self.local_expert_start, self.local_expert_end))

    def _resolve_expert_ids(self, active_expert_ids: Sequence[int] | None) -> list[int]:
        expected = self._expected_expert_ids()
        if active_expert_ids is None:
            return expected

        expert_ids = list(active_expert_ids)
        if expert_ids != expected:
            raise ValueError(f"A2ATokenDispatcher expects active_expert_ids={expected}, got {expert_ids}.")
        return expert_ids

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        *,
        active_expert_ids: Sequence[int] | None = None,
        hidden_states_scale: torch.Tensor | None = None,
    ) -> DispatchOutput:
        if topk_output.ids is None or topk_output.weights is None:
            raise RuntimeError("Token dispatch requires top-k ids and weights.")

        self._resolve_expert_ids(active_expert_ids)

        flat_hidden_states = hidden_states.repeat_interleave(self.top_k, dim=0)
        flat_expert_ids = topk_output.ids.reshape(-1)
        flat_token_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device).repeat_interleave(self.top_k)
        flat_topk_slot_indices = (
            torch.arange(self.top_k, device=hidden_states.device)
            .unsqueeze(0)
            .expand(hidden_states.shape[0], self.top_k)
            .reshape(-1)
        )

        target_ep_ranks = torch.div(flat_expert_ids, self.num_local_experts, rounding_mode="floor")
        local_expert_ids = flat_expert_ids.remainder(self.num_local_experts)
        sort_indices = torch.argsort(target_ep_ranks, stable=True)

        sorted_hidden_states = flat_hidden_states.index_select(0, sort_indices)
        sorted_local_expert_ids = local_expert_ids.index_select(0, sort_indices)

        send_counts = torch.bincount(target_ep_ranks, minlength=self.ep_size).to(dtype=torch.long)
        recv_counts = torch.empty_like(send_counts)
        _all_to_all_single(
            recv_counts,
            send_counts,
            output_split_sizes=[1] * self.ep_size,
            input_split_sizes=[1] * self.ep_size,
            group=self.ep_group,
        )

        send_counts_list = tuple(int(v) for v in send_counts.cpu().tolist())
        recv_counts_list = tuple(int(v) for v in recv_counts.cpu().tolist())

        recv_hidden_states = _all_to_all_tensor(
            sorted_hidden_states,
            recv_counts=recv_counts_list,
            send_counts=send_counts_list,
            group=self.ep_group,
        )
        recv_local_expert_ids = _all_to_all_tensor(
            sorted_local_expert_ids,
            recv_counts=recv_counts_list,
            send_counts=send_counts_list,
            group=self.ep_group,
        )

        recv_hidden_states_scale = None
        if hidden_states_scale is not None:
            flat_hidden_states_scale = hidden_states_scale.repeat_interleave(self.top_k, dim=0)
            sorted_hidden_states_scale = flat_hidden_states_scale.index_select(0, sort_indices)
            recv_hidden_states_scale = _all_to_all_tensor(
                sorted_hidden_states_scale,
                recv_counts=recv_counts_list,
                send_counts=send_counts_list,
                group=self.ep_group,
            )

        expert_token_indices = []
        expert_topk_slot_indices = []
        for local_expert_id in range(self.num_local_experts):
            token_idx = torch.where(recv_local_expert_ids == local_expert_id)[0]
            expert_token_indices.append(token_idx)
            expert_topk_slot_indices.append(torch.empty_like(token_idx))

        return DispatchOutput(
            hidden_states=recv_hidden_states,
            topk_output=topk_output,
            num_tokens=hidden_states.shape[0],
            expert_token_indices=tuple(expert_token_indices),
            expert_topk_slot_indices=tuple(expert_topk_slot_indices),
            hidden_states_scale=recv_hidden_states_scale,
            context=A2ADispatchContext(
                send_counts=send_counts_list,
                recv_counts=recv_counts_list,
                sort_indices=sort_indices,
                flat_token_indices=flat_token_indices,
                flat_topk_slot_indices=flat_topk_slot_indices,
            ),
        )

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        if combine_input.context is None:
            raise RuntimeError("A2ATokenDispatcher.combine requires dispatch context.")
        ctx = combine_input.context
        if not isinstance(ctx, A2ADispatchContext):
            raise TypeError(f"Unexpected A2A dispatch context type: {type(ctx)!r}.")

        processed_hidden_states = combine_input.expert_hidden_states[0].new_zeros(
            (sum(ctx.recv_counts), combine_input.hidden_size)
        )
        for expert_hidden_states, token_idx in zip(
            combine_input.expert_hidden_states,
            combine_input.expert_token_indices,
            strict=True,
        ):
            if token_idx.numel() == 0:
                continue
            processed_hidden_states.index_copy_(0, token_idx, expert_hidden_states)

        returned_hidden_states = _all_to_all_tensor(
            processed_hidden_states,
            recv_counts=ctx.send_counts,
            send_counts=ctx.recv_counts,
            group=self.ep_group,
        )

        unsorted_hidden_states = torch.empty_like(returned_hidden_states)
        unsorted_hidden_states.index_copy_(0, ctx.sort_indices, returned_hidden_states)

        routing_weights = combine_input.topk_weights[
            ctx.flat_token_indices,
            ctx.flat_topk_slot_indices,
        ].to(unsorted_hidden_states.dtype)
        weighted_hidden_states = unsorted_hidden_states * routing_weights.unsqueeze(-1)

        final_hidden_states = torch.zeros(
            (combine_input.num_tokens, combine_input.hidden_size),
            dtype=weighted_hidden_states.dtype,
            device=weighted_hidden_states.device,
        )
        final_hidden_states.index_add_(0, ctx.flat_token_indices, weighted_hidden_states)
        return final_hidden_states
