from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import torch

from diffulex.moe.topk.output import TopKOutput


@dataclass(frozen=True)
class RouterMetadata:
    router_logits: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @classmethod
    def empty(
        cls,
        hidden_states: torch.Tensor,
        *,
        num_experts: int,
        top_k: int,
    ) -> "RouterMetadata":
        """Empty router output for idle/local-empty ranks in EP collectives."""
        return cls(
            router_logits=hidden_states.new_empty((0, num_experts)),
            topk_ids=torch.full(
                (0, top_k),
                -1,
                device=hidden_states.device,
                dtype=torch.int32,
            ),
            topk_weights=torch.empty(
                (0, top_k),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ),
        )

    @classmethod
    def from_topk_output(cls, topk_output: TopKOutput) -> "RouterMetadata":
        return cls(
            router_logits=topk_output.router_logits,
            topk_ids=topk_output.ids,
            topk_weights=topk_output.weights,
        )


class DispatcherStage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH_A = auto()
    AFTER_DISPATCH_B = auto()
    AFTER_COMBINE_A = auto()


@dataclass(frozen=True)
class DispatchMetadata:
    num_tokens: int
    hidden_size: int
    dtype: torch.dtype
    device: torch.device
    send_splits: list[int]
    recv_splits: list[int]
    recv_hidden_states: torch.Tensor
    recv_local_expert: torch.Tensor
    recv_token_indices: torch.Tensor
    recv_weights: torch.Tensor
    total_recv_slots: int
    active_dispatch: bool = True
    num_local_tokens: int | None = None
    local_token_indices: torch.Tensor | None = None


@dataclass(frozen=True)
class ExpertExecutionMetadata:
    packed_token_ids: torch.Tensor
    packed_local_expert_ids: torch.Tensor
    packed_weights: torch.Tensor
    num_slots: int
    seg_indptr: torch.Tensor | None = None
    num_recv_tokens_per_expert: torch.Tensor | None = None
    sorted_slot_ids: torch.Tensor | None = None
    expert_block_ids: torch.Tensor | None = None
    num_tokens_post_padded: int | None = None
    disable_aligned_metadata: bool = False


@dataclass(frozen=True)
class DeepEPDispatchMetadata(DispatchMetadata):
    src2dst: torch.Tensor | None = None
    reorder_indices: torch.Tensor | None = None
    reordered_token_indices: torch.Tensor | None = None
    reordered_local_expert_ids: torch.Tensor | None = None
    seg_indptr: torch.Tensor | None = None
    num_recv_tokens_per_expert: torch.Tensor | None = None
    native_handle: object | None = None
    native_recv_num_tokens: int | None = None
    native_recv_topk_ids: torch.Tensor | None = None
    native_recv_topk_weights: torch.Tensor | None = None
    low_latency: bool = False
    low_latency_handle: object | None = None
    low_latency_topk_ids: torch.Tensor | None = None
    low_latency_topk_weights: torch.Tensor | None = None
    low_latency_recv_count: torch.Tensor | None = None
    low_latency_capacity: int | None = None

    def to_expert_execution_metadata(self) -> ExpertExecutionMetadata:
        recv_local_expert = (
            self.reordered_local_expert_ids
            if self.reordered_local_expert_ids is not None
            else self.recv_local_expert
        )
        if recv_local_expert is None:
            raise ValueError("DeepEPDispatchMetadata is missing recv_local_expert information.")
        recv_weights = self.recv_weights
        if recv_weights is None:
            raise ValueError("DeepEPDispatchMetadata is missing recv_weights.")
        num_slots = int(recv_local_expert.numel())
        packed_token_ids = torch.arange(
            num_slots,
            device=recv_local_expert.device,
            dtype=torch.int32,
        )
        return ExpertExecutionMetadata(
            packed_token_ids=packed_token_ids,
            packed_local_expert_ids=recv_local_expert.to(torch.int32).contiguous(),
            packed_weights=recv_weights.contiguous(),
            num_slots=num_slots,
            seg_indptr=self.seg_indptr,
            num_recv_tokens_per_expert=self.num_recv_tokens_per_expert,
            disable_aligned_metadata=self.low_latency,
        )


__all__ = [
    "DispatchMetadata",
    "DeepEPDispatchMetadata",
    "DispatcherStage",
    "ExpertExecutionMetadata",
    "RouterMetadata",
]
