import re

import torch

from diffulex.layer.linear import ReplicatedLinear, divide, tp_all_reduce
from diffulex.moe.layer.base import FusedMoE
from diffulex.moe.layer.sglang_backend import SGLangTPFusedMoEAdapter
from diffulex.utils.checkpoint import LoadContext, ResolvedWeight
from diffulex.distributed.parallel_state import fetch_parallel_state


class TPFusedMoE(FusedMoE):
    """
    Standard TP MoE without token dispatch.

    Every TP rank receives the full local token batch and owns a contiguous
    shard of experts. The router still selects global expert ids; each rank
    computes only the selected experts it owns, then TP all-reduce sums the
    partial expert contributions.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        *,
        hidden_act: str = "silu",
        norm_topk_prob: bool = True,
        moe_gemm_impl: str = "triton",
        moe_topk_impl: str = "triton",
        num_shared_experts: int = 0,
        shared_expert_intermediate_size: int | None = None,
    ) -> None:
        super().__init__(
            hidden_size,
            intermediate_size,
            num_experts,
            top_k,
            hidden_act=hidden_act,
            norm_topk_prob=norm_topk_prob,
            moe_gemm_impl=moe_gemm_impl,
            moe_topk_impl=moe_topk_impl,
            num_shared_experts=num_shared_experts,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
        )
        
        parallel_state = fetch_parallel_state()
        self.tp_rank = parallel_state.get_tp_rank()
        self.tp_size = parallel_state.get_tp_world_size()
        self.tp_group = parallel_state.get_tp_group()
        self.num_local_experts = divide(self.num_experts, self.tp_size)
        self.local_expert_start = self.tp_rank * self.num_local_experts
        self.local_expert_end = self.local_expert_start + self.num_local_experts
        self.sglang_backend_name = self._map_sglang_backend_name(self.moe_gemm_impl)

        self.gate = ReplicatedLinear(hidden_size, self.num_experts, bias=False)
        self.sglang_moe = SGLangTPFusedMoEAdapter(
            num_experts=self.num_experts,
            hidden_size=hidden_size,
            intermediate_size=self.intermediate_size,
            top_k=self.top_k,
            hidden_act=self.hidden_act,
            backend_name=self.sglang_backend_name,
        )

    @staticmethod
    def _map_sglang_backend_name(impl: str) -> str:
        if impl == "triton":
            return "triton"
        if impl == "flashinfer":
            return "flashinfer_cutlass"
        if impl == "naive":
            return "triton"
        raise ValueError(f"Unknown MoE expert_gemm impl: {impl}")

    @property
    def w13(self) -> torch.nn.Parameter:
        return self.sglang_moe.w13_weight

    @property
    def w2(self) -> torch.nn.Parameter:
        return self.sglang_moe.w2_weight

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape
        flat_hidden_states = hidden_states.reshape(-1, original_shape[-1])

        router_logits = self.gate(flat_hidden_states)
        topk_output = self.router(router_logits)
        topk_weights = topk_output.weights
        topk_ids = topk_output.ids
        if self.moe_gemm_impl == "naive":
            final_hidden_states = super().expert_gemm(
                impl="naive",
                hidden_states=flat_hidden_states,
                w13=self.w13,
                w2=self.w2,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                local_expert_start=self.local_expert_start,
                hidden_act=self.hidden_act,
            )
        else:
            final_hidden_states = self.sglang_moe(
                hidden_states=flat_hidden_states,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                router_logits=router_logits,
            )
        if self.tp_size > 1:
            final_hidden_states = tp_all_reduce(final_hidden_states, self.tp_group)

        final_hidden_states = final_hidden_states.reshape(original_shape)
        final_hidden_states = self.add_shared_experts(final_hidden_states, hidden_states)
        return final_hidden_states, router_logits

    def _local_expert_idx(self, expert_idx: int) -> int | None:
        if expert_idx < self.local_expert_start or expert_idx >= self.local_expert_end:
            return None
        return expert_idx - self.local_expert_start

    def _load_sglang_w13(self, loaded_weight: torch.Tensor, expert_idx: int, shard_id: str) -> None:
        self.sglang_moe.layer.weight_loader(self.w13, loaded_weight, "weight", shard_id, expert_idx)
    
    def load_w1(self, loaded_weight: torch.Tensor, expert_idx: int) -> None:
        self._load_sglang_w13(loaded_weight, expert_idx, "w1")
    
    def load_w3(self, loaded_weight: torch.Tensor, expert_idx: int) -> None:
        self._load_sglang_w13(loaded_weight, expert_idx, "w3")
    
    def load_w2(self, loaded_weight: torch.Tensor, expert_idx: int) -> None:
        if loaded_weight.shape == (self.intermediate_size, self.hidden_size):
            loaded_weight = loaded_weight.transpose(0, 1).contiguous()
        self.sglang_moe.layer.weight_loader(self.w2, loaded_weight, "weight", "w2", expert_idx)

    def resolve_checkpoint_weight(self, suffix: str, ctx: LoadContext) -> ResolvedWeight | None:
        # Stacked format: experts.gate_proj.weight (all experts in one tensor)
        stacked_match = re.fullmatch(r"experts\.(gate_proj|up_proj|down_proj)\.weight", suffix)
        if stacked_match is not None:
            proj_name = stacked_match.group(1)
            return ResolvedWeight(
                loader=lambda loaded_weight, proj_name=proj_name: self._load_stacked_expert(
                    loaded_weight, proj_name
                )
            )

        # Individual expert format: experts.0.gate_proj.weight
        match = re.fullmatch(r"experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", suffix)
        if match is None:
            return None

        expert_idx = int(match.group(1))
        proj_name = match.group(2)

        if proj_name == "gate_proj":
            return ResolvedWeight(
                loader=lambda loaded_weight, expert_idx=expert_idx: self.load_w1(
                    loaded_weight,
                    expert_idx,
                )
            )

        if proj_name == "up_proj":
            return ResolvedWeight(
                loader=lambda loaded_weight, expert_idx=expert_idx: self.load_w3(
                    loaded_weight,
                    expert_idx,
                )
            )

        if proj_name == "down_proj":
            return ResolvedWeight(
                loader=lambda loaded_weight, expert_idx=expert_idx: self.load_w2(
                    loaded_weight,
                    expert_idx,
                )
            )

        return None

    def _load_stacked_expert(self, loaded_weight: torch.Tensor, proj_name: str) -> None:
        """Load stacked expert weight [num_experts, ...] and slice own TP shard."""
        for expert_idx in range(int(loaded_weight.shape[0])):
            expert_weight = loaded_weight[expert_idx]
            if proj_name == "gate_proj":
                self.load_w1(expert_weight, expert_idx)
            elif proj_name == "up_proj":
                self.load_w3(expert_weight, expert_idx)
            elif proj_name == "down_proj":
                self.load_w2(expert_weight, expert_idx)

__all__ = ["TPFusedMoE"]
