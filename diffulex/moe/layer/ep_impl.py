import re

import torch
import torch.nn as nn
import torch.distributed as dist

from diffulex_kernel import fused_expert_packed
from diffulex.layer.linear import ReplicatedLinear, divide
from diffulex.moe.config import get_moe_intermediate_size, get_norm_topk_prob, get_num_experts, get_num_experts_per_tok
from diffulex.moe.dispatcher.base_dispatcher import build_token_dispatcher
from diffulex.moe.layer.base import FusedMoE
from diffulex.moe.metadata import DeepEPDispatchMetadata, RouterMetadata
from diffulex.utils.checkpoint import LoadContext, ResolvedWeight
from diffulex.distributed.parallel_state import fetch_parallel_state


class EPFusedMoE(FusedMoE):
    """
    if ep is on, moe layer will only use ep even if tp is on
    so whole expert weight is distributed to ep_size devices

    have all-to-all token dispatch, each rank computes 1/ep_size
    of gate and topk of tokens, and send to owner, then combine
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
        dispatcher_backend: str = "naive",
        deepep_mode: str = "auto",
        deepep_num_max_dispatch_tokens_per_rank: int = 256,
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
            num_shared_experts=num_shared_experts,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
        )
        
        parallel_state = fetch_parallel_state()
        self.ep_rank = parallel_state.get_ep_rank()
        self.ep_size = parallel_state.get_ep_world_size()
        self.ep_group = parallel_state.get_ep_group()
        self.dp_size = parallel_state.get_dp_world_size()
        self.tp_rank = parallel_state.get_tp_rank()
        self.tp_size = parallel_state.get_tp_world_size()
        self.tp_group = parallel_state.get_tp_group()
        self.tp_ranks = parallel_state.base_model.tp_ranks
        self.cross_dp_ep = parallel_state.is_cross_dp_ep_enabled()
        self.dispatcher_backend = dispatcher_backend
        if dispatcher_backend not in {"naive", "deepep"}:
            raise RuntimeError(
                "EPFusedMoE only supports dispatcher-driven A2A backends: 'naive' or 'deepep'. "
                "Use TPFusedMoE with moe_dispatcher_backend='standard' for non-A2A TP MoE."
            )
        if self.ep_size <= 1:
            raise RuntimeError(
                "EPFusedMoE requires expert_parallel_size > 1. "
                "Use TPFusedMoE for standard TP MoE or NaiveFusedMoE for single-rank MoE."
            )
        self.num_local_experts = divide(self.num_experts, self.ep_size)
        self.local_expert_start = self.ep_rank * self.num_local_experts
        self.local_expert_end = self.local_expert_start + self.num_local_experts
        self.active_expert_ids = list(range(self.local_expert_start, self.local_expert_end))

        # every rank process 1 / ep_size of total tokens and do a2a communication
        self.gate = ReplicatedLinear(hidden_size, self.num_experts, bias=False)
        self.w13 = nn.Parameter(
            torch.empty(self.num_local_experts, self.intermediate_size * 2, hidden_size)
        )
        self.w2 = nn.Parameter(
            torch.empty(self.num_local_experts, hidden_size, self.intermediate_size)
        )
        self.dispatcher = build_token_dispatcher(
            dispatcher_backend,
            ep_group=self.ep_group,
            ep_size=self.ep_size,
            num_local_experts=self.num_local_experts,
            top_k=self.top_k,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            params_dtype=self.w13.dtype,
            deepep_mode=deepep_mode,
            num_max_dispatch_tokens_per_rank=deepep_num_max_dispatch_tokens_per_rank,
        )

    @classmethod
    def from_config(cls, config) -> "EPFusedMoE":
        return cls(
            hidden_size=config.hidden_size,
            intermediate_size=get_moe_intermediate_size(config),
            num_experts=get_num_experts(config),
            top_k=get_num_experts_per_tok(config),
            hidden_act=getattr(config, "hidden_act", "silu"),
            norm_topk_prob=get_norm_topk_prob(config),
            moe_gemm_impl=getattr(config, "moe_gemm_impl", "triton"),
            dispatcher_backend=getattr(config, "moe_dispatcher_backend", "naive"),
            deepep_mode=getattr(config, "deepep_mode", "auto"),
            deepep_num_max_dispatch_tokens_per_rank=getattr(
                config,
                "deepep_num_max_dispatch_tokens_per_rank",
                256,
            ),
            num_shared_experts=int(getattr(config, "num_shared_experts", 0) or 0),
        )

    def shard_tokens(self, flat_hidden_states):
        num_tokens = flat_hidden_states.shape[0]
        token_indices = torch.arange(num_tokens, device=flat_hidden_states.device)
        local_token_indices = token_indices[self.ep_rank::self.ep_size]
        local_hidden_states = flat_hidden_states[local_token_indices]
        return local_hidden_states, local_token_indices, num_tokens

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forward_token_sharded_a2a(hidden_states)

    def _forward_token_sharded_a2a(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape
        flat_hidden_states = hidden_states.reshape(-1, original_shape[-1])
        num_tokens = flat_hidden_states.shape[0]
        local_hidden_states, local_token_indices, _ = self.shard_tokens(flat_hidden_states)
        self.dispatcher.set_forward_phase(self.get_current_phase())

        if local_hidden_states.shape[0] == 0:
            router_metadata = RouterMetadata.empty(
                local_hidden_states,
                num_experts=self.num_experts,
                top_k=self.top_k,
            )
            dispatched = self.dispatcher.dispatch(
                local_hidden_states,
                router_metadata.topk_ids,
                router_metadata.topk_weights,
            )
            recv_slot_outputs = self._run_dispatched_experts(dispatched, flat_hidden_states.dtype)
            local_final_hidden_states = self.dispatcher.combine(recv_slot_outputs, dispatched.metadata)
            local_router_logits = router_metadata.router_logits
        else:
            local_router_logits = self.gate(local_hidden_states)
            topk_output = self.router(local_router_logits)
            router_metadata = RouterMetadata.from_topk_output(topk_output)
            dispatched = self.dispatcher.dispatch(
                local_hidden_states,
                router_metadata.topk_ids,
                router_metadata.topk_weights,
            )
            recv_slot_outputs = self._run_dispatched_experts(dispatched, router_metadata.topk_weights.dtype)
            local_final_hidden_states = self.dispatcher.combine(recv_slot_outputs, dispatched.metadata)

        final_hidden_states = torch.zeros_like(flat_hidden_states)
        router_logits = flat_hidden_states.new_zeros((num_tokens, self.num_experts))
        if local_token_indices.numel() > 0:
            final_hidden_states[local_token_indices.long()] = local_final_hidden_states
            router_logits[local_token_indices.long()] = local_router_logits.to(router_logits.dtype)
        dist.all_reduce(final_hidden_states, group=self.ep_group)
        dist.all_reduce(router_logits, group=self.ep_group)
        final_hidden_states = final_hidden_states.reshape(original_shape)
        final_hidden_states = self.add_shared_experts(final_hidden_states, hidden_states)
        return final_hidden_states, router_logits

    def _run_dispatched_experts(self, dispatched, weight_dtype: torch.dtype) -> torch.Tensor:
        dispatch_ctx = dispatched.metadata
        total_recv_slots = int(dispatch_ctx.total_recv_slots)
        if total_recv_slots == 0:
            return torch.empty(
                (0, int(dispatch_ctx.hidden_size)),
                device=dispatch_ctx.device,
                dtype=dispatch_ctx.dtype,
            )
            
        recv_hidden_states = dispatched.recv_hidden_states
        recv_local_expert = dispatched.recv_local_expert_ids
        if isinstance(dispatch_ctx, DeepEPDispatchMetadata):
            recv_slot_outputs = fused_expert_packed(
                hidden_states=recv_hidden_states,
                w13=self.w13,
                w2=self.w2,
                execution_metadata=dispatch_ctx.to_expert_execution_metadata(),
                hidden_act=self.hidden_act,
            ).contiguous()
            if not dispatch_ctx.low_latency:
                recv_slot_outputs.mul_(dispatch_ctx.recv_weights.to(recv_slot_outputs.dtype).unsqueeze(-1))
            return recv_slot_outputs
        
        recv_topk_ids_local = recv_local_expert[:, None].contiguous()
        recv_topk_weights_local = torch.ones(
            (total_recv_slots, 1),
            device=recv_hidden_states.device,
            dtype=weight_dtype,
        )
        recv_slot_outputs = self.expert_gemm(
            impl=self.moe_gemm_impl,
            hidden_states=recv_hidden_states,
            w13=self.w13,
            w2=self.w2,
            topk_ids=recv_topk_ids_local,
            topk_weights=recv_topk_weights_local,
            local_expert_start=0,
            hidden_act=self.hidden_act,
        ).contiguous()
        recv_slot_outputs.mul_(dispatch_ctx.recv_weights.to(recv_slot_outputs.dtype).unsqueeze(-1))
        return recv_slot_outputs
    
    def owns_global_expert(self, expert_idx: int) -> bool:
        return self.local_expert_start <= expert_idx < self.local_expert_end
    
    def global_to_local_expert_id(self, global_expert_idx: int) -> int:
        assert self.owns_global_expert(global_expert_idx), f"global_expert_idx {global_expert_idx} is not owned by this rank"
        return global_expert_idx - self.local_expert_start

    def load_w1(self, loaded_weight: torch.Tensor, local_expert_idx: int) -> None:
        self.w13.data[local_expert_idx, 0 : self.intermediate_size].copy_(loaded_weight)

    def load_w3(self, loaded_weight: torch.Tensor, local_expert_idx: int) -> None:
        self.w13.data[local_expert_idx, self.intermediate_size : 2 * self.intermediate_size].copy_(loaded_weight)

    def load_w2(self, loaded_weight: torch.Tensor, local_expert_idx: int) -> None:
        self.w2.data[local_expert_idx].copy_(loaded_weight)

    def resolve_checkpoint_weight(self, suffix: str, ctx: LoadContext) -> ResolvedWeight | None:
        match = re.fullmatch(r"experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", suffix)
        if match is None:
            return None

        expert_idx = int(match.group(1))
        if not self.owns_global_expert(expert_idx):
            return ResolvedWeight(skip=True)

        local_expert_idx = self.global_to_local_expert_id(expert_idx)
        proj_name = match.group(2)

        if proj_name == "gate_proj":
            return ResolvedWeight(
                loader=lambda loaded_weight, local_expert_idx=local_expert_idx: self.load_w1(
                    loaded_weight,
                    local_expert_idx,
                )
            )

        if proj_name == "up_proj":
            return ResolvedWeight(
                loader=lambda loaded_weight, local_expert_idx=local_expert_idx: self.load_w3(
                    loaded_weight,
                    local_expert_idx,
                )
            )

        if proj_name == "down_proj":
            return ResolvedWeight(
                loader=lambda loaded_weight, local_expert_idx=local_expert_idx: self.load_w2(
                    loaded_weight,
                    local_expert_idx,
                )
            )

        return None


__all__ = ["EPFusedMoE"]
