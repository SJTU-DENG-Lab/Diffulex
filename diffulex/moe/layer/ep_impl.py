import re

import torch
import torch.nn as nn
import torch.distributed as dist

from diffulex.layer.linear import ReplicatedLinear, divide
from diffulex.moe.layer.base import FusedMoE
from diffulex.utils.checkpoint import LoadContext, ResolvedWeight
from diffulex.utils.parallelism import get_ep_rank, get_ep_world_size
from diffulex_kernel import fused_moe


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
    ) -> None:
        super().__init__(
            hidden_size,
            intermediate_size,
            num_experts,
            top_k,
            hidden_act=hidden_act,
            norm_topk_prob=norm_topk_prob
        )
        
        self.ep_rank = get_ep_rank()
        self.ep_size = get_ep_world_size()
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

    def shard_tokens(self, flat_hidden_states):
        num_tokens = flat_hidden_states.shape[0]
        token_indices = torch.arange(num_tokens, device=flat_hidden_states.device)
        local_token_indices = token_indices[self.ep_rank::self.ep_size]
        local_hidden_states = flat_hidden_states[local_token_indices]
        return local_hidden_states, local_token_indices, num_tokens

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape
        flat_hidden_states = hidden_states.reshape(-1, original_shape[-1]).contiguous()
        device = flat_hidden_states.device
        dtype = flat_hidden_states.dtype

        num_tokens, hidden_size = flat_hidden_states.shape
        ep_rank = self.ep_rank
        ep_size = self.ep_size
        top_k = self.top_k

        if num_tokens == 0:
            return flat_hidden_states.reshape(original_shape), None

        # ------------------------------------------------------------------
        # 1) shard tokens: rank r owns token indices [r, r + ep_size, ...]
        # ------------------------------------------------------------------
        global_token_indices = torch.arange(num_tokens, device=device, dtype=torch.int32)
        local_token_indices = global_token_indices[ep_rank::ep_size].contiguous()
        local_hidden_states = flat_hidden_states[local_token_indices].contiguous()  # [T_local, H]
        num_local_tokens = local_hidden_states.shape[0]

        # ------------------------------------------------------------------
        # 2) local gate/topk on GLOBAL experts
        #    NOTE: self.gate must output self.num_experts, not self.num_local_experts
        # ------------------------------------------------------------------
        router_logits_local = self.gate(local_hidden_states)  # [T_local, num_experts]
        topk_output = self.router(router_logits_local)
        topk_weights = topk_output.weights
        topk_ids_global = topk_output.ids
        topk_ids_global = topk_ids_global.to(torch.int32).contiguous()
        topk_weights = topk_weights.contiguous()

        # ------------------------------------------------------------------
        # 3) flatten [T_local, K] => slot-major [S]
        # ------------------------------------------------------------------
        num_slots = num_local_tokens * top_k

        slot_hidden_states = local_hidden_states.repeat_interleave(top_k, dim=0).contiguous()  # [S, H]
        slot_token_indices = (
            local_token_indices.unsqueeze(1)
            .expand(num_local_tokens, top_k)
            .reshape(-1)
            .contiguous()
        )  # [S]

        slot_topk_indices = (
            torch.arange(top_k, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(num_local_tokens, top_k)
            .reshape(-1)
            .contiguous()
        )  # [S]

        slot_expert_ids_global = topk_ids_global.reshape(-1).contiguous()   # [S]
        slot_weights = topk_weights.reshape(-1).contiguous()                # [S]

        # global expert id -> dst rank / local expert id
        dst_rank = torch.div(
            slot_expert_ids_global,
            self.num_local_experts,
            rounding_mode="floor",
        ).to(torch.int32)  # [S]
        dst_local_expert = (
            slot_expert_ids_global - dst_rank * self.num_local_experts
        ).to(torch.int32).contiguous()  # [S]

        # ------------------------------------------------------------------
        # 4) sort send payload by dst_rank so we can all_to_all_single
        # ------------------------------------------------------------------
        sort_idx = torch.argsort(dst_rank)
        dst_rank_sorted = dst_rank[sort_idx]
        send_hidden_states = slot_hidden_states[sort_idx].contiguous()
        send_local_expert = dst_local_expert[sort_idx].contiguous()
        send_token_indices = slot_token_indices[sort_idx].contiguous()
        send_topk_indices = slot_topk_indices[sort_idx].contiguous()
        send_weights = slot_weights[sort_idx].contiguous()
        send_src_rank = torch.full(
            (num_slots,),
            ep_rank,
            device=device,
            dtype=torch.int32,
        )

        # count sends to each dst rank
        send_counts = torch.bincount(dst_rank_sorted.to(torch.int64), minlength=ep_size).to(torch.int32)

        # exchange counts first
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)

        send_counts_cpu = send_counts.cpu()
        recv_counts_cpu = recv_counts.cpu()

        send_splits = send_counts_cpu.tolist()
        recv_splits = recv_counts_cpu.tolist()

        total_recv_slots = int(recv_counts.sum().item())

        # recv buffers
        recv_hidden_states = torch.empty(
            (total_recv_slots, hidden_size),
            device=device,
            dtype=dtype,
        )
        recv_local_expert = torch.empty(
            (total_recv_slots,),
            device=device,
            dtype=torch.int32,
        )
        recv_token_indices = torch.empty(
            (total_recv_slots,),
            device=device,
            dtype=torch.int32,
        )
        recv_topk_indices = torch.empty(
            (total_recv_slots,),
            device=device,
            dtype=torch.int32,
        )
        recv_weights = torch.empty(
            (total_recv_slots,),
            device=device,
            dtype=topk_weights.dtype,
        )
        recv_src_rank = torch.empty(
            (total_recv_slots,),
            device=device,
            dtype=torch.int32,
        )

        # dispatch A2A
        dist.all_to_all_single(
            recv_hidden_states,
            send_hidden_states,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )
        dist.all_to_all_single(
            recv_local_expert,
            send_local_expert,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )
        dist.all_to_all_single(
            recv_token_indices,
            send_token_indices,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )
        dist.all_to_all_single(
            recv_topk_indices,
            send_topk_indices,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )
        dist.all_to_all_single(
            recv_weights,
            send_weights,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )
        dist.all_to_all_single(
            recv_src_rank,
            send_src_rank,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )

        # ------------------------------------------------------------------
        # 5) local expert compute on received slots
        #    Adapt current expert_gemm API by using top_k=1
        # ------------------------------------------------------------------
        if total_recv_slots > 0:
            recv_topk_ids_local = recv_local_expert[:, None].contiguous()  # [R, 1]
            recv_topk_weights_local = torch.ones(
                (total_recv_slots, 1),
                device=device,
                dtype=topk_weights.dtype,
            )
            recv_slot_outputs = self.expert_gemm(
                impl="triton",
                hidden_states=recv_hidden_states,
                w13=self.w13,
                w2=self.w2,
                topk_ids=recv_topk_ids_local,
                topk_weights=recv_topk_weights_local,
                local_expert_start=0,
                hidden_act=self.hidden_act,
            ).contiguous()  # [R, H]
        else:
            recv_slot_outputs = torch.empty((0, hidden_size), device=device, dtype=dtype)

        # ------------------------------------------------------------------
        # 6) send outputs back to source rank (combine A2A)
        # ------------------------------------------------------------------
        # sort by src rank
        back_sort_idx = torch.argsort(recv_src_rank)
        back_dst_rank_sorted = recv_src_rank[back_sort_idx]

        send_back_outputs = recv_slot_outputs[back_sort_idx].contiguous()
        send_back_token_indices = recv_token_indices[back_sort_idx].contiguous()
        send_back_topk_indices = recv_topk_indices[back_sort_idx].contiguous()
        send_back_weights = recv_weights[back_sort_idx].contiguous()

        send_back_counts = torch.bincount(
            back_dst_rank_sorted.to(torch.int64),
            minlength=ep_size,
        ).to(torch.int32)

        recv_back_counts = torch.empty_like(send_back_counts)
        dist.all_to_all_single(recv_back_counts, send_back_counts)

        send_back_splits = send_back_counts.cpu().tolist()
        recv_back_splits = recv_back_counts.cpu().tolist()
        total_returned_slots = int(recv_back_counts.sum().item())

        returned_outputs = torch.empty(
            (total_returned_slots, hidden_size),
            device=device,
            dtype=dtype,
        )
        returned_token_indices = torch.empty(
            (total_returned_slots,),
            device=device,
            dtype=torch.int32,
        )
        returned_topk_indices = torch.empty(
            (total_returned_slots,),
            device=device,
            dtype=torch.int32,
        )
        returned_weights = torch.empty(
            (total_returned_slots,),
            device=device,
            dtype=topk_weights.dtype,
        )

        dist.all_to_all_single(
            returned_outputs,
            send_back_outputs,
            output_split_sizes=recv_back_splits,
            input_split_sizes=send_back_splits,
        )
        dist.all_to_all_single(
            returned_token_indices,
            send_back_token_indices,
            output_split_sizes=recv_back_splits,
            input_split_sizes=send_back_splits,
        )
        dist.all_to_all_single(
            returned_topk_indices,
            send_back_topk_indices,
            output_split_sizes=recv_back_splits,
            input_split_sizes=send_back_splits,
        )
        dist.all_to_all_single(
            returned_weights,
            send_back_weights,
            output_split_sizes=recv_back_splits,
            input_split_sizes=send_back_splits,
        )

        # ------------------------------------------------------------------
        # 7) combine on token-owner rank
        #    owner rank only owns local_token_indices
        # ------------------------------------------------------------------
        local_final_hidden_states = torch.zeros(
            (num_local_tokens, hidden_size),
            device=device,
            dtype=torch.float32,
        )

        if total_returned_slots > 0:
            # token indices owned by this rank are exactly:
            # ep_rank, ep_rank + ep_size, ep_rank + 2*ep_size, ...
            # so inverse mapping is (token_idx - ep_rank) // ep_size
            local_positions = torch.div(
                returned_token_indices - ep_rank,
                ep_size,
                rounding_mode="floor",
            ).to(torch.int64)

            weighted_outputs = returned_outputs.float() * returned_weights.unsqueeze(-1).float()
            local_final_hidden_states.index_add_(0, local_positions, weighted_outputs)

        local_final_hidden_states = local_final_hidden_states.to(dtype).contiguous()

        # ------------------------------------------------------------------
        # 8) restore replicated-token semantics after MoE
        #    gather all local token shards and scatter into full [T, H] on every rank
        # ------------------------------------------------------------------
        local_token_count = torch.tensor(
            [num_local_tokens],
            device=device,
            dtype=torch.int64,
        )
        gathered_counts = [torch.zeros_like(local_token_count) for _ in range(ep_size)]
        dist.all_gather(gathered_counts, local_token_count)
        gathered_counts = [int(x.item()) for x in gathered_counts]

        
        max_local_tokens = max(gathered_counts)
        # pad token indices
        padded_local_token_indices = torch.full(
            (max_local_tokens,),
            fill_value=-1,
            device=device,
            dtype=torch.int32,
        )
        padded_local_token_indices[:num_local_tokens] = local_token_indices.contiguous()
        # pad outputs
        padded_local_outputs = torch.zeros(
            (max_local_tokens, hidden_size),
            device=device,
            dtype=dtype,
        )
        padded_local_outputs[:num_local_tokens] = local_final_hidden_states.contiguous()

        gathered_token_indices = [
            torch.empty((cnt,), device=device, dtype=torch.int32) for cnt in gathered_counts
        ]
        gathered_outputs = [
            torch.empty((cnt, hidden_size), device=device, dtype=dtype) for cnt in gathered_counts
        ]

        dist.all_gather(gathered_token_indices, padded_local_token_indices)
        dist.all_gather(gathered_outputs, padded_local_outputs)

        final_hidden_states = torch.empty(
            (num_tokens, hidden_size),
            device=device,
            dtype=dtype,
        )

        for rank in range(ep_size):
            cnt = gathered_counts[rank]
            if cnt == 0:
                continue
            idx = gathered_token_indices[rank][:cnt].long()
            out = gathered_outputs[rank][:cnt]
            final_hidden_states[idx] = out

        return final_hidden_states.reshape(original_shape), None
    
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
