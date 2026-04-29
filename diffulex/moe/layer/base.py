from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffulex.attention import fetch_attn_metadata
from diffulex.logger import get_logger
from diffulex.moe.config import (
    get_moe_intermediate_size,
    get_num_experts,
    get_num_experts_per_tok,
    get_norm_topk_prob,
)
from diffulex.moe.topk import build_topk_router
from diffulex_kernel import fused_moe
from diffulex.layer.activation import SiluAndMul
from diffulex.layer.linear import ColumnParallelLinear, RowParallelLinear

_VLLM_FUSED_MOE = None
_VLLM_FUSED_MOE_LOAD_ERR: Exception | None = None
_VLLM_MODULAR_LOAD_ERR: Exception | None = None
logger = get_logger(__name__)


def _load_vllm_fused_moe():
    """Load the vendored vLLM fused_moe implementation on demand.

    The vendored module imports vLLM, so keep this lazy to avoid making vLLM a
    hard dependency unless the vLLM MoE backend is explicitly selected.
    """
    global _VLLM_FUSED_MOE, _VLLM_FUSED_MOE_LOAD_ERR
    if _VLLM_FUSED_MOE is not None:
        return _VLLM_FUSED_MOE
    if _VLLM_FUSED_MOE_LOAD_ERR is not None:
        return None

    try:
        from diffulex_kernel.python.vllm_fuse_moe import fused_moe as vllm_fused_moe

        _VLLM_FUSED_MOE = vllm_fused_moe
        return _VLLM_FUSED_MOE
    except Exception as exc:
        _VLLM_FUSED_MOE_LOAD_ERR = exc
        return None


def _load_vllm_modular_moe():
    global _VLLM_MODULAR_LOAD_ERR
    if _VLLM_MODULAR_LOAD_ERR is not None:
        return None

    try:
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.config import (
            FUSED_MOE_UNQUANTIZED_CONFIG,
            FusedMoEConfig,
            FusedMoEParallelConfig,
            RoutingMethodType,
        )
        from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts
        from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
        from vllm.model_executor.layers.fused_moe.prepare_finalize import (
            make_moe_prepare_and_finalize_no_dp_ep,
        )
        from vllm.v1.worker.workspace import (
            init_workspace_manager,
            is_workspace_manager_initialized,
        )

        return {
            "FUSED_MOE_UNQUANTIZED_CONFIG": FUSED_MOE_UNQUANTIZED_CONFIG,
            "FusedMoEConfig": FusedMoEConfig,
            "FusedMoEKernel": FusedMoEKernel,
            "FusedMoEParallelConfig": FusedMoEParallelConfig,
            "MoEActivation": MoEActivation,
            "RoutingMethodType": RoutingMethodType,
            "TritonExperts": TritonExperts,
            "init_workspace_manager": init_workspace_manager,
            "is_workspace_manager_initialized": is_workspace_manager_initialized,
            "make_moe_prepare_and_finalize_no_dp_ep": make_moe_prepare_and_finalize_no_dp_ep,
        }
    except Exception as exc:
        _VLLM_MODULAR_LOAD_ERR = exc
        return None


class SharedExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, *, hidden_act: str = "silu") -> None:
        super().__init__()
        if hidden_act != "silu":
            raise NotImplementedError("SharedExpertMLP currently supports only silu.")
        self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(torch.cat((self.gate_proj(hidden_states), self.up_proj(hidden_states)), dim=-1))
        )

class FusedMoE(nn.Module, ABC):

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
        num_shared_experts: int = 0,
        shared_expert_intermediate_size: int | None = None,
    ) -> None:
        super().__init__()

        if hidden_act != "silu":
            raise NotImplementedError("only silu is supported currently")
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_act = hidden_act
        self.norm_topk_prob = norm_topk_prob
        self.moe_gemm_impl = str(moe_gemm_impl)
        self.num_shared_experts = num_shared_experts
        self._vllm_modular_kernel = None
        self._vllm_modular_logged = False

        self.router = build_topk_router(
            "triton",
            top_k=top_k,
            renormalize=norm_topk_prob,
            scoring_func="softmax",
        )
        self.shared_experts = None
        if num_shared_experts > 0:
            shared_intermediate_size = int(shared_expert_intermediate_size or intermediate_size * num_shared_experts)
            self.shared_experts = SharedExpertMLP(
                hidden_size,
                shared_intermediate_size,
                hidden_act=hidden_act,
            )

        self.fetch_attn_metadata = fetch_attn_metadata

    @classmethod
    def from_config(cls, config) -> "FusedMoE":
        return cls(
            hidden_size=config.hidden_size,
            intermediate_size=get_moe_intermediate_size(config),
            num_experts=get_num_experts(config),
            top_k=get_num_experts_per_tok(config),
            hidden_act=getattr(config, "hidden_act", "silu"),
            norm_topk_prob=get_norm_topk_prob(config),
            moe_gemm_impl=getattr(config, "moe_gemm_impl", "triton"),
            num_shared_experts=int(getattr(config, "num_shared_experts", 0) or 0),
        )
    
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # return final_hidden_states, router_logits
        raise NotImplementedError

    def add_shared_experts(self, routed_states: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.shared_experts is None:
            return routed_states
        shared_states = self.shared_experts(hidden_states)
        return routed_states + shared_states

    @staticmethod
    def _phase_from_prefill_flags(is_prefill) -> str:
        if isinstance(is_prefill, bool):
            return "prefill" if is_prefill else "decode"

        if torch.is_tensor(is_prefill):
            if is_prefill.numel() == 0:
                return "unknown"
            flags = is_prefill.to(dtype=torch.bool)
            all_prefill = bool(flags.all().item())
            any_prefill = bool(flags.any().item())
        else:
            try:
                flags = [bool(flag) for flag in is_prefill]
            except TypeError:
                return "unknown"
            if not flags:
                return "unknown"
            all_prefill = all(flags)
            any_prefill = any(flags)

        if all_prefill:
            return "prefill"
        if not any_prefill:
            return "decode"
        return "mixed"

    def get_current_phase(self) -> str:
        """Return the current inference phase for this forward pass."""
        try:
            attn_metadata = self.fetch_attn_metadata()
        except Exception:
            return "unknown"

        if attn_metadata is None:
            return "unknown"

        phase = self._phase_from_prefill_flags(attn_metadata.is_prefill)
        if phase != "unknown":
            return phase

        status_table = attn_metadata.status_table
        if status_table is None:
            return "unknown"

        if torch.is_tensor(status_table):
            return self._phase_from_prefill_flags(status_table == 0)

        try:
            return self._phase_from_prefill_flags([int(status) == 0 for status in status_table])
        except TypeError:
            return self._phase_from_prefill_flags(int(status_table) == 0)

    def _build_vllm_modular_kernel(self, hidden_states: torch.Tensor, w13: torch.Tensor):
        api = _load_vllm_modular_moe()
        if api is None:
            raise RuntimeError(f"vLLM modular MoE backend is unavailable: {_VLLM_MODULAR_LOAD_ERR!r}")
        if str(self.hidden_act) != "silu":
            raise NotImplementedError("vllm_modular MoE currently supports only silu activation.")

        if not api["is_workspace_manager_initialized"]():
            api["init_workspace_manager"](hidden_states.device)

        parallel_config = api["FusedMoEParallelConfig"].make_no_parallel()
        moe_config = api["FusedMoEConfig"](
            num_experts=int(w13.shape[0]),
            experts_per_token=int(self.top_k),
            hidden_dim=int(self.hidden_size),
            intermediate_size_per_partition=int(self.intermediate_size),
            num_local_experts=int(w13.shape[0]),
            num_logical_experts=int(w13.shape[0]),
            activation=api["MoEActivation"].SILU,
            device=hidden_states.device,
            routing_method=(
                api["RoutingMethodType"].Renormalize
                if self.norm_topk_prob
                else api["RoutingMethodType"].Default
            ),
            moe_parallel_config=parallel_config,
            in_dtype=hidden_states.dtype,
            max_num_tokens=max(1, int(hidden_states.shape[0])),
        )
        quant_config = api["FUSED_MOE_UNQUANTIZED_CONFIG"]
        prepare_finalize = api["make_moe_prepare_and_finalize_no_dp_ep"](use_monolithic=False)
        fused_experts = api["TritonExperts"](moe_config=moe_config, quant_config=quant_config)
        if not self._vllm_modular_logged:
            logger.info(
                "Initialized vLLM modular MoE backend: hidden=%s intermediate=%s local_experts=%s "
                "top_k=%s dtype=%s tokens=%s",
                self.hidden_size,
                self.intermediate_size,
                int(w13.shape[0]),
                self.top_k,
                hidden_states.dtype,
                int(hidden_states.shape[0]),
            )
            self._vllm_modular_logged = True
        return api["FusedMoEKernel"](
            prepare_finalize,
            fused_experts,
            shared_experts=None,
            moe_parallel_config=parallel_config,
            inplace=False,
        )

    def _vllm_modular_expert_gemm(
        self,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        local_expert_start: int,
    ) -> torch.Tensor:
        api = _load_vllm_modular_moe()
        if api is None:
            raise RuntimeError(f"vLLM modular MoE backend is unavailable: {_VLLM_MODULAR_LOAD_ERR!r}")
        if self._vllm_modular_kernel is None:
            self._vllm_modular_kernel = self._build_vllm_modular_kernel(hidden_states, w13)

        local_topk_ids = topk_ids.to(torch.int64) - int(local_expert_start)
        valid = (local_topk_ids >= 0) & (local_topk_ids < w13.shape[0])
        safe_topk_ids = torch.where(valid, local_topk_ids, torch.zeros_like(local_topk_ids))
        safe_topk_weights = torch.where(valid, topk_weights, torch.zeros_like(topk_weights))
        return self._vllm_modular_kernel.apply(
            hidden_states=hidden_states,
            w1=w13,
            w2=w2,
            topk_weights=safe_topk_weights,
            topk_ids=safe_topk_ids,
            activation=api["MoEActivation"].SILU,
            global_num_experts=int(w13.shape[0]),
            expert_map=None,
            apply_router_weight_on_input=False,
            shared_experts_input=None,
        )

    @torch.compiler.disable
    def expert_gemm(
        self,
        impl: str,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        local_expert_start: int = 0,
        hidden_act: str = "silu"
    ) -> torch.Tensor:
        if impl == "triton":
            out = fused_moe(
                hidden_states=hidden_states,
                w13=w13,
                w2=w2,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                local_expert_start=local_expert_start,
                hidden_act=hidden_act,
            )
            return out
        if impl == "vllm_modular":
            return self._vllm_modular_expert_gemm(
                hidden_states=hidden_states,
                w13=w13,
                w2=w2,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                local_expert_start=local_expert_start,
            )
        if impl == "vllm":
            vllm_fused_moe = _load_vllm_fused_moe()
            if vllm_fused_moe is None:
                # Soft fallback to current kernel so diagnostics can continue.
                out = fused_moe(
                    hidden_states=hidden_states,
                    w13=w13,
                    w2=w2,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    local_expert_start=local_expert_start,
                    hidden_act=hidden_act,
                )
            else:
                # External fused_moe expects local expert ids.
                local_topk_ids = topk_ids.to(torch.int64) - int(local_expert_start)
                valid = (local_topk_ids >= 0) & (local_topk_ids < w13.shape[0])
                safe_topk_ids = torch.where(valid, local_topk_ids, torch.zeros_like(local_topk_ids))
                safe_topk_weights = torch.where(valid, topk_weights, torch.zeros_like(topk_weights))
                out = vllm_fused_moe(
                    hidden_states=hidden_states,
                    w1=w13,
                    w2=w2,
                    topk_weights=safe_topk_weights,
                    topk_ids=safe_topk_ids,
                    inplace=False,
                )
            return out
        if impl == "naive":
            num_tokens, hidden_size = hidden_states.shape
            num_local_experts = w13.shape[0]
            intermediate_size = w13.shape[1] // 2
            final_hidden_states = hidden_states.new_zeros((num_tokens, hidden_size))
            local_topk_ids = topk_ids.to(torch.int64) - int(local_expert_start)
            
            for token_idx in range(num_tokens):
                token_hidden = hidden_states[token_idx]
                token_out = torch.zeros(hidden_size, device=hidden_states.device, dtype=torch.float32)

                for slot_idx in range(topk_ids.shape[1]):
                    local_expert_idx = int(local_topk_ids[token_idx, slot_idx].item())

                    if local_expert_idx < 0 or local_expert_idx >= num_local_experts:
                        continue

                    weight = topk_weights[token_idx, slot_idx]
                    if weight.item() == 0:
                        continue

                    expert_w13 = w13[local_expert_idx]
                    gate_proj = expert_w13[:intermediate_size]
                    up_proj = expert_w13[intermediate_size:]
                    
                    gate = torch.matmul(token_hidden, gate_proj.transpose(0, 1))
                    up = torch.matmul(token_hidden, up_proj.transpose(0, 1))
                    activated = F.silu(gate) * up
                    
                    expert_out = torch.matmul(activated, w2[local_expert_idx].transpose(0, 1))
                    token_out += expert_out.float() * weight.float()
                
                final_hidden_states[token_idx] = token_out.to(hidden_states.dtype)

            return final_hidden_states
        raise ValueError(f"Unknown MoE expert_gemm impl: {impl}")


__all__ = ["FusedMoE", "SharedExpertMLP"]
