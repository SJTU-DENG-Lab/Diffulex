from abc import ABC
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffulex.attention import fetch_attn_metadata
from diffulex.distributed.parallel_state import fetch_parallel_state
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
from diffulex.layer.linear import MergedColumnParallelLinear, RowParallelLinear
from diffulex.utils.checkpoint import LoadContext, ResolvedWeight
from diffulex.vllm_compat import vllm_current_config

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
            FusedMoEConfig,
            FusedMoEParallelConfig,
            RoutingMethodType,
        )
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )
        from vllm.v1.worker.workspace import (
            init_workspace_manager,
            is_workspace_manager_initialized,
        )

        return {
            "FusedMoEConfig": FusedMoEConfig,
            "FusedMoEParallelConfig": FusedMoEParallelConfig,
            "MoEActivation": MoEActivation,
            "RoutingMethodType": RoutingMethodType,
            "UnquantizedFusedMoEMethod": UnquantizedFusedMoEMethod,
            "init_workspace_manager": init_workspace_manager,
            "is_workspace_manager_initialized": is_workspace_manager_initialized,
        }
    except Exception as exc:
        _VLLM_MODULAR_LOAD_ERR = exc
        return None


class _VllmModularMoELayerAdapter(nn.Module):
    """Small RoutedExperts-shaped adapter for vLLM modular MoE kernels."""

    def __init__(
        self,
        w13: torch.Tensor,
        w2: torch.Tensor,
        *,
        activation,
        moe_config=None,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        clone_weights: bool = True,
    ) -> None:
        super().__init__()
        if clone_weights:
            self.w13_weight = nn.Parameter(w13.detach().clone(), requires_grad=False)
            self.w2_weight = nn.Parameter(w2.detach().clone(), requires_grad=False)
        else:
            self.w13_weight = w13 if isinstance(w13, nn.Parameter) else nn.Parameter(w13, requires_grad=False)
            self.w2_weight = w2 if isinstance(w2, nn.Parameter) else nn.Parameter(w2, requires_grad=False)
        self.activation = activation
        self.moe_config = moe_config
        self.global_num_experts = int(global_num_experts)
        self.apply_router_weight_on_input = False
        self.expert_map = expert_map

    def _expert_routing_tables(self):
        return None


def _make_vllm_runtime_config() -> SimpleNamespace:
    state = fetch_parallel_state()
    return SimpleNamespace(
        tensor_parallel_size=int(state.tp_size),
        data_parallel_size=int(state.dp_size),
        expert_parallel_size=int(state.ep_size),
        distributed_timeout_seconds=600,
    )


def _build_vllm_expert_map(
    *,
    num_global_experts: int,
    num_local_experts: int,
    local_expert_start: int,
    device: torch.device,
) -> torch.Tensor | None:
    if int(num_local_experts) == int(num_global_experts) and int(local_expert_start) == 0:
        return None

    expert_map = torch.full(
        (int(num_global_experts),),
        -1,
        dtype=torch.int32,
        device=device,
    )
    local_end = int(local_expert_start) + int(num_local_experts)
    expert_map[int(local_expert_start):local_end] = torch.arange(
        int(num_local_experts),
        dtype=torch.int32,
        device=device,
    )
    return expert_map


class SharedExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, *, hidden_act: str = "silu") -> None:
        super().__init__()
        if hidden_act != "silu":
            raise NotImplementedError("SharedExpertMLP currently supports only silu.")
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size, intermediate_size], bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()
        self._register_state_dict_hook(self._add_state_dict_aliases)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(hidden_states)))

    def _add_state_dict_aliases(self, module, state_dict, prefix, local_metadata) -> None:
        gate_weight, up_weight = self.gate_up_proj.weight.split(self.gate_up_proj.weight.shape[0] // 2, dim=0)
        state_dict[prefix + "gate_proj.weight"] = gate_weight
        state_dict[prefix + "up_proj.weight"] = up_weight

    def resolve_checkpoint_weight(self, suffix: str, ctx: LoadContext) -> ResolvedWeight | None:
        if suffix == "gate_proj.weight":
            return ResolvedWeight(param=self.gate_up_proj.weight, shard_id=0)
        if suffix == "up_proj.weight":
            return ResolvedWeight(param=self.gate_up_proj.weight, shard_id=1)
        return None


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
        self._vllm_modular_method = None
        self._vllm_modular_layer = None
        self._vllm_modular_logged = False
        self._shared_experts_stream = None
        self._shared_experts_stream_device = None

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

    def start_shared_experts_async(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.cuda.Stream] | None:
        if self.shared_experts is None:
            return None
        if os.getenv("DIFFULEX_DISABLE_MOE_SHARED_EXPERT_OVERLAP", "0") == "1":
            return None
        if not hidden_states.is_cuda or not torch.cuda.is_available():
            return None

        device = hidden_states.device
        stream = self._shared_experts_stream
        if stream is None or self._shared_experts_stream_device != device:
            if torch.cuda.is_current_stream_capturing():
                return None
            with torch.cuda.device(device):
                stream = torch.cuda.Stream()
            self._shared_experts_stream = stream
            self._shared_experts_stream_device = device

        current_stream = torch.cuda.current_stream(device)
        stream.wait_stream(current_stream)
        with torch.cuda.stream(stream):
            shared_states = self.shared_experts(hidden_states)
        return shared_states, stream

    def finish_shared_experts_async(
        self,
        routed_states: torch.Tensor,
        shared_work: tuple[torch.Tensor, torch.cuda.Stream],
    ) -> torch.Tensor:
        shared_states, stream = shared_work
        torch.cuda.current_stream(routed_states.device).wait_stream(stream)
        return routed_states + shared_states

    @staticmethod
    def _phase_from_prefill_flags(is_prefill) -> str:
        if isinstance(is_prefill, bool):
            return "prefill" if is_prefill else "decode"

        if torch.is_tensor(is_prefill):
            return "unknown"
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

        has_prefill = getattr(attn_metadata, "has_prefill_static", None)
        all_prefill = getattr(attn_metadata, "all_prefill_static", None)
        if has_prefill is not None and all_prefill is not None:
            if bool(all_prefill):
                return "prefill"
            if not bool(has_prefill):
                return "decode"
            return "mixed"

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

    def _build_vllm_modular_kernel(
        self,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        local_expert_start: int,
    ):
        api = _load_vllm_modular_moe()
        if api is None:
            raise RuntimeError(f"vLLM modular MoE backend is unavailable: {_VLLM_MODULAR_LOAD_ERR!r}")
        if str(self.hidden_act) != "silu":
            raise NotImplementedError("vllm_modular MoE currently supports only silu activation.")

        if not api["is_workspace_manager_initialized"]():
            api["init_workspace_manager"](hidden_states.device)

        parallel_config = api["FusedMoEParallelConfig"].make_no_parallel()
        moe_config = api["FusedMoEConfig"](
            num_experts=int(self.num_experts),
            experts_per_token=int(self.top_k),
            hidden_dim=int(self.hidden_size),
            intermediate_size_per_partition=int(self.intermediate_size),
            num_local_experts=int(w13.shape[0]),
            num_logical_experts=int(self.num_experts),
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

        with vllm_current_config(_make_vllm_runtime_config()):
            method = api["UnquantizedFusedMoEMethod"](moe_config)

        expert_map = _build_vllm_expert_map(
            num_global_experts=int(self.num_experts),
            num_local_experts=int(w13.shape[0]),
            local_expert_start=int(local_expert_start),
            device=hidden_states.device,
        )
        layer = _VllmModularMoELayerAdapter(
            w13,
            w2,
            activation=api["MoEActivation"].SILU,
            moe_config=moe_config,
            global_num_experts=int(self.num_experts),
            expert_map=expert_map,
        )
        method.process_weights_after_loading(layer)
        self._log_vllm_modular_init(hidden_states, w13, expert_map)
        self._vllm_modular_method = method
        self._vllm_modular_layer = layer
        return method.moe_kernel

    def _log_vllm_modular_init(
        self,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        expert_map: torch.Tensor | None,
    ) -> None:
        if self._vllm_modular_logged:
            return
        logger.info(
            "Initialized vLLM modular MoE backend: hidden=%s intermediate=%s local_experts=%s "
            "global_experts=%s top_k=%s dtype=%s tokens=%s expert_map=%s",
            self.hidden_size,
            self.intermediate_size,
            int(w13.shape[0]),
            self.num_experts,
            self.top_k,
            hidden_states.dtype,
            int(hidden_states.shape[0]),
            expert_map is not None,
        )
        self._vllm_modular_logged = True

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
            self._vllm_modular_kernel = self._build_vllm_modular_kernel(
                hidden_states,
                w13,
                w2,
                local_expert_start,
            )
        if self._vllm_modular_method is None or self._vllm_modular_layer is None:
            raise RuntimeError("vLLM modular MoE backend was not initialized correctly.")
        required_topk_dtype = getattr(self._vllm_modular_method, "topk_indices_dtype", None)
        if required_topk_dtype is not None and topk_ids.dtype != required_topk_dtype:
            topk_ids = topk_ids.to(required_topk_dtype)

        return self._vllm_modular_method.forward_native(
            layer=self._vllm_modular_layer,
            x=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=None,
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
        hidden_act: str = "silu",
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
