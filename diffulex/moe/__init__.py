from diffulex.moe.config import (
    get_mlp_only_layers,
    get_moe_intermediate_size,
    get_num_experts,
    get_num_experts_per_tok,
    get_norm_topk_prob,
    is_moe_layer,
)
from diffulex.distributed.parallel_state import fetch_parallel_state


def build_mlp_or_moe(config, layer_idx: int, dense_factory):
    """Build a dense MLP or MoE block according to the config."""
    if is_moe_layer(config, layer_idx):
        from diffulex.moe.layer import build_moe_block

        parallel_state = fetch_parallel_state()
        if parallel_state.is_ep_enabled():
            return build_moe_block("ep", config)
        
        if parallel_state.is_tp_enabled():
            return build_moe_block("tp", config)
        
        return build_moe_block("naive", config)
    
    return dense_factory()


def __getattr__(name: str):
    if name in ("FusedMoE", "SparseMoEBlock"):
        from diffulex.moe.layer import FusedMoE

        return FusedMoE
    if name == "build_moe_block":
        from diffulex.moe.layer import build_moe_block

        return build_moe_block
    if name in ("DispatcherOutput", "TokenDispatcher", "build_token_dispatcher"):
        from diffulex.moe.dispatcher import DispatcherOutput, TokenDispatcher, build_token_dispatcher

        return {
            "DispatcherOutput": DispatcherOutput,
            "TokenDispatcher": TokenDispatcher,
            "build_token_dispatcher": build_token_dispatcher,
        }[name]
    if name == "DeepEPDispatcher":
        from diffulex.moe.dispatcher.deepep_dispatcher import DeepEPDispatcher

        return DeepEPDispatcher
    if name == "NaiveA2ADispatcher":
        from diffulex.moe.dispatcher.naive_dispatcher import NaiveA2ADispatcher

        return NaiveA2ADispatcher
    if name == "DeepEPMode":
        from diffulex.moe.mode import DeepEPMode

        return DeepEPMode
    if name in ("DispatchMetadata", "DeepEPDispatchMetadata", "DispatcherStage", "ExpertExecutionMetadata", "RouterMetadata"):
        from diffulex.moe.metadata import (
            DeepEPDispatchMetadata,
            DispatchMetadata,
            DispatcherStage,
            ExpertExecutionMetadata,
            RouterMetadata,
        )

        return {
            "DeepEPDispatchMetadata": DeepEPDispatchMetadata,
            "DispatchMetadata": DispatchMetadata,
            "DispatcherStage": DispatcherStage,
            "ExpertExecutionMetadata": ExpertExecutionMetadata,
            "RouterMetadata": RouterMetadata,
        }[name]
    raise AttributeError(name)


__all__ = [
    "FusedMoE",
    "SparseMoEBlock",
    "DeepEPDispatchMetadata",
    "DispatchMetadata",
    "DispatcherStage",
    "DeepEPDispatcher",
    "DeepEPMode",
    "DispatcherOutput",
    "ExpertExecutionMetadata",
    "NaiveA2ADispatcher",
    "RouterMetadata",
    "TokenDispatcher",
    "build_token_dispatcher",
    "build_mlp_or_moe",
    "get_mlp_only_layers",
    "get_moe_intermediate_size",
    "get_num_experts",
    "get_num_experts_per_tok",
    "get_norm_topk_prob",
    "is_moe_layer",
]
