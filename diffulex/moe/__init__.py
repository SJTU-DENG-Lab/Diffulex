from diffulex.moe.config import (
    get_mlp_only_layers,
    get_moe_intermediate_size,
    get_num_experts,
    get_num_experts_per_tok,
    get_norm_topk_prob,
    is_moe_layer,
)
from diffulex.moe.layer import FusedMoE, build_moe_block
from diffulex.utils.parallelism import is_ep_enabled, is_tp_enabled


def build_mlp_or_moe(config, layer_idx: int, dense_factory):
    """Build a dense MLP or MoE block according to the config."""
    if is_moe_layer(config, layer_idx):
        if is_ep_enabled():
            return build_moe_block("ep", config)
        
        if is_tp_enabled():
            return build_moe_block("tp", config)
        
        return build_moe_block("trivial", config)
    
    return dense_factory()


__all__ = [
    "FusedMoE",

    "build_mlp_or_moe",
    
    "get_mlp_only_layers",
    "get_moe_intermediate_size",
    "get_num_experts",
    "get_num_experts_per_tok",
    "get_norm_topk_prob",
    "is_moe_layer",
]
