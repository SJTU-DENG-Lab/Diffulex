from diffulex.moe.config import (
    get_mlp_only_layers,
    get_moe_intermediate_size,
    get_num_experts,
    get_num_experts_per_tok,
    get_norm_topk_prob,
    is_moe_layer,
)
from diffulex.moe.moe_impl import SparseMoEBlock


def build_mlp_or_moe(config, layer_idx: int, dense_factory):
    """Build a dense MLP or MoE block according to the config."""
    if is_moe_layer(config, layer_idx):
        return SparseMoEBlock.from_config(config)
    return dense_factory()


__all__ = [
    "SparseMoEBlock",
    "build_mlp_or_moe",
    "get_mlp_only_layers",
    "get_moe_intermediate_size",
    "get_num_experts",
    "get_num_experts_per_tok",
    "get_norm_topk_prob",
    "is_moe_layer",
]
