from diffulex.moe.layer.base import FusedMoE
from diffulex.moe.layer.trivial_impl import TrivialFusedMoE
from diffulex.moe.layer.tp_impl import TPFusedMoE
from diffulex.moe.layer.ep_impl import EPFusedMoE


"""
Note:
Currently supported layouts:
0. single device
1. TP only: tp_size > 1, ep_size = 1
2. pure EP: ep_size > 1, tp_size = 1, world_size == ep_size
3. full-expert TP == EP: ep_size > 1, tp_size == ep_size, world_size == ep_size == tp_size
"""


def build_moe_block(
        impl: str,
        config,
) -> FusedMoE:
    if impl == "trivial":
        return TrivialFusedMoE.from_config(config)
    elif impl == "tp":
        return TPFusedMoE.from_config(config)
    elif impl == "ep":
        return EPFusedMoE.from_config(config)
    else:
        raise NotImplementedError


__all__ = [
    "build_moe_block",
    "FusedMoE",

    "TrivialFusedMoE",
    "TPFusedMoE",
    "EPFusedMoE",
]
