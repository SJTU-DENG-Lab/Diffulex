from __future__ import annotations

import torch
import torch.nn as nn

from diffulex.attention import Attention
from diffulex.layer.activation import SiluAndMul
from diffulex.layer.layernorm import RMSNorm
from diffulex.layer.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from diffulex.layer.rotary_embedding import get_rope
from diffulex.utils.checkpoint import LoadContext, ResolvedWeight


class MergedQKVAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int,
        head_dim: int | None = None,
        qkv_bias: bool = False,
        out_bias: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: tuple | None = None,
        rotary_dim: int | None = None,
        attn_impl: str = "triton",
        qk_norm_eps: float | None = None,
        q_norm_name: str = "q_norm",
        k_norm_name: str = "k_norm",
    ) -> None:
        super().__init__()
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.qkv_proj.lora_target_aliases = ("qkv_proj", "query_key_value", "q_proj", "k_proj", "v_proj")
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.q_size = self.qkv_proj.q_proj_shard_size
        self.kv_size = self.qkv_proj.kv_proj_shard_size
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=out_bias,
        )
        self.o_proj.lora_target_aliases = ("o_proj", "dense", "attn_out")
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim or self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.scaling = self.head_dim**-0.5
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            attn_impl=attn_impl,
        )
        self._q_norm_name = q_norm_name
        self._k_norm_name = k_norm_name
        if qk_norm_eps is not None:
            setattr(self, q_norm_name, RMSNorm(self.head_dim, eps=qk_norm_eps))
            setattr(self, k_norm_name, RMSNorm(self.head_dim, eps=qk_norm_eps))

    @property
    def q_norm_module(self) -> RMSNorm | None:
        return getattr(self, self._q_norm_name, None)

    @property
    def k_norm_module(self) -> RMSNorm | None:
        return getattr(self, self._k_norm_name, None)

    def _apply_per_head_norm(
        self,
        x: torch.Tensor,
        num_heads: int,
        norm: RMSNorm | None,
    ) -> torch.Tensor:
        if norm is None:
            return x
        original_shape = x.shape
        x = x.reshape(*original_shape[:-1], num_heads, self.head_dim)
        x = norm(x)
        return x.reshape(original_shape)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        q = self._apply_per_head_norm(q, self.num_heads, self.q_norm_module)
        k = self._apply_per_head_norm(k, self.num_kv_heads, self.k_norm_module)
        q, k = self.rotary_emb(positions, q, k)
        return self.o_proj(self.attn(q, k, v, mask))

    def resolve_checkpoint_weight(self, suffix: str, ctx: LoadContext) -> ResolvedWeight | None:
        split_map = {
            "q_proj.weight": (self.qkv_proj.weight, "q"),
            "q_proj.bias": (self.qkv_proj.bias, "q"),
            "k_proj.weight": (self.qkv_proj.weight, "k"),
            "k_proj.bias": (self.qkv_proj.bias, "k"),
            "v_proj.weight": (self.qkv_proj.weight, "v"),
            "v_proj.bias": (self.qkv_proj.bias, "v"),
            "query.weight": (self.qkv_proj.weight, "q"),
            "query.bias": (self.qkv_proj.bias, "q"),
            "key.weight": (self.qkv_proj.weight, "k"),
            "key.bias": (self.qkv_proj.bias, "k"),
            "value.weight": (self.qkv_proj.weight, "v"),
            "value.bias": (self.qkv_proj.bias, "v"),
        }
        target = split_map.get(suffix)
        if target is not None:
            param, shard_id = target
            if param is None:
                return ResolvedWeight(skip=True)
            return ResolvedWeight(param=param, shard_id=shard_id)

        direct_map = {
            "qkv_proj.weight": self.qkv_proj.weight,
            "qkv_proj.bias": self.qkv_proj.bias,
            "query_key_value.weight": self.qkv_proj.weight,
            "query_key_value.bias": self.qkv_proj.bias,
            "o_proj.weight": self.o_proj.weight,
            "o_proj.bias": self.o_proj.bias,
            "dense.weight": self.o_proj.weight,
            "dense.bias": self.o_proj.bias,
            "q_norm.weight": getattr(self.q_norm_module, "weight", None),
            "k_norm.weight": getattr(self.k_norm_module, "weight", None),
            "query_layernorm.weight": getattr(self.q_norm_module, "weight", None),
            "key_layernorm.weight": getattr(self.k_norm_module, "weight", None),
        }
        param = direct_map.get(suffix)
        if param is not None:
            return ResolvedWeight(param=param)
        return None


class MergedSwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, *, hidden_act: str = "silu") -> None:
        super().__init__()
        if hidden_act != "silu":
            raise NotImplementedError("MergedSwiGLUMLP currently supports only silu.")
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
        )
        self.gate_up_proj.lora_target_aliases = ("gate_up_proj", "gate_proj", "up_proj", "ff_proj")
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        self.down_proj.lora_target_aliases = ("down_proj", "ff_out")
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(hidden_states)))

    def resolve_checkpoint_weight(self, suffix: str, ctx: LoadContext) -> ResolvedWeight | None:
        split_map = {
            "gate_proj.weight": (self.gate_up_proj.weight, 0),
            "gate_proj.bias": (self.gate_up_proj.bias, 0),
            "ff_proj.weight": (self.gate_up_proj.weight, 0),
            "ff_proj.bias": (self.gate_up_proj.bias, 0),
            "up_proj.weight": (self.gate_up_proj.weight, 1),
            "up_proj.bias": (self.gate_up_proj.bias, 1),
        }
        target = split_map.get(suffix)
        if target is not None:
            param, shard_id = target
            if param is None:
                return ResolvedWeight(skip=True)
            return ResolvedWeight(param=param, shard_id=shard_id)

        direct_map = {
            "gate_up_proj.weight": self.gate_up_proj.weight,
            "gate_up_proj.bias": self.gate_up_proj.bias,
            "down_proj.weight": self.down_proj.weight,
            "down_proj.bias": self.down_proj.bias,
            "ff_out.weight": self.down_proj.weight,
            "ff_out.bias": self.down_proj.bias,
        }
        param = direct_map.get(suffix)
        if param is not None:
            return ResolvedWeight(param=param)
        return None
