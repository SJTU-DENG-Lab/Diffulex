from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from diffulex.distributed.parallel_state import fetch_parallel_state
from diffulex.distributed.tp_comm import tp_all_reduce


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LoRAMixin:
    """Mixin class to add LoRA support to existing linear layers."""

    def __init_lora__(self, r: int = 0, lora_alpha: float = 1.0, lora_dropout: float = 0.0):
        if r > 0:
            self.r = r
            self.lora_alpha = lora_alpha
            self.scaling = lora_alpha / r

            # Initialize LoRA parameters
            if hasattr(self, "output_size_per_partition"):
                out_features = self.output_size_per_partition
            else:
                out_features = self.output_size

            if hasattr(self, "input_size_per_partition"):
                in_features = self.input_size_per_partition
            else:
                in_features = self.input_size

            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
            self.merged = False

            # Initialize weights
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)
        else:
            self.r = 0
            self.merged = True

    def merge_lora(self):
        """Merge LoRA weights into base weight."""
        if not (hasattr(self, "r") and self.r > 0 and not self.merged):
            return
        # If base weight is missing, we cannot merge in-place. Keep LoRA unmerged and apply via lora_forward.
        weight = getattr(self, "weight", None)
        if weight is None or not hasattr(weight, "data"):
            return
        self.weight.data += self.scaling * torch.mm(self.lora_B, self.lora_A)
        self.merged = True

    def lora_forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        """Apply LoRA forward pass."""
        if not hasattr(self, "r") or self.r == 0 or self.merged:
            return base_output

        lora_out = F.linear(self.lora_dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return base_output + lora_out * self.scaling


class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        parallel_state = fetch_parallel_state()
        self.tp_rank = parallel_state.get_tp_rank()
        self.tp_size = parallel_state.get_tp_world_size()
        self.tp_group = parallel_state.get_tp_group()

    def _forward_base(self, x: torch.Tensor, bias: nn.Parameter | None) -> torch.Tensor:
        return F.linear(x, self.weight, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase, LoRAMixin):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ):
        LinearBase.__init__(self, input_size, output_size, None)
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

        self.__init_lora__(r, lora_alpha, lora_dropout)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self._forward_base(x, self.bias)
        return self.lora_forward(x, base_out)


class ColumnParallelLinear(LinearBase, LoRAMixin):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ):
        LinearBase.__init__(self, input_size, output_size, 0)
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self._forward_out_features = int(self.output_size_per_partition)

        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

        self.__init_lora__(r, lora_alpha, lora_dropout)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self._forward_base(x, self.bias)
        return self.lora_forward(x, base_out)


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ):
        self.output_sizes = output_sizes
        super().__init__(
            input_size,
            sum(output_sizes),
            bias=bias,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | None = None,
    ):
        if loaded_shard_id is None:
            offset = 0
            for shard_id, output_size in enumerate(self.output_sizes):
                shard_weight = loaded_weight.narrow(self.tp_dim, offset, output_size)
                self.weight_loader(param, shard_weight, shard_id)
                offset += output_size
            return
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        parallel_state = fetch_parallel_state()
        tp_size = parallel_state.get_tp_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        self.q_proj_shard_size = self.num_heads * self.head_size
        self.kv_proj_shard_size = self.num_kv_heads * self.head_size
        input_size = hidden_size
        output_size = (self.q_proj_shard_size + 2 * self.kv_proj_shard_size) * tp_size
        self.output_sizes = [
            self.q_proj_shard_size * tp_size,
            self.kv_proj_shard_size * tp_size,
            self.kv_proj_shard_size * tp_size,
        ]
        super().__init__(
            input_size,
            output_size,
            bias,
            r,
            lora_alpha,
            lora_dropout,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ):
        if loaded_shard_id is None:
            q_size = self.total_num_heads * self.head_size
            kv_size = self.total_num_kv_heads * self.head_size
            q, k, v = loaded_weight.split((q_size, kv_size, kv_size), dim=self.tp_dim)
            self.weight_loader(param, q, "q")
            self.weight_loader(param, k, "k")
            self.weight_loader(param, v, "v")
            return
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.q_proj_shard_size
            shard_offset = 0
        else:
            shard_size = self.kv_proj_shard_size
            shard_offset = self.q_proj_shard_size
            if loaded_shard_id == "v":
                shard_offset += self.kv_proj_shard_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        if loaded_shard_id == "q":
            shard_id = self.tp_rank
        else:
            shard_id = self.tp_rank // self.num_kv_head_replicas
        loaded_weight = loaded_weight.chunk(
            self.tp_size if loaded_shard_id == "q" else self.total_num_kv_heads,
            self.tp_dim,
        )[shard_id]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase, LoRAMixin):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ):
        LinearBase.__init__(self, input_size, output_size, 1)
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size

        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_per_partition))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

        self.__init_lora__(r, lora_alpha, lora_dropout)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.tp_rank == 0 else None
        tp_group = self.tp_group
        y = self._forward_base(x, bias)
        if hasattr(self, "r") and self.r > 0 and not self.merged:
            lora_out = F.linear(self.lora_dropout(x), self.lora_A)
            lora_out = F.linear(lora_out, self.lora_B)
            if self.tp_size > 1:
                y = tp_all_reduce(y, tp_group)
                lora_out = tp_all_reduce(lora_out, tp_group)
            return y + lora_out * self.scaling
        if self.tp_size > 1:
            y = tp_all_reduce(y, tp_group)
        return y
