import torch

import torch.nn as nn
import torch.nn.functional as F

from diffulex.distributed.parallel_state import fetch_parallel_state
from diffulex.distributed.tp_comm import tp_all_reduce, tp_gather_to_rank0


class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        parallel_state = fetch_parallel_state()
        self.tp_rank = parallel_state.get_tp_rank()
        self.tp_size = parallel_state.get_tp_world_size()
        self.tp_group = parallel_state.get_tp_group()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            y = tp_all_reduce(y, self.tp_group)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        super().__init__(num_embeddings, embedding_dim)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
        self._local_logits_workspace: torch.Tensor | None = None

    def _get_local_logits_workspace(self, x: torch.Tensor) -> torch.Tensor:
        shape = (*x.shape[:-1], self.num_embeddings_per_partition)
        workspace = self._local_logits_workspace
        if (
            workspace is None
            or tuple(workspace.shape) != tuple(shape)
            or workspace.device != x.device
            or workspace.dtype != x.dtype
        ):
            workspace = torch.empty(shape, device=x.device, dtype=x.dtype)
            self._local_logits_workspace = workspace
        return workspace

    def _linear_into_workspace(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            return F.linear(x, self.weight, self.bias)

        logits = self._get_local_logits_workspace(x)
        torch.mm(x, self.weight.t(), out=logits)
        if self.bias is not None:
            logits.add_(self.bias)
        return logits

    def forward(self, x: torch.Tensor):
        logits = self._linear_into_workspace(x)
        if self.tp_size > 1:
            logits = tp_gather_to_rank0(logits, self.tp_group, self.tp_size, self.tp_rank)
        return logits
