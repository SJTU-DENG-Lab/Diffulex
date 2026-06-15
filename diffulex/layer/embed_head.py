import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os

from diffulex.distributed.parallel_state import fetch_parallel_state
from diffulex.vllm_compat import get_vllm_tp_group


LM_HEAD_FP32 = os.environ.get("DIFFULEX_LM_HEAD_FP32", "0") == "1"
LM_HEAD_FP32_GATHER = LM_HEAD_FP32 or os.environ.get("DIFFULEX_LM_HEAD_FP32_GATHER", "0") == "1"


def _tp_all_reduce(x: torch.Tensor, group) -> torch.Tensor:
    vllm_tp_group = get_vllm_tp_group()
    if vllm_tp_group is not None:
        try:
            return vllm_tp_group.all_reduce(x)
        except Exception:
            pass
    dist.all_reduce(x, group=group)
    return x


def _tp_gather_to_rank0(x: torch.Tensor, group, tp_size: int, tp_rank: int) -> torch.Tensor | None:
    vllm_tp_group = get_vllm_tp_group()
    if vllm_tp_group is not None:
        try:
            return vllm_tp_group.gather(x, dst=0, dim=-1)
        except Exception:
            pass

    gathered = [torch.empty_like(x) for _ in range(tp_size)]
    dist.all_gather(gathered, x, group=group)
    return torch.cat(gathered, -1) if tp_rank == 0 else None


def _tp_all_gather(x: torch.Tensor, group, tp_size: int) -> torch.Tensor:
    gathered = [torch.empty_like(x) for _ in range(tp_size)]
    dist.all_gather(gathered, x, group=group)
    return torch.cat(gathered, -1)


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
            y = _tp_all_reduce(y, self.tp_group)
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
        if LM_HEAD_FP32:
            return F.linear(
                x.to(torch.float32),
                self.weight.to(torch.float32),
                self.bias.to(torch.float32) if self.bias is not None else None,
            ).to(x.dtype)

        if x.dim() != 2:
            return F.linear(x, self.weight, self.bias)

        logits = self._get_local_logits_workspace(x)
        torch.mm(x, self.weight.t(), out=logits)
        if self.bias is not None:
            logits.add_(self.bias)
        return logits

    def forward(self, x: torch.Tensor, gather_all: bool = False):
        logits = self._linear_into_workspace(x)
        if self.tp_size > 1:
            if gather_all:
                if LM_HEAD_FP32_GATHER:
                    logits_dtype = logits.dtype
                    logits = _tp_all_gather(logits.to(torch.float32), self.tp_group, self.tp_size)
                    logits = logits.to(logits_dtype)
                else:
                    logits = _tp_all_gather(logits, self.tp_group, self.tp_size)
            elif LM_HEAD_FP32_GATHER:
                logits_dtype = logits.dtype
                logits = _tp_gather_to_rank0(logits.to(torch.float32), self.tp_group, self.tp_size, self.tp_rank)
                logits = logits.to(logits_dtype) if logits is not None else None
            else:
                logits = _tp_gather_to_rank0(logits, self.tp_group, self.tp_size, self.tp_rank)
        return logits
