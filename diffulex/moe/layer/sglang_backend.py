from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.nn as nn

from diffulex.distributed.parallel_state import fetch_parallel_state
from diffulex.distributed.sglang_backend import ensure_sglang_moe_parallel_state

try:
    from sglang.srt.layers.moe import utils as sglang_moe_utils
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE as SGLangFusedMoE
    from sglang.srt.layers.moe.topk import StandardTopKOutput as SGLangStandardTopKOutput
    from sglang.srt.server_args import (
        get_global_server_args as get_sglang_global_server_args,
        set_global_server_args_for_scheduler as set_sglang_global_server_args_for_scheduler,
    )
except Exception:
    sglang_moe_utils = None
    SGLangFusedMoE = None
    SGLangStandardTopKOutput = None
    get_sglang_global_server_args = None
    set_sglang_global_server_args_for_scheduler = None


@contextmanager
def _temporary_sglang_moe_runner_backend(backend_name: str | None):
    if sglang_moe_utils is None or backend_name is None:
        yield
        return

    previous = sglang_moe_utils.MOE_RUNNER_BACKEND
    try:
        sglang_moe_utils.MOE_RUNNER_BACKEND = sglang_moe_utils.MoeRunnerBackend(backend_name)
        yield
    finally:
        sglang_moe_utils.MOE_RUNNER_BACKEND = previous


def ensure_sglang_server_args() -> None:
    if (
        get_sglang_global_server_args is None
        or set_sglang_global_server_args_for_scheduler is None
    ):
        return
    try:
        get_sglang_global_server_args()
    except Exception:
        set_sglang_global_server_args_for_scheduler(
            SimpleNamespace(
                kt_weight_path=None,
                kt_num_gpu_experts=None,
                kt_cpuinfer=None,
                kt_threadpool_count=None,
                chunked_prefill_size=None,
                kt_method=None,
                kt_max_deferred_experts_per_token=None,
                get_hf_config=lambda: None,
            )
        )


def make_sglang_topk_output(
    *,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    router_logits: torch.Tensor,
):
    if SGLangStandardTopKOutput is None:
        raise RuntimeError("sglang StandardTopKOutput is unavailable in the current environment.")
    return SGLangStandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids.to(torch.int32),
        router_logits=router_logits,
    )


class SGLangTPFusedMoEAdapter(nn.Module):
    def __init__(
        self,
        *,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        hidden_act: str,
        layer_id: int = 0,
        backend_name: str = "triton",
    ) -> None:
        super().__init__()
        if SGLangFusedMoE is None:
            raise RuntimeError("sglang FusedMoE is unavailable in the current environment.")

        ensure_sglang_moe_parallel_state(fetch_parallel_state())
        ensure_sglang_server_args()

        self.backend_name = backend_name
        with _temporary_sglang_moe_runner_backend(backend_name):
            self.layer = SGLangFusedMoE(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                layer_id=layer_id,
                top_k=top_k,
                params_dtype=torch.get_default_dtype(),
                reduce_results=False,
                use_presharded_weights=True,
                activation=hidden_act,
                inplace=False,
            )

    @property
    def w13_weight(self) -> nn.Parameter:
        return self.layer.w13_weight

    @property
    def w2_weight(self) -> nn.Parameter:
        return self.layer.w2_weight

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        topk_output = make_sglang_topk_output(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            router_logits=router_logits,
        )
        with _temporary_sglang_moe_runner_backend(self.backend_name):
            return self.layer(hidden_states, topk_output)


__all__ = [
    "SGLangTPFusedMoEAdapter",
    "ensure_sglang_server_args",
    "make_sglang_topk_output",
]
