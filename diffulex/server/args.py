from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence


def parse_device_ids(value: str | None) -> list[int]:
    if value is None or value.strip() == "":
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


@dataclass
class ServerArgs:
    model: str
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    zmq_command_addr: str = ""
    zmq_event_addr: str = ""

    model_name: str = "dream"
    decoding_strategy: str = "d2f"
    sampling_mode: str = "naive"
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    master_addr: str = "localhost"
    master_port: int = 2333
    distributed_timeout_seconds: int = 600
    device_ids: list[int] | None = None

    block_size: int = 32
    buffer_size: int = 4
    page_size: int = 32
    max_num_batched_tokens: int = 4096
    max_num_reqs: int = 128
    max_model_len: int = 2048
    enable_prefill_cudagraph: bool = True
    prefill_cudagraph_max_len: int = 0
    enable_torch_compile: bool = True
    enable_cudagraph_torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"
    auto_max_nfe_warmup_steps: int = 8
    auto_max_nfe_tpf_floor: float = 1.0
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    attn_impl: str = "triton"
    enable_prefix_caching: bool = True
    kv_cache_layout: str = "unified"
    moe_dispatcher_backend: str = "standard"
    moe_gemm_impl: str = "triton"
    deepep_mode: str = "auto"
    deepep_num_max_dispatch_tokens_per_rank: int = 256
    add_block_threshold: float | None = None
    semi_complete_threshold: float | None = None
    accept_threshold: float | None = None
    remask_threshold: float | None = None

    use_lora: bool = False
    lora_path: str = ""
    pre_merge_lora: bool = False

    def engine_kwargs(self) -> dict:
        return {
            "model_name": self.model_name,
            "decoding_strategy": self.decoding_strategy,
            "sampling_mode": self.sampling_mode,
            "tensor_parallel_size": self.tensor_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "distributed_timeout_seconds": self.distributed_timeout_seconds,
            "device_ids": self.device_ids or [],
            "block_size": self.block_size,
            "buffer_size": self.buffer_size,
            "page_size": self.page_size,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_reqs": self.max_num_reqs,
            "max_model_len": self.max_model_len,
            "enable_prefill_cudagraph": self.enable_prefill_cudagraph,
            "prefill_cudagraph_max_len": self.prefill_cudagraph_max_len,
            "enable_torch_compile": self.enable_torch_compile,
            "enable_cudagraph_torch_compile": self.enable_cudagraph_torch_compile,
            "torch_compile_mode": self.torch_compile_mode,
            "auto_max_nfe_warmup_steps": self.auto_max_nfe_warmup_steps,
            "auto_max_nfe_tpf_floor": self.auto_max_nfe_tpf_floor,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": self.enforce_eager,
            "attn_impl": self.attn_impl,
            "enable_prefix_caching": self.enable_prefix_caching,
            "kv_cache_layout": self.kv_cache_layout,
            "moe_dispatcher_backend": self.moe_dispatcher_backend,
            "moe_gemm_impl": self.moe_gemm_impl,
            "deepep_mode": self.deepep_mode,
            "deepep_num_max_dispatch_tokens_per_rank": self.deepep_num_max_dispatch_tokens_per_rank,
            "decoding_thresholds": {
                "add_block_threshold": 0.1 if self.add_block_threshold is None else self.add_block_threshold,
                "semi_complete_threshold": 0.9
                if self.semi_complete_threshold is None
                else self.semi_complete_threshold,
                "accept_threshold": 0.9 if self.accept_threshold is None else self.accept_threshold,
                "remask_threshold": 0.4 if self.remask_threshold is None else self.remask_threshold,
            },
            "use_lora": self.use_lora,
            "lora_path": self.lora_path,
            "pre_merge_lora": self.pre_merge_lora,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a Diffulex HTTP server")
    parser.add_argument("--model", required=True, help="Path to the local model directory")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--zmq-command-addr", default="", help="Frontend-to-backend ZMQ PUSH/PULL address")
    parser.add_argument("--zmq-event-addr", default="", help="Backend-to-frontend ZMQ PUSH/PULL address")

    parser.add_argument("--model-name", default="dream")
    parser.add_argument("--decoding-strategy", default="d2f")
    parser.add_argument("--sampling-mode", default="naive", choices=["naive", "edit"])
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--master-addr", default="localhost")
    parser.add_argument("--master-port", type=int, default=2333)
    parser.add_argument("--distributed-timeout-seconds", type=int, default=600)
    parser.add_argument("--device-ids", default="", help="Comma-separated logical CUDA device ids")

    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=4)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-reqs", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--disable-prefill-cudagraph", action="store_true")
    parser.add_argument("--prefill-cudagraph-max-len", type=int, default=0)
    parser.add_argument("--disable-torch-compile", action="store_true")
    parser.add_argument("--enable-cudagraph-torch-compile", action="store_true")
    parser.add_argument("--torch-compile-mode", default="reduce-overhead")
    parser.add_argument("--auto-max-nfe-warmup-steps", type=int, default=8)
    parser.add_argument("--auto-max-nfe-tpf-floor", type=float, default=1.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--attn-impl", default="triton", choices=["triton", "naive"])
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--kv-cache-layout", default="unified", choices=["unified", "distinct"])
    parser.add_argument("--moe-dispatcher-backend", default="standard", choices=["standard", "naive", "deepep"])
    parser.add_argument("--moe-gemm-impl", default="triton", choices=["triton", "vllm", "naive"])
    parser.add_argument("--deepep-mode", default="auto", choices=["normal", "low_latency", "auto"])
    parser.add_argument("--deepep-num-max-dispatch-tokens-per-rank", type=int, default=256)
    parser.add_argument("--add-block-threshold", type=float, default=None)
    parser.add_argument("--semi-complete-threshold", type=float, default=None)
    parser.add_argument("--accept-threshold", type=float, default=None)
    parser.add_argument("--remask-threshold", type=float, default=None)

    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--pre-merge-lora", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> ServerArgs:
    ns = build_arg_parser().parse_args(argv)
    return ServerArgs(
        model=ns.model,
        host=ns.host,
        port=ns.port,
        log_level=ns.log_level,
        zmq_command_addr=ns.zmq_command_addr,
        zmq_event_addr=ns.zmq_event_addr,
        model_name=ns.model_name,
        decoding_strategy=ns.decoding_strategy,
        sampling_mode=ns.sampling_mode,
        tensor_parallel_size=ns.tensor_parallel_size,
        data_parallel_size=ns.data_parallel_size,
        master_addr=ns.master_addr,
        master_port=ns.master_port,
        distributed_timeout_seconds=ns.distributed_timeout_seconds,
        device_ids=parse_device_ids(ns.device_ids),
        block_size=ns.block_size,
        buffer_size=ns.buffer_size,
        page_size=ns.page_size,
        max_num_batched_tokens=ns.max_num_batched_tokens,
        max_num_reqs=ns.max_num_reqs,
        max_model_len=ns.max_model_len,
        enable_prefill_cudagraph=not ns.disable_prefill_cudagraph,
        prefill_cudagraph_max_len=ns.prefill_cudagraph_max_len,
        enable_torch_compile=not ns.disable_torch_compile,
        enable_cudagraph_torch_compile=ns.enable_cudagraph_torch_compile,
        torch_compile_mode=ns.torch_compile_mode,
        auto_max_nfe_warmup_steps=ns.auto_max_nfe_warmup_steps,
        auto_max_nfe_tpf_floor=ns.auto_max_nfe_tpf_floor,
        gpu_memory_utilization=ns.gpu_memory_utilization,
        enforce_eager=ns.enforce_eager,
        attn_impl=ns.attn_impl,
        enable_prefix_caching=not ns.disable_prefix_caching,
        kv_cache_layout=ns.kv_cache_layout,
        moe_dispatcher_backend=ns.moe_dispatcher_backend,
        moe_gemm_impl=ns.moe_gemm_impl,
        deepep_mode=ns.deepep_mode,
        deepep_num_max_dispatch_tokens_per_rank=ns.deepep_num_max_dispatch_tokens_per_rank,
        add_block_threshold=ns.add_block_threshold,
        semi_complete_threshold=ns.semi_complete_threshold,
        accept_threshold=ns.accept_threshold,
        remask_threshold=ns.remask_threshold,
        use_lora=ns.use_lora,
        lora_path=ns.lora_path,
        pre_merge_lora=ns.pre_merge_lora,
    )
