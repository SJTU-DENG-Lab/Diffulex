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
    device_ids: list[int] | None = None

    block_size: int = 32
    buffer_size: int = 4
    page_size: int = 32
    max_num_batched_tokens: int = 4096
    max_num_reqs: int = 128
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    kv_cache_layout: str = "unified"
    add_block_threshold: float | None = None
    semi_complete_threshold: float | None = None
    accept_threshold: float | None = None
    remask_threshold: float | None = None

    use_lora: bool = False
    lora_path: str = ""
    pre_merge_lora: bool = False

    def engine_kwargs(self) -> dict:
        if self.data_parallel_size != 1:
            raise ValueError("Phase 1 HTTP serving supports data_parallel_size=1 only")
        return {
            "model_name": self.model_name,
            "decoding_strategy": self.decoding_strategy,
            "sampling_mode": self.sampling_mode,
            "tensor_parallel_size": self.tensor_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "device_ids": self.device_ids or [],
            "block_size": self.block_size,
            "buffer_size": self.buffer_size,
            "page_size": self.page_size,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_reqs": self.max_num_reqs,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": self.enforce_eager,
            "enable_prefix_caching": self.enable_prefix_caching,
            "kv_cache_layout": self.kv_cache_layout,
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
    parser.add_argument("--device-ids", default="", help="Comma-separated logical CUDA device ids")

    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=4)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-reqs", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--kv-cache-layout", default="unified", choices=["unified", "distinct"])
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
        device_ids=parse_device_ids(ns.device_ids),
        block_size=ns.block_size,
        buffer_size=ns.buffer_size,
        page_size=ns.page_size,
        max_num_batched_tokens=ns.max_num_batched_tokens,
        max_num_reqs=ns.max_num_reqs,
        max_model_len=ns.max_model_len,
        gpu_memory_utilization=ns.gpu_memory_utilization,
        enforce_eager=ns.enforce_eager,
        enable_prefix_caching=not ns.disable_prefix_caching,
        kv_cache_layout=ns.kv_cache_layout,
        add_block_threshold=ns.add_block_threshold,
        semi_complete_threshold=ns.semi_complete_threshold,
        accept_threshold=ns.accept_threshold,
        remask_threshold=ns.remask_threshold,
        use_lora=ns.use_lora,
        lora_path=ns.lora_path,
        pre_merge_lora=ns.pre_merge_lora,
    )
