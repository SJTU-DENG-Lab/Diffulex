from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from diffulex.profiling import ProfilerConfig


@dataclass
class DecodingThresholds:
    add_block_threshold: float
    semi_complete_threshold: float
    accept_threshold: float
    edit_threshold: float = 0.0
    remask_threshold: float = 0.4
    token_stability_threshold: float = 0.0


@dataclass
class ModelLoadConfig:
    model: str
    model_name: str
    hf_config: Any | None = None
    tokenizer_vocab_size: int | None = None
    eos: int = -1
    mask_token_id: int = 151666
    use_lora: bool = False
    lora_path: str = ""
    pre_merge_lora: bool = False


@dataclass
class DecodingConfig:
    strategy: str
    sampling_mode: str
    block_size: int
    buffer_size: int
    multi_block_prefix_full: bool
    thresholds: DecodingThresholds
    max_post_edit_steps: int = 16
    penalty_lambda: float = 0.0
    enable_vectorized_sampler: bool = False
    enable_vectorized_sampler_compile: bool = False


@dataclass
class SchedulerConfig:
    max_num_batched_tokens: int = 4096
    max_num_reqs: int = 128
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.9
    auto_max_nfe_warmup_steps: int = 8
    auto_max_nfe_tpf_floor: float = 1.0


@dataclass
class ParallelConfig:
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    master_addr: str = "localhost"
    master_port: int = 2333
    distributed_backend: str = "nccl"
    distributed_timeout_seconds: int = 600
    shm_name: str = "diffulex_shm"
    device_start: int = 0
    device_ids: list[int] = field(default_factory=list)


@dataclass
class KernelConfig:
    enforce_eager: bool = False
    attn_impl: str = "triton_grouped"
    enable_prefill_cudagraph: bool = False
    enable_full_static_runner: bool = True
    prefill_cudagraph_max_len: int = 0
    enable_torch_compile: bool = True
    enable_cudagraph_torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"
    enable_vllm_layers: bool = True
    moe_dispatcher_backend: str = "standard"
    moe_gemm_impl: str = "triton"
    deepep_mode: str = "auto"
    deepep_num_max_dispatch_tokens_per_rank: int = 256


@dataclass
class CacheConfig:
    page_size: int = 32
    enable_prefix_caching: bool = True
    num_pages: int = -1
    k_cache_hdim_split_factor_x: int = 8
    kv_cache_layout: str = "unified"


@dataclass
class TokenMergeConfig:
    mode: str = "dmax_topk"
    top_k: int = 1
    renormalize: bool = True
    weight: float = 1.0
    dmax_sampler_fast_path: bool = True
    dmax_force_prefill_active: bool = False
    enable_vectorized_sampler: bool = False
    enable_vectorized_sampler_compile: bool = False


@dataclass
class RuntimeConfig:
    model: ModelLoadConfig
    decoding: DecodingConfig
    scheduler: SchedulerConfig
    parallel: ParallelConfig
    kernel: KernelConfig
    cache: CacheConfig
    token_merge: TokenMergeConfig
    profiler: ProfilerConfig
    strategy: object | None = None
