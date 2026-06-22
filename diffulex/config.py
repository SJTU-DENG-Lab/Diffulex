import os
import json

from dataclasses import dataclass, field
from typing import Any

from diffulex.config_types import (
    CacheConfig,
    DecodingConfig,
    DecodingThresholds,
    KernelConfig,
    ModelLoadConfig,
    ParallelConfig,
    RuntimeConfig,
    SchedulerConfig,
    TokenMergeConfig,
)
from diffulex.distributed.parallel_state import get_world_size
from diffulex.engine.strategy_config_registry import StrategyConfigRegistry
from diffulex.logger import get_logger
from diffulex.hf_config_registry import HFConfigRegistry
from diffulex.profiling import ProfilerConfig

logger = get_logger(__name__)
SUPPORTED_PAGE_BLOCK_SIZES = (4, 8, 16, 32)
GEMMA_BLOCK_SIZE = 256
EDIT_SAMPLING_MODEL_NAMES = {
    "llada2",
    "llada2_moe",
    "llada2_mini",
    "llada2dot1_mini",
    "llada2_mini_dmax",
}
DIFFUSION_GEMMA_MODEL_NAMES = {"diffusion_gemma"}

def _token_content(token) -> str | None:
    if isinstance(token, dict):
        return token.get("content")
    if isinstance(token, str):
        return token
    return None


def _load_tokenizer_mask_token_id(model: str) -> int | None:
    special_tokens_path = os.path.join(model, "special_tokens_map.json")
    tokenizer_config_path = os.path.join(model, "tokenizer_config.json")
    added_tokens_path = os.path.join(model, "added_tokens.json")
    try:
        with open(special_tokens_path, "r", encoding="utf-8") as f:
            mask_token = _token_content(json.load(f).get("mask_token"))
    except Exception:
        mask_token = None
    if not mask_token:
        return None

    try:
        with open(added_tokens_path, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
        token_id = token_to_id.get(mask_token)
        if token_id is not None:
            return int(token_id)
    except Exception:
        pass

    try:
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            added_decoder = json.load(f).get("added_tokens_decoder", {})
        for token_id, token_info in added_decoder.items():
            if _token_content(token_info) == mask_token:
                return int(token_id)
    except Exception:
        return None
    return None


@dataclass
class Config:
    model: str
    lora_path: str = ""
    model_name: str = "dream"
    decoding_strategy: str = "d2f"  # "d2f", "multi_bd", "dmax", or "diffusion_gemma"

    # Sampling Harness
    hf_config: Any | None = None
    tokenizer_vocab_size: int | None = None
    eos: int = -1
    mask_token_id: int = 151666
    block_size: int = 32
    buffer_size: int = 4
    multi_block_prefix_full: bool = True
    token_merge_mode: str = "dmax_topk"  # "dmax_topk" or "iter_smooth_topk"
    token_merge_top_k: int = 1
    token_merge_renormalize: bool = True
    token_merge_weight: float = 1.0
    dmax_sampler_fast_path: bool = True
    dmax_force_prefill_active: bool = False
    enable_vectorized_sampler: bool = False
    enable_vectorized_sampler_compile: bool = False
    sampling_mode: str = "naive"  # "naive" or "edit"
    max_post_edit_steps: int = 16  # max refinement steps after all masks filled (JointThreshold)
    penalty_lambda: float = 0.0  # repetition penalty coefficient (JointThreshold)
    use_lora: bool = False
    pre_merge_lora: bool = False
    max_num_batched_tokens: int = 4096
    max_num_reqs: int = 128
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.9
    skip_warmup: bool = False
    decoding_thresholds: DecodingThresholds | dict | None = None
    # TODO: Should be deprecated in the future
    add_block_threshold: float | None = None
    semi_complete_threshold: float | None = None
    accept_threshold: float | None = None
    remask_threshold: float | None = None
    token_stability_threshold: float | None = None
    # Truncation
    auto_max_nfe_warmup_steps: int = 8
    auto_max_nfe_tpf_floor: float = 1.0

    # Parallelism
    data_parallel_size: int = 1
    tensor_parallel_size: int = 2
    expert_parallel_size: int = 1
    master_addr: str = "localhost"
    master_port: int = 2333
    distributed_backend: str = "nccl"
    distributed_timeout_seconds: int = 600
    shm_name: str = "diffulex_shm"
    device_start: int = 0
    device_ids: list[int] = field(default_factory=lambda: [])

    # CUDA Graph
    enforce_eager: bool = False
    attn_impl: str = "triton_grouped"  # "triton_grouped", "triton", or "naive"
    enable_prefill_cudagraph: bool = False
    enable_full_static_runner: bool = True
    prefill_cudagraph_max_len: int = 0
    enable_torch_compile: bool = True
    enable_cudagraph_torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"
    enable_vllm_layers: bool = True

    # MoE
    moe_dispatcher_backend: str = "standard"  # "standard", "naive", or "deepep"
    moe_gemm_impl: str = "triton"  # "triton", "vllm", or "naive"
    deepep_mode: str = "auto"  # "normal", "low_latency", or "auto"
    deepep_num_max_dispatch_tokens_per_rank: int = 256

    # KV Cache Page Table
    page_size: int = 32
    enable_prefix_caching: bool = True
    num_pages: int = -1
    k_cache_hdim_split_factor_x: int = 8
    kv_cache_layout: str = "unified"  # "unified" or "distinct"

    # DiffusionGemma block sampler controls. These are only used when
    # model_name == "diffusion_gemma".
    diffusion_gemma_max_denoising_steps: int = 48
    diffusion_gemma_stability_threshold: int = 2
    diffusion_gemma_t_min: float = 0.0
    diffusion_gemma_t_max: float = 1.0
    diffusion_gemma_confidence_threshold: float = 0.1
    diffusion_gemma_entropy_bound: float = 1.0

    # Profiling
    profiler_config: ProfilerConfig | dict | None = None

    def _validate_sampling_mode(self) -> None:
        if self.sampling_mode == "edit" and self.model_name not in EDIT_SAMPLING_MODEL_NAMES:
            allowed = ", ".join(sorted(EDIT_SAMPLING_MODEL_NAMES))
            raise ValueError(
                f"sampling_mode='edit' is only supported for model_name in {{{allowed}}}, "
                f"got: {self.model_name}"
            )

        if self.model_name == "llada2dot1_mini" and self.sampling_mode != "edit":
            raise ValueError("model_name='llada2dot1_mini' requires sampling_mode='edit'.")

        if self.decoding_strategy == "dmax":
            if self.sampling_mode != "edit":
                raise ValueError("decoding_strategy='dmax' requires sampling_mode='edit'.")

    @property
    def is_diffusion_gemma(self) -> bool:
        return self.model_name in DIFFUSION_GEMMA_MODEL_NAMES

    def __post_init__(self):
        if not os.path.isdir(self.model):
            raise ValueError(f"model must be an existing directory, got: {self.model}")
        self.profiler_config = ProfilerConfig.from_value(self.profiler_config)

        if self.is_diffusion_gemma:
            self.decoding_strategy = "diffusion_gemma"

        self.strategy = StrategyConfigRegistry.normalize(self)

        supported_page_block_sizes = (
            SUPPORTED_PAGE_BLOCK_SIZES + (GEMMA_BLOCK_SIZE,)
            if self.is_diffusion_gemma
            else SUPPORTED_PAGE_BLOCK_SIZES
        )

        if self.page_size not in supported_page_block_sizes:
            raise ValueError(
                f"page_size must be one of {supported_page_block_sizes}, got: {self.page_size}"
            )

        if self.block_size not in supported_page_block_sizes:
            raise ValueError(
                f"block_size must be one of {supported_page_block_sizes}, got: {self.block_size}"
            )

        if self.block_size > self.page_size:
            raise ValueError(
                "block_size must be <= page_size, "
                f"got: block_size={self.block_size}, page_size={self.page_size}"
            )

        if not 1 <= self.tensor_parallel_size <= 8:
            raise ValueError(
                "tensor_parallel_size must be in [1, 8], "
                f"got: {self.tensor_parallel_size}"
            )

        if not 1 <= self.data_parallel_size <= 1024:
            raise ValueError(
                "data_parallel_size must be in [1, 1024], "
                f"got: {self.data_parallel_size}"
            )

        if not 1 <= self.expert_parallel_size <= 32768:
            raise ValueError(
                "expert_parallel_size must be in [1, 32768], "
                f"got: {self.expert_parallel_size}"
            )


        if self.token_merge_top_k <= 0:
            raise ValueError(f"token_merge_top_k must be positive, got: {self.token_merge_top_k}")
        
        if self.token_merge_mode not in ("dmax_topk", "iter_smooth_topk"):
            raise ValueError(
                "token_merge_mode must be one of {'dmax_topk', 'iter_smooth_topk'}, "
                f"got: {self.token_merge_mode}"
            )
            
        if self.token_merge_weight < 0:
            raise ValueError(f"token_merge_weight must be non-negative, got: {self.token_merge_weight}")
        
        if self.sampling_mode not in ("naive", "edit"):
            raise ValueError(
                "sampling_mode must be one of {'naive', 'edit'}, "
                f"got: {self.sampling_mode}"
            )
        self._validate_sampling_mode()
        if self.kv_cache_layout not in {"unified", "distinct"}:
            raise ValueError(
                "kv_cache_layout must be one of {'unified', 'distinct'}, "
                f"got: {self.kv_cache_layout}"
            )
        if self.attn_impl not in {"triton", "triton_grouped", "naive"}:
            raise ValueError(
                "attn_impl must be one of {'triton', 'triton_grouped', 'naive'}, "
                f"got: {self.attn_impl}"
            )
        if self.moe_dispatcher_backend not in {"standard", "naive", "deepep"}:
            raise ValueError(
                "moe_dispatcher_backend must be one of {'standard', 'naive', 'deepep'}, "
                f"got: {self.moe_dispatcher_backend}"
            )
        if self.moe_gemm_impl not in {"triton", "vllm", "vllm_modular", "naive"}:
            raise ValueError(
                "moe_gemm_impl must be one of {'triton', 'vllm', 'vllm_modular', 'naive'}, "
                f"got: {self.moe_gemm_impl}"
            )
        if self.deepep_mode not in {"normal", "low_latency", "auto"}:
            raise ValueError(
                "deepep_mode must be one of {'normal', 'low_latency', 'auto'}, "
                f"got: {self.deepep_mode}"
            )
        if self.deepep_num_max_dispatch_tokens_per_rank <= 0:
            raise ValueError(
                "deepep_num_max_dispatch_tokens_per_rank must be positive, "
                f"got: {self.deepep_num_max_dispatch_tokens_per_rank}"
            )
        if self.expert_parallel_size != 1 or self.moe_dispatcher_backend != "standard":
            raise ValueError(
                "MoE A2A backends are currently unsupported. Require expert_parallel_size == 1 "
                "and moe_dispatcher_backend == 'standard', "
                f"got expert_parallel_size={self.expert_parallel_size}, "
                f"moe_dispatcher_backend={self.moe_dispatcher_backend}."
            )
            
        if not (isinstance(self.master_port, int) and 0 < self.master_port < 65536):
            raise ValueError(
                "master_port must be an int in (0, 65536), "
                f"got: {self.master_port}"
            )

        if not (
            isinstance(self.distributed_timeout_seconds, int)
            and self.distributed_timeout_seconds > 0
        ):
            raise ValueError(
                "distributed_timeout_seconds must be a positive int, "
                f"got: {self.distributed_timeout_seconds}"
            )
            
            
        if not (isinstance(self.device_start, int) and self.device_start >= 0):
            raise ValueError(
                "device_start must be a non-negative int, "
                f"got: {self.device_start}"
            )

        # LoRA validation
        if self.use_lora:
            if not self.lora_path:
                raise ValueError("lora_path must be provided when use_lora is True")

            if not os.path.exists(self.lora_path):
                logger.warning(f"LoRA path {self.lora_path} does not exist")

        loaded_hf_config = self.hf_config is None
        if loaded_hf_config:
            self.hf_config = HFConfigRegistry.load(self.model)
        default_mask_token_id = Config.__dataclass_fields__["mask_token_id"].default
        hf_mask_token_id = getattr(self.hf_config, "mask_token_id", None)
        if self.mask_token_id == default_mask_token_id and hf_mask_token_id is not None:
            self.mask_token_id = int(hf_mask_token_id)
        elif self.mask_token_id == default_mask_token_id:
            tokenizer_mask_token_id = _load_tokenizer_mask_token_id(self.model)
            if tokenizer_mask_token_id is not None:
                self.mask_token_id = int(tokenizer_mask_token_id)
        for name in (
            "attn_impl",
            "moe_dispatcher_backend",
            "moe_gemm_impl",
            "deepep_mode",
            "deepep_num_max_dispatch_tokens_per_rank",
            "expert_parallel_size",
            "tensor_parallel_size",
            "data_parallel_size",
            "mask_token_id",
        ):
            setattr(self.hf_config, name, getattr(self, name))
            text_config = getattr(self.hf_config, "text_config", None)
            if text_config is not None:
                setattr(text_config, name, getattr(self, name))
        HFConfigRegistry.postprocess(self.hf_config, self, None if loaded_hf_config else {})
        text_config = getattr(self.hf_config, "text_config", None)
        cfg_max_model_len = getattr(
            self.hf_config,
            "max_position_embeddings",
            getattr(self.hf_config, "max_sequence_length", None),
        )
        if cfg_max_model_len is None and text_config is not None:
            cfg_max_model_len = getattr(
                text_config,
                "max_position_embeddings",
                getattr(text_config, "max_sequence_length", None),
            )
        if cfg_max_model_len is None:
            raise AttributeError(f"Cannot determine max model length from config: {type(self.hf_config)}")
        self.max_model_len = min(self.max_model_len, cfg_max_model_len)
        
        if self.max_num_batched_tokens < self.max_model_len and not self.is_diffusion_gemma:
            raise ValueError(
                "max_num_batched_tokens must be >= max_model_len after HF config clamp, "
                f"got max_num_batched_tokens={self.max_num_batched_tokens}, "
                f"max_model_len={self.max_model_len}"
            )
        if self.prefill_cudagraph_max_len < 0:
            raise ValueError(
                "prefill_cudagraph_max_len must be non-negative, "
                f"got: {self.prefill_cudagraph_max_len}"
            )
        if self.auto_max_nfe_warmup_steps <= 0:
            raise ValueError(
                "auto_max_nfe_warmup_steps must be positive, "
                f"got: {self.auto_max_nfe_warmup_steps}"
            )
        if self.auto_max_nfe_tpf_floor <= 0:
            raise ValueError(
                "auto_max_nfe_tpf_floor must be positive, "
                f"got: {self.auto_max_nfe_tpf_floor}"
            )

        if not self.device_ids:
            import torch
            # When CUDA_VISIBLE_DEVICES is set, PyTorch maps physical devices to logical device 0, 1, ...
            # So we should use logical device indices (0, 1, ...) instead of physical device IDs
            self.device_ids = list(range(torch.cuda.device_count()))
            logger.info(f"Using CUDA devices: {self.device_ids}")

        required_world_size = get_world_size(
            self.tensor_parallel_size,
            self.expert_parallel_size,
            self.data_parallel_size,
        )
        if len(self.device_ids) < required_world_size:
            raise ValueError(
                "Requested parallel world size exceeds available CUDA devices, "
                f"required={required_world_size}, device_ids={self.device_ids}."
            )

        # Build decoding_thresholds: dict or flat keys -> DecodingThresholds
        d = self.__dict__
        if isinstance(self.decoding_thresholds, dict):
            for key in (
                "add_block_threshold",
                "semi_complete_threshold",
                "accept_threshold",
                "edit_threshold",
                "remask_threshold",
                "token_stability_threshold",
            ):
                if d.get(key) is not None:
                    self.decoding_thresholds[key] = d[key]
            if "edit_threshold" not in self.decoding_thresholds:
                self.decoding_thresholds["edit_threshold"] = 0.0
            if "remask_threshold" not in self.decoding_thresholds:
                self.decoding_thresholds["remask_threshold"] = 0.4
            if "token_stability_threshold" not in self.decoding_thresholds:
                self.decoding_thresholds["token_stability_threshold"] = 0.0
            self.decoding_thresholds = DecodingThresholds(**self.decoding_thresholds)
        elif self.decoding_thresholds is None:
            add_block_threshold = d.get("add_block_threshold")
            semi_complete_threshold = d.get("semi_complete_threshold")
            accept_threshold = d.get("accept_threshold")
            edit_threshold = d.get("edit_threshold")
            remask_threshold = d.get("remask_threshold")
            token_stability_threshold = d.get("token_stability_threshold")
            self.decoding_thresholds = DecodingThresholds(
                add_block_threshold=0.1 if add_block_threshold is None else add_block_threshold,
                semi_complete_threshold=0.9 if semi_complete_threshold is None else semi_complete_threshold,
                accept_threshold=0.9 if accept_threshold is None else accept_threshold,
                edit_threshold=0.0 if edit_threshold is None else edit_threshold,
                remask_threshold=0.4 if remask_threshold is None else remask_threshold,
                token_stability_threshold=0.0
                if token_stability_threshold is None
                else token_stability_threshold,
            )

        if not 0.0 <= self.decoding_thresholds.accept_threshold <= 1.0:
            raise ValueError(
                "decoding_thresholds.accept_threshold must be in [0, 1], "
                f"got: {self.decoding_thresholds.accept_threshold}"
            )
        if not 0.0 <= self.decoding_thresholds.edit_threshold <= 1.0:
            raise ValueError(
                "decoding_thresholds.edit_threshold must be in [0, 1], "
                f"got: {self.decoding_thresholds.edit_threshold}"
            )
        if not 0.0 <= self.decoding_thresholds.remask_threshold <= 1.0:
            raise ValueError(
                "decoding_thresholds.remask_threshold must be in [0, 1], "
                f"got: {self.decoding_thresholds.remask_threshold}"
            )
        if not 0.0 <= self.decoding_thresholds.token_stability_threshold <= 1.0:
            raise ValueError(
                "decoding_thresholds.token_stability_threshold must be in [0, 1], "
                f"got: {self.decoding_thresholds.token_stability_threshold}"
            )
        if not isinstance(self.max_post_edit_steps, int) or self.max_post_edit_steps < 0:
            raise ValueError(
                "max_post_edit_steps must be a non-negative int, "
                f"got: {self.max_post_edit_steps}"
            )
        if not isinstance(self.penalty_lambda, (int, float)) or self.penalty_lambda < 0:
            raise ValueError(
                "penalty_lambda must be a non-negative float, "
                f"got: {self.penalty_lambda}"
            )
        self.accept_threshold = self.decoding_thresholds.accept_threshold
        self.remask_threshold = self.decoding_thresholds.remask_threshold
        self._build_runtime_config()

    def _build_runtime_config(self) -> None:
        self.model_config = ModelLoadConfig(
            model=self.model,
            model_name=self.model_name,
            hf_config=self.hf_config,
            tokenizer_vocab_size=self.tokenizer_vocab_size,
            eos=self.eos,
            mask_token_id=self.mask_token_id,
            use_lora=self.use_lora,
            lora_path=self.lora_path,
            pre_merge_lora=self.pre_merge_lora,
        )
        self.decoding_config = DecodingConfig(
            strategy=self.decoding_strategy,
            sampling_mode=self.sampling_mode,
            block_size=self.block_size,
            buffer_size=self.buffer_size,
            multi_block_prefix_full=self.multi_block_prefix_full,
            thresholds=self.decoding_thresholds,
            max_post_edit_steps=self.max_post_edit_steps,
            penalty_lambda=self.penalty_lambda,
            enable_vectorized_sampler=self.enable_vectorized_sampler,
            enable_vectorized_sampler_compile=self.enable_vectorized_sampler_compile,
        )
        self.scheduler_config = SchedulerConfig(
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            auto_max_nfe_warmup_steps=self.auto_max_nfe_warmup_steps,
            auto_max_nfe_tpf_floor=self.auto_max_nfe_tpf_floor,
        )
        self.parallel_config = ParallelConfig(
            data_parallel_size=self.data_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
            expert_parallel_size=self.expert_parallel_size,
            master_addr=self.master_addr,
            master_port=self.master_port,
            distributed_backend=self.distributed_backend,
            distributed_timeout_seconds=self.distributed_timeout_seconds,
            shm_name=self.shm_name,
            device_start=self.device_start,
            device_ids=list(self.device_ids),
        )
        self.kernel_config = KernelConfig(
            enforce_eager=self.enforce_eager,
            attn_impl=self.attn_impl,
            enable_prefill_cudagraph=self.enable_prefill_cudagraph,
            enable_full_static_runner=self.enable_full_static_runner,
            prefill_cudagraph_max_len=self.prefill_cudagraph_max_len,
            enable_torch_compile=self.enable_torch_compile,
            enable_cudagraph_torch_compile=self.enable_cudagraph_torch_compile,
            torch_compile_mode=self.torch_compile_mode,
            enable_vllm_layers=self.enable_vllm_layers,
            moe_dispatcher_backend=self.moe_dispatcher_backend,
            moe_gemm_impl=self.moe_gemm_impl,
            deepep_mode=self.deepep_mode,
            deepep_num_max_dispatch_tokens_per_rank=self.deepep_num_max_dispatch_tokens_per_rank,
        )
        self.cache_config = CacheConfig(
            page_size=self.page_size,
            enable_prefix_caching=self.enable_prefix_caching,
            num_pages=self.num_pages,
            k_cache_hdim_split_factor_x=self.k_cache_hdim_split_factor_x,
            kv_cache_layout=self.kv_cache_layout,
        )
        self.token_merge_config = TokenMergeConfig(
            mode=self.token_merge_mode,
            top_k=self.token_merge_top_k,
            renormalize=self.token_merge_renormalize,
            weight=self.token_merge_weight,
            dmax_sampler_fast_path=self.dmax_sampler_fast_path,
            dmax_force_prefill_active=self.dmax_force_prefill_active,
            enable_vectorized_sampler=self.enable_vectorized_sampler,
            enable_vectorized_sampler_compile=self.enable_vectorized_sampler_compile,
        )
        self.runtime = RuntimeConfig(
            model=self.model_config,
            decoding=self.decoding_config,
            scheduler=self.scheduler_config,
            parallel=self.parallel_config,
            kernel=self.kernel_config,
            cache=self.cache_config,
            token_merge=self.token_merge_config,
            profiler=self.profiler_config,
            strategy=self.strategy,
        )

    @property
    def kv_cache_page_size(self) -> int:
        """Alias for page_size, used by engine/model_runner."""
        return self.page_size

    @property
    def add_new_block_threshold(self) -> float:
        return self.decoding_thresholds.add_block_threshold

    @property
    def complete_threshold(self) -> float:
        return self.decoding_thresholds.semi_complete_threshold
