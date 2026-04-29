import os

from dataclasses import dataclass, field
from transformers import AutoConfig
from diffulex.distributed.parallel_state import get_world_size
from diffulex.logger import get_logger

logger = get_logger(__name__)
SUPPORTED_PAGE_BLOCK_SIZES = (4, 8, 16, 32)
EDIT_SAMPLING_MODEL_NAMES = {
    "llada2",
    "llada2_moe",
    "llada2_mini",
    "llada2dot1_mini",
    "llada2_mini_dmax",
}
DMAX_MODEL_NAMES = EDIT_SAMPLING_MODEL_NAMES


@dataclass
class DecodingThresholds:
    add_block_threshold: float  # whether add a new block
    semi_complete_threshold: float  # whether unleash the decoding of the next block
    accept_threshold: float  # whether the token should be decoded
    remask_threshold: float = 0.4  # whether a filled token should be remasked
    token_stability_threshold: float = 0.0  # whether decoded tokens are stable enough to add a new block


@dataclass
class Config:
    model: str
    lora_path: str = ""
    model_name: str = "dream"
    decoding_strategy: str = "d2f"  # "d2f", "multi_bd"

    # Sampling Harness
    hf_config: AutoConfig | None = None
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
    sampling_mode: str = "naive"  # "naive" or "edit"
    use_lora: bool = False
    pre_merge_lora: bool = False
    max_num_batched_tokens: int = 4096
    max_num_reqs: int = 128
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.9
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
    distributed_timeout_seconds: int = 600
    shm_name: str = "diffulex_shm"
    device_start: int = 0
    device_ids: list[int] = field(default_factory=lambda: [])

    # CUDA Graph
    enforce_eager: bool = False
    attn_impl: str = "triton"  # "triton" or "naive"
    enable_prefill_cudagraph: bool = True
    enable_full_static_runner: bool = True
    prefill_cudagraph_max_len: int = 0
    enable_torch_compile: bool = True
    enable_cudagraph_torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"

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
            allowed = ", ".join(sorted(DMAX_MODEL_NAMES))
            if self.model_name not in DMAX_MODEL_NAMES:
                raise ValueError(
                    f"decoding_strategy='dmax' is only supported for model_name in {{{allowed}}}, "
                    f"got: {self.model_name}"
                )
            if self.sampling_mode != "edit":
                raise ValueError("decoding_strategy='dmax' requires sampling_mode='edit'.")

    def __post_init__(self):
        if not os.path.isdir(self.model):
            raise ValueError(f"model must be an existing directory, got: {self.model}")

        if self.decoding_strategy == "d2f":
            if not self.multi_block_prefix_full:
                logger.warning("Forcing multi_block_prefix_full=True for decoding_strategy=d2f.")
            if self.enable_prefix_caching:
                logger.warning("Disabling prefix caching for decoding_strategy=d2f.")
            self.multi_block_prefix_full = True
            self.enable_prefix_caching = False
        elif self.decoding_strategy in ("multi_bd", "dmax"):
            if self.multi_block_prefix_full:
                logger.warning(f"Forcing multi_block_prefix_full=False for decoding_strategy={self.decoding_strategy}.")
            self.multi_block_prefix_full = False
            if self.enable_prefix_caching:
                logger.info(f"Enabling prefix caching for decoding_strategy={self.decoding_strategy}.")

        if self.page_size not in SUPPORTED_PAGE_BLOCK_SIZES:
            raise ValueError(
                f"page_size must be one of {SUPPORTED_PAGE_BLOCK_SIZES}, got: {self.page_size}"
            )

        if self.block_size not in SUPPORTED_PAGE_BLOCK_SIZES:
            raise ValueError(
                f"block_size must be one of {SUPPORTED_PAGE_BLOCK_SIZES}, got: {self.block_size}"
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
        if self.attn_impl not in {"triton", "naive"}:
            raise ValueError(
                "attn_impl must be one of {'triton', 'naive'}, "
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

        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
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
        cfg_max_model_len = (
            self.hf_config.max_position_embeddings
            if hasattr(self.hf_config, "max_position_embeddings")
            else self.hf_config.max_sequence_length
        )
        self.max_model_len = min(self.max_model_len, cfg_max_model_len)
        
        if self.max_num_batched_tokens < self.max_model_len:
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
                "remask_threshold",
                "token_stability_threshold",
            ):
                if d.get(key) is not None:
                    self.decoding_thresholds[key] = d[key]
            if "remask_threshold" not in self.decoding_thresholds:
                self.decoding_thresholds["remask_threshold"] = 0.4
            if "token_stability_threshold" not in self.decoding_thresholds:
                self.decoding_thresholds["token_stability_threshold"] = 0.0
            self.decoding_thresholds = DecodingThresholds(**self.decoding_thresholds)
        elif self.decoding_thresholds is None:
            add_block_threshold = d.get("add_block_threshold")
            semi_complete_threshold = d.get("semi_complete_threshold")
            accept_threshold = d.get("accept_threshold")
            remask_threshold = d.get("remask_threshold")
            token_stability_threshold = d.get("token_stability_threshold")
            self.decoding_thresholds = DecodingThresholds(
                add_block_threshold=0.1 if add_block_threshold is None else add_block_threshold,
                semi_complete_threshold=0.9 if semi_complete_threshold is None else semi_complete_threshold,
                accept_threshold=0.9 if accept_threshold is None else accept_threshold,
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
        self.accept_threshold = self.decoding_thresholds.accept_threshold
        self.remask_threshold = self.decoding_thresholds.remask_threshold

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
