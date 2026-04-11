import os

from dataclasses import dataclass, field
from transformers import AutoConfig
from diffulex.logger import get_logger

logger = get_logger(__name__)
SUPPORTED_PAGE_BLOCK_SIZES = (4, 8, 16, 32)


@dataclass
class DecodingThresholds:
    add_block_threshold: float  # whether add a new block
    semi_complete_threshold: float  # whether unleash the decoding of the next block
    decoding_threshold: float  # whether the token should be decoded


@dataclass
class Config:
    model: str
    lora_path: str = ""
    model_name: str = "dream"
    decoding_strategy: str = "d2f"  # "d2f", "multi_bd"

    mask_token_id: int = 151666
    block_size: int = 32
    buffer_size: int = 4
    multi_block_prefix_full: bool = True

    decoding_thresholds: DecodingThresholds | dict | None = None
    # TODO: Should be deprecated in the future
    add_block_threshold: float | None = None
    semi_complete_threshold: float | None = None
    decoding_threshold: float | None = None

    use_lora: bool = False
    pre_merge_lora: bool = False
    max_num_batched_tokens: int = 4096
    max_num_reqs: int = 128
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.9

    data_parallel_size: int = 1
    tensor_parallel_size: int = 2
    master_addr: str = "localhost"
    master_port: int = 2333
    shm_name: str = "diffulex_shm"
    device_start: int = 0
    device_ids: list[int] = field(default_factory=lambda: [])

    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    page_size: int = 32
    enable_prefix_caching: bool = True
    num_pages: int = -1
    k_cache_hdim_split_factor_x: int = 8
    kv_cache_layout: str = "unified"  # "unified" or "distinct"

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
        elif self.decoding_strategy == "multi_bd":
            if self.multi_block_prefix_full:
                logger.warning("Forcing multi_block_prefix_full=False for decoding_strategy=multi_bd.")
            self.multi_block_prefix_full = False
            if self.enable_prefix_caching:
                logger.info("Enabling prefix caching for decoding_strategy=multi_bd.")

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

        if not (isinstance(self.master_port, int) and 0 < self.master_port < 65536):
            raise ValueError(
                "master_port must be an int in (0, 65536), "
                f"got: {self.master_port}"
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

        if not self.device_ids:
            import torch

            # When CUDA_VISIBLE_DEVICES is set, PyTorch maps physical devices to logical device 0, 1, ...
            # So we should use logical device indices (0, 1, ...) instead of physical device IDs
            self.device_ids = list(range(torch.cuda.device_count()))
            logger.info(f"Using CUDA devices: {self.device_ids}")

        # Build decoding_thresholds: dict or flat keys -> DecodingThresholds
        d = self.__dict__
        if isinstance(self.decoding_thresholds, dict):
            self.decoding_thresholds = DecodingThresholds(**self.decoding_thresholds)
        elif self.decoding_thresholds is None:
            add_block_threshold = d.get("add_block_threshold")
            semi_complete_threshold = d.get("semi_complete_threshold")
            decoding_threshold = d.get("decoding_threshold")
            self.decoding_thresholds = DecodingThresholds(
                add_block_threshold=0.1 if add_block_threshold is None else add_block_threshold,
                semi_complete_threshold=0.9 if semi_complete_threshold is None else semi_complete_threshold,
                decoding_threshold=0.9 if decoding_threshold is None else decoding_threshold,
            )

    @property
    def kv_cache_page_size(self) -> int:
        """Alias for page_size, used by engine/model_runner/tp_worker."""
        return self.page_size

    # TODO: Should be deprecated in the future
    @property
    def accept_threshold(self) -> float:
        return self.decoding_thresholds.decoding_threshold

    @property
    def add_new_block_threshold(self) -> float:
        return self.decoding_thresholds.add_block_threshold

    @property
    def complete_threshold(self) -> float:
        return self.decoding_thresholds.semi_complete_threshold
