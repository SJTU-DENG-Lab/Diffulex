"""
Benchmark Configuration - Configuration management with separated engine and eval configs
"""

from __future__ import annotations

import base64
import json

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml

from diffulex.config import Config as DiffulexConfig


MODEL_ARG_COMPLEX_PREFIX = "b64json:"
DEFAULT_DECODING_THRESHOLDS = {
    "add_block_threshold": 0.1,
    "semi_complete_threshold": 0.9,
    "accept_threshold": 0.9,
    "remask_threshold": 0.4,
}
FLAT_THRESHOLD_KEYS = (
    "add_block_threshold",
    "semi_complete_threshold",
    "accept_threshold",
    "remask_threshold",
)


def diffulex_core_engine_fields() -> set[str]:
    """Diffulex Config fields that can be forwarded from benchmark config."""
    return {
        name
        for name in DiffulexConfig.__dataclass_fields__.keys()
        if name not in {"model", "hf_config"}
    }


CORE_ENGINE_FIELDS = diffulex_core_engine_fields()


def normalize_engine_input_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Apply compatibility aliases for engine config input."""
    d = dict(config_dict)
    if "block_size" not in d and "diffusion_block_size" in d:
        d["block_size"] = d.pop("diffusion_block_size")
    return d


def encode_model_arg_value(value: Any) -> Any:
    """Encode complex values so lm-eval model_args can round-trip them safely."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if "," not in value and not value.startswith(MODEL_ARG_COMPLEX_PREFIX):
            return value
        payload = json.dumps(value, ensure_ascii=False)
    else:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    token = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")
    return f"{MODEL_ARG_COMPLEX_PREFIX}{token}"


def decode_model_arg_value(value: Any) -> Any:
    """Decode values produced by encode_model_arg_value()."""
    if not isinstance(value, str) or not value.startswith(MODEL_ARG_COMPLEX_PREFIX):
        return value
    payload = value[len(MODEL_ARG_COMPLEX_PREFIX) :]
    raw = base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8")
    return json.loads(raw)


def parse_engine_arg_override(value: str) -> Any:
    """Parse CLI --engine-arg values using YAML scalar/list/dict semantics."""
    return yaml.safe_load(value)


def extract_diffulex_engine_kwargs(source: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only Diffulex Config kwargs and normalize defaults/aliases."""
    normalized = normalize_engine_input_dict(source)
    kwargs = {k: v for k, v in normalized.items() if k in CORE_ENGINE_FIELDS and v is not None}

    strategy = kwargs.get("decoding_strategy")
    if strategy in ("multi_block_diffusion", "block_diffusion", "fast_dllm"):
        kwargs["decoding_strategy"] = "multi_bd"

    if not kwargs.get("use_lora", False):
        kwargs["lora_path"] = ""

    if kwargs.get("decoding_thresholds") is None and not any(kwargs.get(k) is not None for k in FLAT_THRESHOLD_KEYS):
        kwargs["decoding_thresholds"] = dict(DEFAULT_DECODING_THRESHOLDS)

    return kwargs


@dataclass
class EngineConfig:
    """
    Engine configuration - Parameters for Diffulex engine initialization
    """

    # Model and weights
    model_path: str
    tokenizer_path: Optional[str] = None
    model_name: str = "dream"  # Options: dream, sdar, fast_dllm_v2, llada
    decoding_strategy: str = "d2f"  # Options: d2f, multi_bd
    sampling_mode: str = "naive"  # Options: naive, edit
    mask_token_id: int = 151666

    # LoRA configuration
    use_lora: bool = False
    lora_path: str = ""
    pre_merge_lora: bool = True  # Merge LoRA into base at load to avoid per-forward LoRA compute

    # Parallelism configuration
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    expert_parallel_size: int = 1

    # Memory and capacity configuration
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 2048
    max_num_batched_tokens: int = 4096
    max_num_reqs: int = 128

    # Engine behavior configuration
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    kv_cache_layout: str = "unified"  # Options: unified, distinct
    page_size: int = 32
    token_merge_mode: str = "dmax_topk"
    token_merge_top_k: int = 1
    token_merge_renormalize: bool = True
    token_merge_weight: float = 1.0

    # D2F/MultiBD-specific configuration
    decoding_thresholds: Optional[Dict[str, float]] = (
        None  # {add_block_threshold, semi_complete_threshold, accept_threshold, remask_threshold}
    )
    block_size: int = 32  # Aligned with diffulex.config.Config.block_size
    buffer_size: int = 4
    multi_block_prefix_full: bool = True
    extra_engine_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def explicit_field_names(cls) -> set[str]:
        return {
            f.name
            for f in cls.__dataclass_fields__.values()
            if f.init and f.name != "extra_engine_kwargs"
        }

    @classmethod
    def accepted_input_fields(cls) -> set[str]:
        return cls.explicit_field_names() | CORE_ENGINE_FIELDS | {"diffusion_block_size"}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EngineConfig":
        """Create engine configuration from dictionary while preserving extra core config fields."""
        d = normalize_engine_input_dict(config_dict)
        valid = cls.explicit_field_names()
        filtered = {k: v for k, v in d.items() if k in valid}
        engine = cls(**filtered)
        engine.extra_engine_kwargs = {
            k: v
            for k, v in d.items()
            if k not in valid and k in CORE_ENGINE_FIELDS
        }
        return engine

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
            if field.name != "extra_engine_kwargs"
        }
        data.update(self.extra_engine_kwargs)
        return data

    def apply_updates(self, updates: Dict[str, Any]) -> None:
        """Apply engine updates, preserving unknown-but-core fields for future configs."""
        normalized = normalize_engine_input_dict(updates)
        valid = self.explicit_field_names()
        for key, value in normalized.items():
            if key in valid:
                setattr(self, key, value)
            elif key in CORE_ENGINE_FIELDS:
                self.extra_engine_kwargs[key] = value

    def get_diffulex_kwargs(self) -> Dict[str, Any]:
        """Get arguments to pass to Diffulex engine (aligned with diffulex.config.Config)."""
        return extract_diffulex_engine_kwargs(self.to_dict())


@dataclass
class EvalConfig:
    """
    Evaluation configuration - Parameters for benchmark evaluation
    """

    # Task/Dataset configuration (lm-eval task name; use bundled * _diffulex tasks for offline JSON)
    dataset_name: str = "gsm8k_diffulex"
    dataset_split: str = "test"
    dataset_limit: Optional[int] = None
    # Directory of custom task YAMLs for lm-eval (--include_path). None → diffulex_bench/tasks next to main.
    include_path: Optional[str] = None
    # Optional JSON data file override for tasks that declare `dataset_kwargs.data_files`.
    dataset_data_files: Optional[str] = None

    # Sampling configuration
    temperature: float = 0.0
    max_tokens: int = 256
    max_nfe: Optional[int] = None
    max_repetition_run: Optional[int] = None
    ignore_eos: bool = False
    add_bos_token: Optional[bool] = None  # Base model: False; Instruct/chat: True

    # Output configuration
    output_dir: str = "benchmark_results"
    # If True, lm-eval outputs + diffulex stats/trajectory go under output_dir/run_<time>_<task>/
    use_run_subdirectory: bool = True
    save_results: bool = True
    use_tqdm: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EvalConfig":
        """Create evaluation configuration from dictionary"""
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}

    def get_sampling_params(self):
        """Get sampling parameters"""
        from diffulex import SamplingParams

        return SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_nfe=self.max_nfe,
            max_repetition_run=self.max_repetition_run,
            ignore_eos=self.ignore_eos,
        )


@dataclass
class BenchmarkConfig:
    """
    Benchmark configuration - Combines engine and evaluation configurations
    """

    engine: EngineConfig
    eval: EvalConfig

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BenchmarkConfig":
        """
        Create benchmark configuration from dictionary

        Supports both flat and nested dictionary structures for backward compatibility
        """
        # Check if config_dict has nested structure
        if "engine" in config_dict and "eval" in config_dict:
            engine = EngineConfig.from_dict(config_dict["engine"])
            eval_config = EvalConfig.from_dict(config_dict["eval"])
        else:
            # Flat structure - backward compatibility
            # Split fields into engine and eval
            engine_fields = EngineConfig.accepted_input_fields()

            engine_dict = {k: v for k, v in config_dict.items() if k in engine_fields}
            eval_dict = {k: v for k, v in config_dict.items() if k not in engine_fields}

            engine = EngineConfig.from_dict(engine_dict)
            eval_config = EvalConfig.from_dict(eval_dict)

        return cls(engine=engine, eval=eval_config)

    @classmethod
    def from_json(cls, json_path: str) -> "BenchmarkConfig":
        """Load configuration from JSON file"""
        with open(json_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BenchmarkConfig":
        """Load configuration from YAML file"""
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with nested structure"""
        return {
            "engine": self.engine.to_dict(),
            "eval": self.eval.to_dict(),
        }

    def save_json(self, json_path: str):
        """Save to JSON file"""
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def save_yaml(self, yaml_path: str):
        """Save to YAML file"""
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False)

    def get_diffulex_kwargs(self) -> Dict[str, Any]:
        """Get arguments to pass to Diffulex engine"""
        return self.engine.get_diffulex_kwargs()

    def get_sampling_params(self):
        """Get sampling parameters"""
        return self.eval.get_sampling_params()

    # Convenience properties for backward compatibility
    @property
    def model_path(self) -> str:
        return self.engine.model_path

    @property
    def tokenizer_path(self) -> Optional[str]:
        return self.engine.tokenizer_path

    @property
    def model_name(self) -> str:
        return self.engine.model_name

    @property
    def decoding_strategy(self) -> str:
        return self.engine.decoding_strategy

    @property
    def dataset_name(self) -> str:
        return self.eval.dataset_name

    @property
    def dataset_limit(self) -> Optional[int]:
        return self.eval.dataset_limit

    @property
    def output_dir(self) -> str:
        return self.eval.output_dir

    @dataset_name.setter
    def dataset_name(self, value: str):
        self.eval.dataset_name = value

    @dataset_limit.setter
    def dataset_limit(self, value: Optional[int]):
        self.eval.dataset_limit = value

    @output_dir.setter
    def output_dir(self, value: str):
        self.eval.output_dir = value

    @model_path.setter
    def model_path(self, value: str):
        self.engine.model_path = value
