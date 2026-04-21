from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml


@dataclass
class EndpointConfig:
    base_url: str
    model: str
    engine_name: str = "sglang"
    api_key: str = "EMPTY"
    tokenizer_path: Optional[str] = None
    trust_remote_code: bool = True
    apply_chat_template: bool = False
    chat_completions: bool = False
    timeout: float = 600.0
    verify: bool = True


@dataclass
class EvalConfig:
    dataset_name: str = "gsm8k_diffulex"
    dataset_limit: Optional[int] = 10
    include_path: Optional[str] = None
    dataset_data_files: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 256
    ignore_eos: bool = False
    add_bos_token: Optional[bool] = None
    output_dir: str = "benchmark_results/oai_lm_eval"
    use_run_subdirectory: bool = True
    save_results: bool = True

    @classmethod
    def from_dict(cls, data: Dict) -> "EvalConfig":
        valid = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in valid})


@dataclass
class BenchmarkConfig:
    endpoint: EndpointConfig
    eval: EvalConfig

    @classmethod
    def from_yaml(cls, path: str) -> "BenchmarkConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            endpoint=EndpointConfig(**raw.get("endpoint", {})),
            eval=EvalConfig.from_dict(raw.get("eval", {})),
        )
