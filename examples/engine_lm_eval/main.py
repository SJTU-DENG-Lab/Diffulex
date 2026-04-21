from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from examples.engine_lm_eval.config import BenchmarkConfig

try:
    from lm_eval.__main__ import cli_evaluate
except ImportError:
    cli_evaluate = None

from examples.engine_lm_eval.lm_eval_model import EngineOpenAILM  # noqa: F401


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_.")
    return value or "unknown"


def _resolve_include_path(config: BenchmarkConfig) -> Optional[Path]:
    raw = config.eval.include_path
    if raw is not None and str(raw).strip() == "":
        return None
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = Path(os.getcwd()) / p
        return p.resolve()
    return (Path(__file__).resolve().parents[2] / "diffulex_bench" / "tasks").resolve()


def _task_name_to_yaml_map(include_root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for yml in include_root.rglob("*.yaml"):
        try:
            text = yml.read_text(encoding="utf-8")
        except Exception:
            continue
        m = re.search(r"(?m)^\s*task:\s*([^\s#]+)\s*$", text)
        if m:
            mapping.setdefault(m.group(1).strip(), yml)
    return mapping


def _rewrite_task_data_files(task_yaml: Path, data_files: str) -> bool:
    text = task_yaml.read_text(encoding="utf-8")
    replacement_value = json.dumps(str(Path(data_files).expanduser().resolve()))
    replaced, n = re.subn(r"(?m)^(\s*data_files:\s*).*$", rf"\1{replacement_value}", text, count=1)
    if n == 0:
        return False
    task_yaml.write_text(replaced, encoding="utf-8")
    return True


def _resolve_include_path_with_override(config: BenchmarkConfig) -> tuple[Optional[Path], Optional[Path]]:
    include_path = _resolve_include_path(config)
    data_files = config.eval.dataset_data_files
    if not data_files:
        return include_path, None
    if include_path is None or not include_path.is_dir():
        return include_path, None
    tmp_root = Path(tempfile.mkdtemp(prefix="engine_lm_eval_tasks_")).resolve()
    tmp_tasks = tmp_root / "tasks"
    shutil.copytree(include_path, tmp_tasks, dirs_exist_ok=True)
    task_map = _task_name_to_yaml_map(tmp_tasks)
    for task_name in [name.strip() for name in config.eval.dataset_name.split(",") if name.strip()]:
        task_yaml = task_map.get(task_name)
        if task_yaml is not None:
            _rewrite_task_data_files(task_yaml, data_files)
    return tmp_tasks, tmp_root


def _run_output_dir(config: BenchmarkConfig) -> Path:
    engine_name = _slugify(config.endpoint.engine_name)
    root = Path(config.eval.output_dir).expanduser() / engine_name
    if not config.eval.use_run_subdirectory:
        root.mkdir(parents=True, exist_ok=True)
        return root.resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = _slugify(Path(config.endpoint.model).name)
    task_name = _slugify(config.eval.dataset_name.replace(",", "_"))
    out = root / f"run_{stamp}_{engine_name}_{model_name}_{task_name}"
    out.mkdir(parents=True, exist_ok=True)
    return out.resolve()


def config_to_model_args(config: BenchmarkConfig, save_dir: Path) -> str:
    args = {
        "engine_name": config.endpoint.engine_name,
        "base_url": config.endpoint.base_url,
        "model": config.endpoint.model,
        "api_key": config.endpoint.api_key,
        "tokenizer_path": config.endpoint.tokenizer_path or config.endpoint.model,
        "batch_size": 1,
        "max_new_tokens": config.eval.max_tokens,
        "temperature": config.eval.temperature,
        "ignore_eos": config.eval.ignore_eos,
        "add_bos_token": config.eval.add_bos_token if config.eval.add_bos_token is not None else False,
        "apply_chat_template": config.endpoint.apply_chat_template,
        "chat_completions": config.endpoint.chat_completions,
        "timeout": config.endpoint.timeout,
        "verify": config.endpoint.verify,
        "save_dir": str(save_dir),
        "trust_remote_code": config.endpoint.trust_remote_code,
    }
    return ",".join(f"{k}={v}" for k, v in args.items() if v is not None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal lm-eval runner backed by an OpenAI-compatible API engine")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    if cli_evaluate is None:
        raise RuntimeError("lm-evaluation-harness is not installed.")

    config = BenchmarkConfig.from_yaml(args.config)
    output_dir = _run_output_dir(config)
    include_path, cleanup_root = _resolve_include_path_with_override(config)
    model_args = config_to_model_args(config, output_dir)
    output_file = output_dir / f"{_slugify(config.endpoint.engine_name)}_lm_eval_results.json"

    sys.argv = [
        "lm_eval",
        "--model",
        "engine_oai",
        "--model_args",
        model_args,
        "--tasks",
        config.eval.dataset_name,
        "--output_path",
        str(output_file),
    ]
    if include_path is not None:
        sys.argv.extend(["--include_path", str(include_path)])
    if config.eval.dataset_limit:
        sys.argv.extend(["--limit", str(config.eval.dataset_limit)])

    try:
        cli_evaluate()
    finally:
        if cleanup_root is not None:
            shutil.rmtree(cleanup_root, ignore_errors=True)


if __name__ == "__main__":
    main()
