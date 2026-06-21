#!/usr/bin/env python3
"""Resolve experiment group configs into concrete diffulex_bench YAML files."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


def merge_dict(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    if not override:
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def sanitize(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "run"


def split_list(raw: str) -> list[str]:
    return [item for item in re.split(r"[,\s]+", raw.strip()) if item]


def match_filter(group_name: str, exp: dict[str, Any], patterns: list[str]) -> bool:
    if not patterns:
        return True
    haystack = " ".join(
        [group_name]
        + [
            str(exp.get(key, ""))
            for key in ("name", "variant", "task", "model", "decoding_strategy", "sampling_mode")
        ]
    ).lower()
    return any(pattern.lower() in haystack for pattern in patterns)


def resolve_model_path(model: dict[str, Any]) -> str:
    env_name = model.get("env")
    raw = os.environ.get(env_name, "") if env_name else ""
    if not raw:
        raw = str(model["path"])
    return str(Path(os.path.expandvars(raw)).expanduser())


def path_is_dir(path: str) -> bool:
    try:
        return Path(path).is_dir()
    except OSError:
        return False


def resolve_file(path: str, base_dir: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    candidate = base_dir / p
    if candidate.exists():
        return candidate.resolve()
    return (Path.cwd() / p).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--defaults-config", required=True)
    parser.add_argument("--config-pattern", default="*.yml")
    parser.add_argument("--config-files", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--plan-tsv", required=True)
    parser.add_argument("--filter", default="")
    parser.add_argument("--dataset-limit", default="")
    parser.add_argument("--max-num-reqs", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_dir = Path(args.config_dir).expanduser()
    if not config_dir.is_absolute():
        config_dir = (Path.cwd() / config_dir).resolve()
    defaults_path = resolve_file(args.defaults_config, config_dir)
    base_config = load_yaml(defaults_path)

    if args.config_files.strip():
        group_files = [resolve_file(item, config_dir) for item in split_list(args.config_files)]
    else:
        group_files = sorted(config_dir.glob(args.config_pattern))
        group_files = [p.resolve() for p in group_files if not p.name.startswith("_")]

    output_root = Path(args.output_root).expanduser()
    log_root = Path(args.log_dir).expanduser()
    plan_tsv = Path(args.plan_tsv).expanduser()
    resolved_root = output_root / "resolved_configs"
    resolved_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    default_block = base_config.get("defaults", {})
    base_engine = default_block.get("engine", {})
    base_eval = default_block.get("eval", {})
    base_thresholds = default_block.get("thresholds", {})
    base_tasks = base_config.get("tasks", {})
    base_models = base_config.get("models", {})

    patterns = split_list(args.filter)
    dataset_limit = args.dataset_limit.strip()
    max_num_reqs = args.max_num_reqs.strip()

    rows: list[dict[str, Any]] = []
    for group_file in group_files:
        group = load_yaml(group_file)

        group_name = sanitize(str(group.get("name") or group_file.stem))
        group_defaults = group.get("defaults", {})
        default_engine = merge_dict(base_engine, group_defaults.get("engine"))
        default_eval = merge_dict(base_eval, group_defaults.get("eval"))
        default_thresholds = merge_dict(base_thresholds, group_defaults.get("thresholds"))
        tasks = merge_dict(base_tasks, group.get("tasks"))
        models = merge_dict(base_models, group.get("models"))

        for source_index, exp in enumerate(group.get("experiments", []), start=1):
            if not match_filter(group_name, exp, patterns):
                continue

            exp_name = sanitize(str(exp["name"]))
            model = models[exp["model"]]
            task_name = tasks[exp["task"]]
            model_path = resolve_model_path(model)

            row_index = len(rows) + 1
            run_name = f"{row_index:02d}_{group_name}__{exp_name}"
            run_dir = output_root / "runs" / run_name
            config_path = resolved_root / f"{run_name}.yml"
            log_path = log_root / f"{run_name}.log"

            thresholds = merge_dict(default_thresholds, exp.get("thresholds"))

            engine = merge_dict(default_engine, exp.get("engine"))
            engine.update(
                {
                    "model_path": model_path,
                    "model_name": model["model_name"],
                    "mask_token_id": model["mask_token_id"],
                    "decoding_strategy": exp["decoding_strategy"],
                    "sampling_mode": exp["sampling_mode"],
                    "block_size": int(exp["block_size"]),
                    "buffer_size": int(exp["buffer_size"]),
                    "page_size": int(exp.get("page_size", exp["block_size"])),
                    "decoding_thresholds": thresholds,
                }
            )
            if max_num_reqs:
                engine["max_num_reqs"] = int(max_num_reqs)

            eval_config = merge_dict(default_eval, exp.get("eval"))
            eval_config.update(
                {
                    "dataset_name": task_name,
                    "output_dir": str(run_dir),
                    "use_run_subdirectory": False,
                }
            )
            if dataset_limit:
                eval_config["dataset_limit"] = int(dataset_limit)

            concrete = {"engine": engine, "eval": eval_config}
            with config_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(concrete, f, sort_keys=False)

            metadata = {
                "group_file": str(group_file),
                "group": group_name,
                "source_index": source_index,
                "name": exp["name"],
                "variant": exp.get("variant"),
                "model_key": exp["model"],
                "model_path": model_path,
                "task_key": exp["task"],
                "dataset_name": task_name,
                "config_path": str(config_path),
                "output_dir": str(run_dir),
                "log_file": str(log_path),
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            with (run_dir / "experiment_config.json").open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            rows.append(
                {
                    "index": row_index,
                    "group": group_name,
                    "name": exp["name"],
                    "variant": exp.get("variant", ""),
                    "task": exp["task"],
                    "model": exp["model"],
                    "model_path": model_path,
                    "model_exists": "1" if path_is_dir(model_path) else "0",
                    "config": str(config_path),
                    "output_dir": str(run_dir),
                    "log_file": str(log_path),
                }
            )

    headers = [
        "index",
        "group",
        "name",
        "variant",
        "task",
        "model",
        "model_path",
        "model_exists",
        "config",
        "output_dir",
        "log_file",
    ]
    plan_tsv.parent.mkdir(parents=True, exist_ok=True)
    with plan_tsv.open("w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for row in rows:
            f.write("\t".join(str(row[h]) for h in headers) + "\n")

    print(f"Loaded defaults: {defaults_path}")
    print("Loaded group configs:")
    for group_file in group_files:
        print(f"  - {group_file}")
    print(f"Resolved {len(rows)} experiment(s)")
    print(f"Plan: {plan_tsv}")
    print(f"Configs: {resolved_root}")


if __name__ == "__main__":
    main()
