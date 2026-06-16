#!/usr/bin/env python3
"""Summarize benchmark results into a markdown table.

Usage:
    python script/summarize_benchmark.py <results_dir>
"""
import json
import glob
import os
import sys
from pathlib import Path

DATASET_ORDER = [
    "gsm8k_llada2",
    "math500_llada2",
    "mbpp_plus_llada2",
    "humaneval_plus_llada2",
]

DATASET_LABELS = {
    "gsm8k_llada2": "GSM8K",
    "math500_llada2": "MATH500",
    "mbpp_plus_llada2": "MBPP+",
    "humaneval_plus_llada2": "HumanEval+",
}

SCORE_KEY_PRIORITY = ["exact_match,none", "pass@1,none", "exact,none"]


def find_score(results_json: dict) -> str:
    results = results_json.get("results", {})
    for task_name, task_results in results.items():
        for key in SCORE_KEY_PRIORITY:
            if key in task_results:
                score = task_results[key]
                return f"{score * 100:.2f}"
    return "N/A"


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory")
        sys.exit(1)

    rows = []
    scores = []
    tpfs = []
    global_tps_list = []
    avg_tps_list = []

    for ds_name in DATASET_ORDER:
        ds_dir = results_dir / ds_name
        if not ds_dir.is_dir():
            continue

        # Find the model subdirectory (first subdir)
        model_dirs = [d for d in ds_dir.iterdir() if d.is_dir()]
        if not model_dirs:
            continue
        model_dir = model_dirs[0]

        # Find the config subdirectory (bs*/buf*)
        config_dirs = [d for d in model_dir.iterdir() if d.is_dir() and not d.name.startswith("__")]
        if not config_dirs:
            config_dirs = [model_dir]
        exp_dir = config_dirs[0]

        # Read diffulex_stats.json
        stats_path = exp_dir / "diffulex_stats.json"
        tpf = "N/A"
        decode_tps = "N/A"
        avg_decode_tps = "N/A"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            tpf_val = stats.get("tpf", stats.get("last_batch_tpf"))
            if tpf_val is not None:
                tpf = f"{tpf_val:.2f}"
                tpfs.append(tpf_val)
            dtps_val = stats.get("decode_throughput_tok_s")
            if dtps_val is not None:
                decode_tps = f"{dtps_val:.2f}"
                global_tps_list.append(dtps_val)
            avg_dtps_val = stats.get("avg_decode_tps")
            if avg_dtps_val is not None:
                avg_decode_tps = f"{avg_dtps_val:.2f}"
                avg_tps_list.append(avg_dtps_val)

        # Read results JSON
        score_str = "N/A"
        results_pattern = str(exp_dir / "**" / "results_*.json")
        results_files = glob.glob(results_pattern, recursive=True)
        if results_files:
            with open(results_files[0]) as f:
                results_json = json.load(f)
            score_str = find_score(results_json)
            try:
                scores.append(float(score_str))
            except ValueError:
                pass

        label = DATASET_LABELS.get(ds_name, ds_name)
        rows.append((label, score_str, tpf, decode_tps, avg_decode_tps))

    # Compute averages
    avg_score = f"{sum(scores) / len(scores):.2f}" if scores else "N/A"
    avg_tpf = f"{sum(tpfs) / len(tpfs):.2f}" if tpfs else "N/A"
    avg_global_tps = f"{sum(global_tps_list) / len(global_tps_list):.2f}" if global_tps_list else "N/A"
    avg_avg_tps = f"{sum(avg_tps_list) / len(avg_tps_list):.2f}" if avg_tps_list else "N/A"

    # Build markdown
    lines = []
    lines.append("# Benchmark Summary")
    lines.append("")
    lines.append(f"**Results directory:** `{results_dir}`")
    lines.append("")
    lines.append("| Dataset   | Acc (%) | TPF   | Decode TPS (global) | Avg Decode TPS (dInfer) |")
    lines.append("|-----------|---------|-------|---------------------|-------------------------|")
    for label, score, tpf, dtps, avg_dtps in rows:
        lines.append(f"| {label:<9} | {score:<7} | {tpf:<5} | {dtps:<19} | {avg_dtps:<23} |")
    lines.append(f"| **Mean**  | **{avg_score}** | **{avg_tpf}** | **{avg_global_tps}** | **{avg_avg_tps}** |")
    lines.append("")

    md_content = "\n".join(lines)

    # Write to results directory
    md_path = results_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Summary written to: {md_path}")
    print()
    print(md_content)


if __name__ == "__main__":
    main()
