#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any


def _float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _load_profile_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        enriched = dict(row)
        enriched["profile_file"] = str(path)
        enriched["profile_name"] = path.name.removesuffix(".summary.json")
        enriched["self_cuda_time_ms"] = _float(row, "self_cuda_time_total_us") / 1000.0
        enriched["cuda_time_ms"] = _float(row, "cuda_time_total_us") / 1000.0
        enriched["self_cpu_time_ms"] = _float(row, "self_cpu_time_total_us") / 1000.0
        enriched["cpu_time_ms"] = _float(row, "cpu_time_total_us") / 1000.0
        out.append(enriched)
    return out


def _summary_path_for_trace(path: Path) -> Path:
    name = path.name
    if name.endswith(".trace.json.gz"):
        return path.with_name(name.removesuffix(".trace.json.gz") + ".summary.json")
    if name.endswith(".trace.json"):
        return path.with_name(name.removesuffix(".trace.json") + ".summary.json")
    return path.with_suffix(".summary.json")


def _load_trace_rows(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        trace = json.load(f)
    events = trace.get("traceEvents", trace if isinstance(trace, list) else [])
    grouped: dict[str, dict[str, Any]] = {}
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        name = str(event.get("name", "unknown"))
        dur = float(event.get("dur") or 0.0)
        if dur <= 0:
            continue
        cat = str(event.get("cat", "")).lower()
        row = grouped.setdefault(
            name,
            {
                "name": name,
                "count": 0,
                "self_cpu_time_total_us": 0.0,
                "cpu_time_total_us": 0.0,
                "self_cuda_time_total_us": 0.0,
                "cuda_time_total_us": 0.0,
                "self_cpu_memory_usage": 0,
                "self_cuda_memory_usage": 0,
                "input_shapes": "",
                "profile_file": str(path),
                "profile_name": path.name.removesuffix(".gz").removesuffix(".trace.json"),
                "source": "chrome_trace",
            },
        )
        row["count"] += 1
        if "cuda" in cat or "kernel" in cat or "gpu" in cat:
            row["self_cuda_time_total_us"] += dur
            row["cuda_time_total_us"] += dur
        else:
            row["self_cpu_time_total_us"] += dur
            row["cpu_time_total_us"] += dur

    out = []
    for row in grouped.values():
        row["self_cuda_time_ms"] = _float(row, "self_cuda_time_total_us") / 1000.0
        row["cuda_time_ms"] = _float(row, "cuda_time_total_us") / 1000.0
        row["self_cpu_time_ms"] = _float(row, "self_cpu_time_total_us") / 1000.0
        row["cpu_time_ms"] = _float(row, "cpu_time_total_us") / 1000.0
        out.append(row)
    return out


def _is_marker(name: str) -> bool:
    return name.startswith("diffulex.") or ".client." in name or name.startswith("sglang.")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "profile_name",
        "name",
        "count",
        "self_cuda_time_ms",
        "cuda_time_ms",
        "self_cpu_time_ms",
        "cpu_time_ms",
        "self_cuda_memory_usage",
        "profile_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict[str, Any]], *, limit: int) -> str:
    lines = [
        "| profile | event | count | self cuda ms | cuda ms | self cpu ms | cpu ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:limit]:
        lines.append(
            "| {profile} | `{name}` | {count} | {self_cuda:.3f} | {cuda:.3f} | {self_cpu:.3f} | {cpu:.3f} |".format(
                profile=row.get("profile_name", ""),
                name=str(row.get("name", "")).replace("|", "\\|"),
                count=row.get("count", ""),
                self_cuda=float(row.get("self_cuda_time_ms", 0.0)),
                cuda=float(row.get("cuda_time_ms", 0.0)),
                self_cpu=float(row.get("self_cpu_time_ms", 0.0)),
                cpu=float(row.get("cpu_time_ms", 0.0)),
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate torch profiler summary.json files into CSV/Markdown.")
    parser.add_argument("profile_root", type=Path)
    parser.add_argument("--top-n", type=int, default=40)
    args = parser.parse_args()

    profile_root = args.profile_root.expanduser().resolve()
    summary_files = sorted(profile_root.rglob("*.summary.json"))
    all_rows: list[dict[str, Any]] = []
    for path in summary_files:
        all_rows.extend(_load_profile_rows(path))
    summary_set = {path.resolve() for path in summary_files}
    trace_files = sorted(profile_root.rglob("*.trace.json")) + sorted(profile_root.rglob("*.trace.json.gz"))
    for path in trace_files:
        summary_path = _summary_path_for_trace(path).resolve()
        if summary_path in summary_set:
            related_rows = [
                row
                for row in all_rows
                if Path(str(row.get("profile_file", ""))).resolve() == summary_path
            ]
            has_cuda = any(float(row.get("self_cuda_time_ms", 0.0)) > 0 for row in related_rows)
            if has_cuda:
                continue
        try:
            all_rows.extend(_load_trace_rows(path))
        except Exception as exc:
            print(f"skip trace parse {path}: {exc}")

    all_rows.sort(
        key=lambda row: (
            float(row.get("self_cuda_time_ms", 0.0)),
            float(row.get("self_cpu_time_ms", 0.0)),
        ),
        reverse=True,
    )
    marker_rows = [row for row in all_rows if _is_marker(str(row.get("name", "")))]

    profile_root.mkdir(parents=True, exist_ok=True)
    _write_csv(profile_root / "torch_profile_top_ops.csv", all_rows)
    _write_csv(profile_root / "torch_profile_markers.csv", marker_rows)

    md = [
        "# Torch Profile Summary",
        "",
        f"- profile root: `{profile_root}`",
        f"- summary files: {len(summary_files)}",
        f"- trace-only files parsed: {len(trace_files)}",
        f"- rows: {len(all_rows)}",
        "",
        "## Annotated Stages",
        "",
        _markdown_table(marker_rows, limit=args.top_n),
        "",
        "## Top Operators",
        "",
        _markdown_table(all_rows, limit=args.top_n),
        "",
    ]
    (profile_root / "torch_profile_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {profile_root / 'torch_profile_summary.md'}")
    print(f"Wrote {profile_root / 'torch_profile_markers.csv'}")
    print(f"Wrote {profile_root / 'torch_profile_top_ops.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
