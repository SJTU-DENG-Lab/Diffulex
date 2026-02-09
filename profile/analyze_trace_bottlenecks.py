#!/usr/bin/env python3
"""
Analyze huge torch chrome trace (streaming) to locate non-GEMM bottlenecks.

Outputs:
- duration of user_annotation "diffulex.generate(profiled)" (wall-ish)
- GPU active time (union of kernel/memcpy/memset intervals) to estimate GPU idle gaps
- top CUDA runtime/driver API calls by CPU time

Designed to work without loading the >2GB JSON into memory.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _extract_str_after_key(line: str, key: str) -> Optional[str]:
    k = f"\"{key}\""
    pos = line.find(k)
    if pos < 0:
        return None
    colon = line.find(":", pos + len(k))
    if colon < 0:
        return None
    q1 = line.find('"', colon)
    if q1 < 0:
        return None
    q2 = line.find('"', q1 + 1)
    if q2 < 0:
        return None
    return line[q1 + 1 : q2]


def _extract_num_after_key(line: str, key: str) -> Optional[float]:
    k = f"\"{key}\""
    pos = line.find(k)
    if pos < 0:
        return None
    colon = line.find(":", pos + len(k))
    if colon < 0:
        return None
    frag = line[colon + 1 :].strip()
    comma = frag.find(",")
    if comma >= 0:
        frag = frag[:comma]
    try:
        return float(frag.strip())
    except Exception:
        return None


def _extract_json_object_value(line: str, key: str) -> Optional[Any]:
    """
    Extract JSON object/array value following `"key":` on the same line.
    Assumes the value is a JSON object {...} or array [...] and is fully contained in the line.
    """
    k = f"\"{key}\""
    pos = line.find(k)
    if pos < 0:
        return None
    colon = line.find(":", pos + len(k))
    if colon < 0:
        return None
    # find first '{' or '[' after colon
    start = None
    for i in range(colon, len(line)):
        if line[i] == "{":
            start = i
            open_ch, close_ch = "{", "}"
            break
        if line[i] == "[":
            start = i
            open_ch, close_ch = "[", "]"
            break
    if start is None:
        return None
    depth = 0
    end = None
    for i in range(start, len(line)):
        ch = line[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    frag = line[start:end]
    try:
        return json.loads(frag)
    except Exception:
        return None


@dataclass
class Interval:
    start: float
    end: float


def _merge_intervals(intervals: List[Interval]) -> List[Interval]:
    if not intervals:
        return []
    intervals.sort(key=lambda x: x.start)
    merged: List[Interval] = [intervals[0]]
    for it in intervals[1:]:
        last = merged[-1]
        if it.start <= last.end:
            if it.end > last.end:
                last.end = it.end
        else:
            merged.append(it)
    return merged


def analyze(trace_path: Path) -> Dict[str, Any]:
    # union intervals for GPU activity across all streams
    gpu_intervals: List[Interval] = []
    gpu_min_ts: Optional[float] = None
    gpu_max_end: Optional[float] = None

    # also per stream, to detect if one stream is idle most of the time
    gpu_intervals_by_stream: Dict[int, List[Interval]] = defaultdict(list)

    # user annotation
    generate_dur_us: Optional[float] = None

    # runtime/driver api durations (cpu-side)
    cuda_runtime: Counter[str] = Counter()
    cuda_driver: Counter[str] = Counter()

    in_events = False
    in_obj = False
    depth = 0
    buf: List[str] = []

    def _consume_event(text: str) -> None:
        nonlocal generate_dur_us, gpu_min_ts, gpu_max_end
        # quick checks without json parsing
        if '"cat"' not in text or '"name"' not in text:
            return
        cat = None
        name = None
        # extract cat/name
        # cat and name appear on first line typically, but safe on full text.
        for line in text.splitlines():
            if cat is None and '"cat"' in line:
                v = _extract_str_after_key(line, "cat")
                if v:
                    cat = v
            if name is None and '"name"' in line:
                v = _extract_str_after_key(line, "name")
                if v:
                    name = v
            if cat is not None and name is not None:
                break
        if cat is None or name is None:
            return

        if cat == "user_annotation" and name == "diffulex.generate(profiled)":
            # duration in us
            for line in text.splitlines():
                if '"dur"' in line:
                    d = _extract_num_after_key(line, "dur")
                    if d is not None:
                        generate_dur_us = d
                        break
            return

        # cuda runtime/driver (CPU)
        if cat == "cuda_runtime":
            d = None
            for line in text.splitlines():
                if '"dur"' in line:
                    d = _extract_num_after_key(line, "dur")
                    break
            if d is not None:
                cuda_runtime[name] += d
            return
        if cat == "cuda_driver":
            d = None
            for line in text.splitlines():
                if '"dur"' in line:
                    d = _extract_num_after_key(line, "dur")
                    break
            if d is not None:
                cuda_driver[name] += d
            return

        # GPU activity events
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            ts = None
            dur = None
            stream = None
            for line in text.splitlines():
                if ts is None and '"ts"' in line:
                    ts = _extract_num_after_key(line, "ts")
                if dur is None and '"dur"' in line:
                    dur = _extract_num_after_key(line, "dur")
                if stream is None and '"args"' in line and "stream" in line:
                    # args is often multi-line; rely on json fragment extraction when seen
                    pass
            # extract args object to fetch stream quickly (safe, small)
            args_obj = None
            for line in text.splitlines():
                if '"args"' in line:
                    args_obj = _extract_json_object_value(line, "args")
                    break
            if isinstance(args_obj, dict):
                try:
                    stream = int(args_obj.get("stream", -1))
                except Exception:
                    stream = None
            if ts is None or dur is None:
                return
            start = ts
            end = ts + dur
            gpu_intervals.append(Interval(start, end))
            if stream is not None and stream >= 0:
                gpu_intervals_by_stream[stream].append(Interval(start, end))
            gpu_min_ts = start if gpu_min_ts is None else min(gpu_min_ts, start)
            gpu_max_end = end if gpu_max_end is None else max(gpu_max_end, end)
            return

    with trace_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not in_events:
                if '"traceEvents"' in line and "[" in line:
                    in_events = True
                continue
            if not in_obj:
                if line.lstrip().startswith("{"):
                    in_obj = True
                    buf = [line]
                    depth = line.count("{") - line.count("}")
                else:
                    if line.lstrip().startswith("]"):
                        break
                    continue
            else:
                buf.append(line)
                depth += line.count("{") - line.count("}")
            if in_obj and depth <= 0:
                _consume_event("".join(buf))
                in_obj = False

    merged = _merge_intervals(gpu_intervals)
    active_us = sum(it.end - it.start for it in merged)
    span_us = (gpu_max_end - gpu_min_ts) if (gpu_min_ts is not None and gpu_max_end is not None) else 0.0

    per_stream_active: Dict[int, float] = {}
    for s, ints in gpu_intervals_by_stream.items():
        m = _merge_intervals(ints)
        per_stream_active[s] = sum(it.end - it.start for it in m)

    top_runtime = cuda_runtime.most_common(30)
    top_driver = cuda_driver.most_common(30)

    return {
        "trace": str(trace_path),
        "generate_dur_us": generate_dur_us,
        "gpu_active_union_us": active_us,
        "gpu_span_us": span_us,
        "gpu_active_ratio_union_over_span": (active_us / span_us) if span_us > 0 else None,
        "gpu_active_ratio_union_over_generate": (active_us / generate_dur_us) if (generate_dur_us and generate_dur_us > 0) else None,
        "gpu_span_over_generate": (span_us / generate_dur_us) if (generate_dur_us and generate_dur_us > 0) else None,
        "gpu_event_count": len(gpu_intervals),
        "gpu_stream_count": len(per_stream_active),
        "top_cuda_runtime_us": top_runtime,
        "top_cuda_driver_us": top_driver,
        "top_stream_active_us": sorted(per_stream_active.items(), key=lambda kv: kv[1], reverse=True)[:10],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    res = analyze(Path(args.trace))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()

