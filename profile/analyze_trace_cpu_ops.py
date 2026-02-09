#!/usr/bin/env python3
"""
Stream-aggregate CPU-side durations from huge torch chrome traces.

We aggregate:
- cat=cpu_op
- cat=python_function
- cat=user_annotation

This helps answer: where is the extra walltime coming from (outside CUDA kernels)?
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple


def _extract_str_after_key(s: str, key: str) -> Optional[str]:
    k = f"\"{key}\""
    pos = s.find(k)
    if pos < 0:
        return None
    colon = s.find(":", pos + len(k))
    if colon < 0:
        return None
    q1 = s.find('"', colon)
    if q1 < 0:
        return None
    q2 = s.find('"', q1 + 1)
    if q2 < 0:
        return None
    return s[q1 + 1 : q2]


def _extract_num_after_key(s: str, key: str) -> Optional[float]:
    k = f"\"{key}\""
    pos = s.find(k)
    if pos < 0:
        return None
    colon = s.find(":", pos + len(k))
    if colon < 0:
        return None
    frag = s[colon + 1 :].strip()
    comma = frag.find(",")
    if comma >= 0:
        frag = frag[:comma]
    try:
        return float(frag.strip())
    except Exception:
        return None


def analyze(trace_path: Path, cats: Tuple[str, ...]) -> Dict[str, Dict[str, Dict[str, float]]]:
    # cat -> name -> (dur_us_sum, calls)
    dur: Dict[str, Counter[str]] = {c: Counter() for c in cats}
    calls: Dict[str, Counter[str]] = {c: Counter() for c in cats}

    in_events = False
    in_obj = False
    depth = 0
    buf = []

    def consume(text: str) -> None:
        if '"cat"' not in text or '"name"' not in text:
            return
        cat = None
        name = None
        d = None
        for line in text.splitlines():
            if cat is None and '"cat"' in line:
                cat = _extract_str_after_key(line, "cat")
            if name is None and '"name"' in line:
                name = _extract_str_after_key(line, "name")
            if d is None and '"dur"' in line:
                d = _extract_num_after_key(line, "dur")
            if cat and name and d is not None:
                break
        if cat not in cats or name is None:
            return
        calls[cat][name] += 1
        if d is not None:
            dur[cat][name] += d

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
                consume("".join(buf))
                in_obj = False

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for c in cats:
        out[c] = {}
        for name, total in dur[c].items():
            out[c][name] = {
                "dur_us": float(total),
                "calls": float(calls[c][name]),
                "avg_us": float(total) / float(calls[c][name]) if calls[c][name] else 0.0,
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    cats = ("cpu_op", "python_function", "user_annotation")
    res = analyze(Path(args.trace), cats)

    # Write a compact report: per-cat topk by dur.
    lines = []
    lines.append(f"Trace: {args.trace}")
    lines.append("")
    for c in cats:
        items = sorted(res[c].items(), key=lambda kv: kv[1]["dur_us"], reverse=True)[: args.topk]
        lines.append(f"== {c} top {args.topk} by dur_us ==")
        for name, st in items:
            lines.append(f"{st['dur_us']:.3f} us  calls={int(st['calls'])}  avg={st['avg_us']:.3f} us  {name}")
        lines.append("")

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote: {args.out}")


if __name__ == "__main__":
    main()

