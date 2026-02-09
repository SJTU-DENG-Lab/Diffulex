#!/usr/bin/env python3
"""
Stream-parse PyTorch chrome trace JSON (very large) and aggregate GEMM shape
distributions for selected ops.

This script is designed for traces exported with record_shapes=True, where op
events contain args["Input Dims"].

Example:
  python profile/analyze_trace_gemm_shapes.py \
    --trace log/torch_profiles/20260125_023133/pytorch_trace_diffulex.generate(profiled).json \
    --out  log/torch_profiles/20260125_023133/gemm_shapes_bf16.txt \
    --ops  aten::mm aten::addmm
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _parse_json_value_fragment(fragment: str) -> Any:
    # fragment: after ':' in a JSON line, possibly ending with ',' and newline.
    frag = fragment.strip()
    if frag.endswith(","):
        frag = frag[:-1]
    return json.loads(frag)

def _extract_json_array_after_key(line: str, key: str) -> Optional[Any]:
    """
    Extract and json-load the array value after `"key":` from a possibly
    multi-field JSON line, e.g.
      ..."Input Dims": [[1,2],[3,4]], "Ev Idx": 5
    """
    k = f"\"{key}\""
    pos = line.find(k)
    if pos < 0:
        return None
    colon = line.find(":", pos + len(k))
    if colon < 0:
        return None
    # Find the first '[' after the colon.
    start = line.find("[", colon)
    if start < 0:
        return None
    depth = 0
    end = -1
    for i in range(start, len(line)):
        ch = line[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    frag = line[start:end]
    try:
        return json.loads(frag)
    except Exception:
        return None


def _extract_quoted_value(line: str) -> Optional[str]:
    # very small helper: extract first "...".
    i = line.find('"')
    if i < 0:
        return None
    j = line.find('"', i + 1)
    if j < 0:
        return None
    return line[i + 1 : j]


def _extract_number_after_colon(line: str) -> Optional[float]:
    # e.g.  "dur": 123.0,
    if ":" not in line:
        return None
    frag = line.split(":", 1)[1].strip()
    if frag.endswith(","):
        frag = frag[:-1]
    try:
        return float(frag)
    except Exception:
        return None

def _extract_number_after_key(line: str, key: str) -> Optional[float]:
    """
    Extract a numeric value after `"key":` from a possibly multi-field JSON line, e.g.
      "ts": 123.0, "dur": 34.5,
    """
    k = f"\"{key}\""
    pos = line.find(k)
    if pos < 0:
        return None
    colon = line.find(":", pos + len(k))
    if colon < 0:
        return None
    frag = line[colon + 1 :].strip()
    # Cut at next comma if present.
    comma = frag.find(",")
    if comma >= 0:
        frag = frag[:comma]
    try:
        return float(frag.strip())
    except Exception:
        return None


def _dims_to_mnk(input_dims: Any) -> Optional[Tuple[int, int, int]]:
    """
    Convert args["Input Dims"] into a best-effort (M,N,K).
    input_dims is typically a list where each element is [] (non-tensor) or
    a list[int] (tensor dims).
    """
    if not isinstance(input_dims, list):
        return None

    tensor_dims: List[List[int]] = []
    for d in input_dims:
        if isinstance(d, list) and len(d) >= 2 and all(isinstance(x, (int, float)) for x in d):
            tensor_dims.append([int(x) for x in d])
    if len(tensor_dims) < 2:
        return None

    a = tensor_dims[0]
    b = tensor_dims[1]
    a_m, a_k = a[-2], a[-1]
    # b could be [k, n] or [n, k] depending on transpose convention.
    if len(b) >= 2 and a_k == b[-2]:
        b_k, b_n = b[-2], b[-1]
        return (a_m, b_n, a_k)
    if len(b) >= 2 and a_k == b[-1]:
        # b is [n, k]
        b_n, b_k = b[-2], b[-1]
        return (a_m, b_n, a_k)

    # fallback: assume [k, n]
    return (a_m, b[-1], a_k)


@dataclass
class ShapeStats:
    calls: int = 0
    dur_us: float = 0.0


def iter_op_events(trace_path: Path, target_ops: set[str]) -> Iterable[Tuple[str, Optional[float], Any]]:
    """
    Yields (op_name, dur_us, input_dims) for events whose "name" is in target_ops.
    Streaming + brace-depth parsing to avoid loading giant JSON into memory.
    """
    in_trace_events = False
    in_event = False
    depth = 0

    name: Optional[str] = None
    dur: Optional[float] = None
    input_dims: Any = None
    want = False

    with trace_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not in_trace_events:
                if '"traceEvents"' in line and "[" in line:
                    in_trace_events = True
                continue

            # Start of a JSON object event in traceEvents list.
            if not in_event:
                stripped = line.lstrip()
                if stripped.startswith("{"):
                    in_event = True
                    depth = stripped.count("{") - stripped.count("}")
                    name = None
                    dur = None
                    input_dims = None
                    want = False
                else:
                    # End of traceEvents list.
                    if line.lstrip().startswith("]"):
                        break
                    continue
            else:
                depth += line.count("{") - line.count("}")

            # Parse fields we care about.
            if '"name"' in line:
                # Some traces put multiple fields on one line:
                #   "ph": "X", "cat": "cpu_op", "name": "aten::mm", ...
                key = '"name":'
                pos = line.find(key)
                if pos >= 0:
                    q1 = line.find('"', pos + len(key))
                    if q1 >= 0:
                        q2 = line.find('"', q1 + 1)
                        if q2 >= 0:
                            name = line[q1 + 1 : q2]
                            want = name in target_ops

            if want and dur is None and '"dur"' in line:
                dur = _extract_number_after_key(line, "dur")

            if want and input_dims is None and "Input Dims" in line:
                input_dims = _extract_json_array_after_key(line, "Input Dims")

            # End of current event object (also works for single-line events).
            if in_event and depth <= 0:
                if want and name is not None:
                    yield (name, dur, input_dims)
                in_event = False


def _human_int(n: float) -> str:
    if n >= 1e9:
        return f"{n/1e9:.3f}B"
    if n >= 1e6:
        return f"{n/1e6:.3f}M"
    if n >= 1e3:
        return f"{n/1e3:.3f}K"
    return f"{int(n)}"


def main() -> None:
    ap = argparse.ArgumentParser("Aggregate GEMM shapes from huge torch chrome trace")
    ap.add_argument("--trace", type=str, required=True, help="Path to pytorch_trace_*.json")
    ap.add_argument("--out", type=str, required=True, help="Output report path")
    ap.add_argument("--ops", type=str, nargs="+", default=["aten::mm", "aten::addmm"], help="Op names to aggregate")
    ap.add_argument("--topk", type=int, default=30)
    args = ap.parse_args()

    trace_path = Path(args.trace)
    out_path = Path(args.out)
    target_ops = set(args.ops)

    # op -> (mnk -> stats)
    agg: Dict[str, Dict[Tuple[int, int, int], ShapeStats]] = defaultdict(dict)
    op_totals: Dict[str, ShapeStats] = defaultdict(ShapeStats)
    op_unknown: Counter[str] = Counter()

    for op, dur_us, input_dims in iter_op_events(trace_path, target_ops):
        op_totals[op].calls += 1
        if dur_us is not None:
            op_totals[op].dur_us += dur_us

        mnk = _dims_to_mnk(input_dims)
        if mnk is None:
            op_unknown[op] += 1
            continue

        st = agg[op].get(mnk)
        if st is None:
            st = ShapeStats()
            agg[op][mnk] = st
        st.calls += 1
        if dur_us is not None:
            st.dur_us += dur_us

    lines: List[str] = []
    lines.append(f"Trace: {trace_path}")
    lines.append(f"Ops: {', '.join(sorted(target_ops))}")
    lines.append("")

    for op in sorted(target_ops):
        tot = op_totals.get(op, ShapeStats())
        lines.append(f"== {op} ==")
        lines.append(f"total calls: {tot.calls}")
        lines.append(f"total dur(us): {tot.dur_us:.3f}")
        lines.append(f"unknown shapes: {op_unknown.get(op, 0)}")
        lines.append("")

        if op not in agg or not agg[op]:
            lines.append("(no shape stats)\n")
            continue

        # Top by total dur
        items = list(agg[op].items())
        items_by_dur = sorted(items, key=lambda kv: kv[1].dur_us, reverse=True)[: args.topk]
        lines.append(f"-- top {args.topk} shapes by total dur(us) --")
        lines.append("M,N,K  calls  total_dur(us)  approx_GFLOP")
        for (m, n, k), st in items_by_dur:
            gflop = 2.0 * m * n * k / 1e9
            lines.append(f"{m},{n},{k}  {st.calls}  {st.dur_us:.3f}  {gflop:.3f}")
        lines.append("")

        # Top by calls
        items_by_calls = sorted(items, key=lambda kv: kv[1].calls, reverse=True)[: args.topk]
        lines.append(f"-- top {args.topk} shapes by calls --")
        lines.append("M,N,K  calls  total_dur(us)  avg_dur(us)")
        for (m, n, k), st in items_by_calls:
            avg = st.dur_us / st.calls if st.calls else 0.0
            lines.append(f"{m},{n},{k}  {st.calls}  {st.dur_us:.3f}  {avg:.3f}")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()

