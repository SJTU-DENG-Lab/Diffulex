from __future__ import annotations

import csv
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch


def profile_enabled() -> bool:
    return bool(os.getenv("DIFFULEX_PROFILE_DIR"))


def record_function(name: str):
    if not profile_enabled():
        return nullcontext()
    return torch.profiler.record_function(name)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


class TorchProfileSession:
    def __init__(self, component: str, *, rank: int | None = None):
        self.enabled = profile_enabled()
        self.component = component
        self.rank = rank
        self.prof: torch.profiler.profile | None = None
        self.started = False
        self.stopped = False
        self.steps = 0
        self.active_steps = _env_int("DIFFULEX_PROFILE_ACTIVE_STEPS", 200)

    def start(self) -> None:
        if not self.enabled or self.started or self.stopped:
            return
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        kwargs = dict(
            activities=activities,
            record_shapes=_env_bool("DIFFULEX_PROFILE_RECORD_SHAPES", False),
            profile_memory=_env_bool("DIFFULEX_PROFILE_MEMORY", False),
            with_stack=_env_bool("DIFFULEX_PROFILE_WITH_STACK", False),
            with_modules=_env_bool("DIFFULEX_PROFILE_WITH_MODULES", False),
            acc_events=True,
        )
        try:
            self.prof = torch.profiler.profile(**kwargs)
        except TypeError:
            kwargs.pop("acc_events", None)
            self.prof = torch.profiler.profile(**kwargs)
        self.prof.start()
        self.started = True

    def step(self) -> None:
        if not self.enabled or self.stopped:
            return
        self.start()
        self.steps += 1
        if self.prof is not None and not self.stopped:
            self.prof.step()
        if self.active_steps > 0 and self.steps >= self.active_steps:
            self.stop()

    def stop(self) -> None:
        if not self.enabled or not self.started or self.stopped:
            return
        assert self.prof is not None
        self.stopped = True
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.prof.stop()
            self._export()
        except Exception as exc:
            prefix = self._prefix()
            prefix.with_suffix(".error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")

    def _prefix(self) -> Path:
        root = Path(os.environ["DIFFULEX_PROFILE_DIR"]).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        rank_part = f".rank{self.rank}" if self.rank is not None else ""
        stamp = os.getenv("DIFFULEX_PROFILE_RUN_ID") or time.strftime("%Y%m%d_%H%M%S")
        return root / f"{stamp}.{self.component}{rank_part}"

    @staticmethod
    def _event_to_row(event: Any) -> dict[str, Any]:
        self_device_time = getattr(
            event,
            "self_cuda_time_total",
            getattr(event, "self_device_time_total", 0.0),
        )
        device_time = getattr(
            event,
            "cuda_time_total",
            getattr(event, "device_time_total", 0.0),
        )
        self_device_memory = getattr(
            event,
            "self_cuda_memory_usage",
            getattr(event, "self_device_memory_usage", 0),
        )
        return {
            "name": event.key,
            "count": event.count,
            "self_cpu_time_total_us": getattr(event, "self_cpu_time_total", 0.0),
            "cpu_time_total_us": getattr(event, "cpu_time_total", 0.0),
            "self_cuda_time_total_us": self_device_time,
            "cuda_time_total_us": device_time,
            "self_cpu_memory_usage": getattr(event, "self_cpu_memory_usage", 0),
            "self_cuda_memory_usage": self_device_memory,
            "input_shapes": str(getattr(event, "input_shapes", "")),
        }

    def _export(self) -> None:
        assert self.prof is not None
        prefix = self._prefix()
        trace_path = prefix.with_suffix(".trace.json")
        summary_txt = prefix.with_suffix(".summary.txt")
        summary_csv = prefix.with_suffix(".summary.csv")
        summary_json = prefix.with_suffix(".summary.json")
        sort_by = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
        events = self.prof.key_averages()
        rows = [self._event_to_row(event) for event in events]
        rows.sort(key=lambda row: row["self_cuda_time_total_us"] or row["self_cpu_time_total_us"], reverse=True)
        self.prof.export_chrome_trace(str(trace_path))
        summary_txt.write_text(events.table(sort_by=sort_by, row_limit=200), encoding="utf-8")
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["name"])
            writer.writeheader()
            writer.writerows(rows)
        summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
