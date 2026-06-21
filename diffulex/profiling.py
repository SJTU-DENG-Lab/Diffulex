from __future__ import annotations

import csv
import json
import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from diffulex.logger import get_logger

logger = get_logger(__name__)

ProfilerKind = Literal["torch"]

_PROFILE_SCOPE_REFCOUNT = 0


@dataclass
class ProfilerConfig:
    profiler: ProfilerKind | None = None
    torch_profiler_dir: str = ""
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    with_modules: bool = False
    use_cuda: bool = True
    delay_iterations: int = 0
    max_iterations: int = 0
    run_id: str | None = None

    @classmethod
    def from_value(cls, value: "ProfilerConfig | dict[str, Any] | None") -> "ProfilerConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(f"Unsupported profiler_config type: {type(value)!r}")

    def __post_init__(self) -> None:
        if self.profiler not in (None, "torch"):
            raise ValueError(f"profiler must be one of {{None, 'torch'}}, got: {self.profiler}")
        if self.profiler == "torch" and not self.torch_profiler_dir:
            raise ValueError("torch_profiler_dir must be set when profiler='torch'.")
        if self.delay_iterations < 0:
            raise ValueError(f"delay_iterations must be non-negative, got: {self.delay_iterations}")
        if self.max_iterations < 0:
            raise ValueError(f"max_iterations must be non-negative, got: {self.max_iterations}")
        if self.torch_profiler_dir:
            self.torch_profiler_dir = str(Path(self.torch_profiler_dir).expanduser().resolve())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _enable_profile_scopes() -> None:
    global _PROFILE_SCOPE_REFCOUNT
    _PROFILE_SCOPE_REFCOUNT += 1


def _disable_profile_scopes() -> None:
    global _PROFILE_SCOPE_REFCOUNT
    _PROFILE_SCOPE_REFCOUNT = max(0, _PROFILE_SCOPE_REFCOUNT - 1)


def profile_scopes_enabled() -> bool:
    return _PROFILE_SCOPE_REFCOUNT > 0


def record_function(name: str):
    if not profile_scopes_enabled():
        return nullcontext()
    return torch.profiler.record_function(name)


profile_scope = record_function

_STAGE_TIMING_COUNTS: dict[str, int] = {}


class CudaStageTimer:
    def __init__(self, scope: str, *, rank: int | None, path: str | None, step: int, enabled: bool) -> None:
        self.scope = scope
        self.rank = rank
        self.path = Path(path).expanduser().resolve() if path else None
        self.step = step
        self.enabled = enabled and self.path is not None
        self.rows: list[dict[str, Any]] = []

    @classmethod
    def from_env(cls, scope: str, *, rank: int | None = None) -> "CudaStageTimer":
        path = os.getenv("DIFFULEX_CUDA_STAGE_TIMING_PATH", "")
        if not path:
            return cls(scope, rank=rank, path=None, step=0, enabled=False)
        max_steps = int(os.getenv("DIFFULEX_CUDA_STAGE_TIMING_MAX_STEPS", "256"))
        key = f"{scope}:{rank}"
        step = _STAGE_TIMING_COUNTS.get(key, 0)
        _STAGE_TIMING_COUNTS[key] = step + 1
        return cls(scope, rank=rank, path=path, step=step, enabled=step < max_steps)

    @contextmanager
    def stage(self, name: str):
        if not self.enabled:
            yield
            return

        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        start_event = end_event = None
        if device is not None:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        cpu_start = time.perf_counter()
        exc_type = None
        try:
            yield
        except BaseException as exc:
            exc_type = type(exc).__name__
            raise
        finally:
            cpu_end = time.perf_counter()
            if end_event is not None:
                end_event.record()
            self.rows.append(
                {
                    "scope": self.scope,
                    "rank": "" if self.rank is None else self.rank,
                    "pid": os.getpid(),
                    "step": self.step,
                    "stage": name,
                    "cpu_ms": (cpu_end - cpu_start) * 1000.0,
                    "cuda_ms": "",
                    "device": "" if device is None else device,
                    "exception": "" if exc_type is None else exc_type,
                    "_start_event": start_event,
                    "_end_event": end_event,
                }
            )

    def flush(self) -> None:
        if not self.enabled or not self.rows or self.path is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = ["scope", "rank", "pid", "step", "stage", "cpu_ms", "cuda_ms", "device", "exception"]
        file_exists = self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            for row in self.rows:
                start_event = row.pop("_start_event")
                end_event = row.pop("_end_event")
                if start_event is not None and end_event is not None:
                    row["cuda_ms"] = start_event.elapsed_time(end_event)
                writer.writerow(row)
        self.rows.clear()


class TorchProfileSession:
    def __init__(
        self,
        component: str,
        *,
        rank: int | None = None,
        config: ProfilerConfig | dict[str, Any] | None = None,
    ):
        self.config = ProfilerConfig.from_value(config)
        self.enabled = self.config.profiler == "torch"
        self.component = component
        self.rank = rank
        self.prof: torch.profiler.profile | None = None
        self.active = False
        self.running = False
        self.stopped = False
        self.steps = 0
        self.profiled_steps = 0

    @property
    def active_steps(self) -> int:
        return self.config.max_iterations

    def start(self, profile_prefix: str | None = None) -> None:
        if not self.enabled:
            raise RuntimeError(
                "Torch profiling is not enabled. Set profiler_config={'profiler': 'torch', "
                "'torch_profiler_dir': '/path/to/traces'}."
            )
        if self.active:
            logger.debug("Ignoring duplicate profile start for %s.", self.component)
            return
        self.active = True
        self.stopped = False
        self.steps = 0
        self.profiled_steps = 0
        if profile_prefix:
            self.config.run_id = profile_prefix
        if self.config.delay_iterations == 0:
            self._start_profiler()

    def _start_profiler(self) -> None:
        if self.running:
            return
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.config.use_cuda and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        kwargs = dict(
            activities=activities,
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            with_modules=self.config.with_modules,
            acc_events=True,
        )
        try:
            self.prof = torch.profiler.profile(**kwargs)
        except TypeError:
            kwargs.pop("acc_events", None)
            self.prof = torch.profiler.profile(**kwargs)
        self.prof.start()
        self.running = True
        _enable_profile_scopes()

    def step(self) -> None:
        if not self.enabled or not self.active or self.stopped:
            return
        self.steps += 1
        if not self.running and self.steps >= self.config.delay_iterations:
            self._start_profiler()
        if not self.running or self.prof is None:
            return
        self.prof.step()
        self.profiled_steps += 1
        if self.config.max_iterations > 0 and self.profiled_steps >= self.config.max_iterations:
            self.stop()

    def stop(self) -> None:
        if not self.enabled or not self.active or self.stopped:
            return
        self.active = False
        self.stopped = True
        if not self.running:
            return
        assert self.prof is not None
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.prof.stop()
            self._export()
        except Exception as exc:
            logger.warning("Failed to stop/export profiler for %s: %s", self.component, exc)
            self._prefix().with_suffix(".error.txt").write_text(
                f"{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
        finally:
            self.running = False
            _disable_profile_scopes()

    def _prefix(self) -> Path:
        root = Path(self.config.torch_profiler_dir)
        root.mkdir(parents=True, exist_ok=True)
        rank_part = f".rank{self.rank}" if self.rank is not None else ""
        stamp = self.config.run_id or time.strftime("%Y%m%d_%H%M%S")
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
        trace_path = prefix.parent / f"{prefix.name}.trace.json"
        summary_txt = prefix.parent / f"{prefix.name}.summary.txt"
        summary_csv = prefix.parent / f"{prefix.name}.summary.csv"
        summary_json = prefix.parent / f"{prefix.name}.summary.json"
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
