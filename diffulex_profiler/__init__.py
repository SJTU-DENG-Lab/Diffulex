"""
Diffulex Profiler - Modular profiling framework for performance analysis of Diffulex inference engine
"""

from diffulex_profiler.profiler import DiffulexProfiler, ProfilerConfig
from diffulex_profiler.metrics import (
    PerformanceMetrics,
    collect_gpu_metrics,
    collect_cpu_metrics,
    collect_memory_metrics,
)
from diffulex_profiler.backends import ProfilerBackend, SimpleTimerBackend
try:
    from diffulex_profiler.backends import VizTracerBackend
except (ImportError, AttributeError):
    VizTracerBackend = None  # type: ignore[misc, assignment]
try:
    from diffulex_profiler.backends import PyTorchProfilerBackend
except (ImportError, AttributeError):
    PyTorchProfilerBackend = None  # type: ignore[misc, assignment]
from diffulex_profiler.exporters import (
    ProfilerExporter,
    JSONExporter,
    CSVExporter,
    SummaryExporter,
)

__all__ = [
    "DiffulexProfiler",
    "ProfilerConfig",
    "PerformanceMetrics",
    "collect_gpu_metrics",
    "collect_cpu_metrics",
    "collect_memory_metrics",
    "ProfilerBackend",
    "SimpleTimerBackend",
    "ProfilerExporter",
    "JSONExporter",
    "CSVExporter",
    "SummaryExporter",
]
if VizTracerBackend is not None:
    __all__.append("VizTracerBackend")
if PyTorchProfilerBackend is not None:
    __all__.append("PyTorchProfilerBackend")

