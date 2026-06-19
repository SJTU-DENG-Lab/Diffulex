from __future__ import annotations

import pytest
import torch

from diffulex.profiling import ProfilerConfig, TorchProfileSession, profile_scopes_enabled, record_function


def test_profiler_config_requires_trace_dir():
    with pytest.raises(ValueError, match="torch_profiler_dir must be set"):
        ProfilerConfig(profiler="torch")


def test_record_function_is_noop_when_profiler_is_inactive():
    assert profile_scopes_enabled() is False
    with record_function("diffulex.test.noop"):
        pass


def test_torch_profile_session_exports_trace_and_summaries(tmp_path):
    session = TorchProfileSession(
        "unit",
        config=ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=str(tmp_path),
            max_iterations=1,
            run_id="profile-test",
            use_cuda=False,
        ),
    )

    session.start()
    with record_function("diffulex.test.profiled_scope"):
        x = torch.ones(4)
        _ = x + 1
    session.step()

    assert session.stopped is True
    assert (tmp_path / "profile-test.unit.trace.json").exists()
    assert (tmp_path / "profile-test.unit.summary.txt").exists()
    assert (tmp_path / "profile-test.unit.summary.csv").exists()
    assert (tmp_path / "profile-test.unit.summary.json").exists()
