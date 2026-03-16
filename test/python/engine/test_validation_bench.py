"""
Validation test runner for diffulex_bench with accuracy checking.
Runs benchmark in subprocess with monkey-patched Attention for validation.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
import torch

pytestmark = [
    pytest.mark.forked,
    pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip validation bench in CI (GPU required)",
    ),
]

OUTPUT_BASE = Path(__file__).resolve().parent / "output" / "validation_bench"

CONFIGS = [
    "validation_bench_sdar_bufsz1.yml",
    "validation_bench_sdar_bufsz2.yml",
    "validation_bench_sdar_bufsz4.yml",
]


def _ensure_output_dir() -> Path:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = OUTPUT_BASE / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("config_name", CONFIGS)
def test_validation_bench_gsm8k(config_name):
    """Run GSM8K benchmark with validation enabled."""
    output_dir = _ensure_output_dir()

    wrapper_script = """
import sys
sys.path.insert(0, 'test/python/engine')
from dummy_attn_with_validation import install_validation_hook
install_validation_hook()

from diffulex_bench.main import main
main()
"""

    cmd = [
        sys.executable, "-c", wrapper_script,
        "--config", f"test/python/engine/config/{config_name}",
        "--output-dir", str(output_dir / config_name.replace('.yml', '')),
        "--save-results",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent.parent.parent)

    print(f"\n{'='*80}")
    print(f"Config: {config_name}")
    print(f"{'='*80}")
    print(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        print(f"STDERR:\n{result.stderr}")

    assert result.returncode == 0, f"Benchmark failed with code {result.returncode}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
