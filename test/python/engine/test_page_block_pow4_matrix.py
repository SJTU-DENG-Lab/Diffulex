import json
import os
import subprocess
import sys

from pathlib import Path

import pytest
import torch


pytestmark = [
    pytest.mark.forked,
    pytest.mark.diffulex_dry_run,
    pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip page/block pow4 matrix in CI (GPU + checkpoint required)",
    ),
]


DEFAULT_MODEL_PATH = "/data1/ckpts/JetLM/SDAR-1.7B-Chat-b32"
MODEL_PATH = os.environ.get("DIFFULEX_TEST_SDAR_MODEL_PATH", DEFAULT_MODEL_PATH)
POW4_VALUES = (4, 8, 16, 32)
PROMPT = (
    "<|im_start|>user\n"
    "Carol and Jennifer are sisters from Los Angeles who love collecting signatures from celebrities. "
    "During their summer break from school, the sisters spend every afternoon collecting signatures. "
    "After five weeks, Carol and Jennifer compare their autograph books, counting up the number of signatures "
    "each sister has collected. Carol has 20 signatures in her book, and Jennifer has 44. The sisters have "
    "three more weeks of summer vacation, and they decide they want to reach 100 signatures between them by "
    "the end of the summer. How many signatures do the sisters need to collect to reach their goal?\n"
    "Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _matrix_cases() -> list[tuple[int, list[int]]]:
    return [
        (block_size, [page_size for page_size in POW4_VALUES if block_size <= page_size])
        for block_size in POW4_VALUES
    ]


def _require_matrix_env(env_name: str) -> None:
    if os.environ.get(env_name) != "1":
        pytest.skip(
            f"Set {env_name}=1 to run the heavy page/block pow4 matrix on a GPU machine."
        )
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not os.path.isdir(MODEL_PATH):
        pytest.skip(f"Model path not found: {MODEL_PATH}")


def _matrix_outdir(tmp_path: Path, label: str) -> Path:
    root = Path(os.environ.get("DIFFULEX_MATRIX_OUTDIR", str(tmp_path)))
    root.mkdir(parents=True, exist_ok=True)
    return root / f"page_block_pow4_{label}.json"


def _run_case(
    *,
    block_size: int,
    page_size: int,
    master_port: int,
    disable_autotune: bool,
) -> dict:
    child_env = os.environ.copy()
    if disable_autotune:
        child_env["DIFFULEX_DISABLE_CHUNKED_PREFILL_AUTOTUNE"] = "1"
    else:
        child_env.pop("DIFFULEX_DISABLE_CHUNKED_PREFILL_AUTOTUNE", None)

    max_tokens = int(os.environ.get("DIFFULEX_MATRIX_MAX_TOKENS", "32"))
    gpu_mem = float(os.environ.get("DIFFULEX_MATRIX_GPU_UTIL", "0.5"))

    script = f"""
import json
import hashlib
from diffulex import Diffulex, SamplingParams

prompt = {PROMPT!r}
sp = SamplingParams(temperature=0.0, max_tokens={max_tokens})
base = dict(
    model_name="sdar",
    decoding_strategy="multi_bd",
    mask_token_id=151669,
    tensor_parallel_size=1,
    data_parallel_size=1,
    gpu_memory_utilization={gpu_mem},
    max_model_len=2048,
    max_num_batched_tokens=4096,
    max_num_reqs=1,
    kv_cache_layout="unified",
    decoding_thresholds={{
        "add_block_threshold": 0.1,
        "semi_complete_threshold": 0.5,
        "decoding_threshold": 0.95,
    }},
    buffer_size=4,
    multi_block_prefix_full=True,
    enforce_eager=True,
    master_port={master_port},
)
llm = Diffulex({MODEL_PATH!r}, **base, block_size={block_size}, page_size={page_size})
try:
    out = llm.generate([prompt], sp, use_tqdm=False).to_benchmark_format()[0]
    payload = {{
        "num_tokens": len(out["token_ids"]),
        "nfe": out["nfe"],
        "text_sha256": hashlib.sha256((out.get("text") or "").encode("utf-8")).hexdigest(),
        "token_sha256": hashlib.sha256(json.dumps(out["token_ids"]).encode("utf-8")).hexdigest(),
    }}
    print(json.dumps(payload, ensure_ascii=False))
finally:
    llm.exit()
"""

    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=child_env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(
            "Pow4 matrix subprocess failed for "
            f"block_size={block_size}, page_size={page_size}, autotune={'off' if disable_autotune else 'on'}.\n"
            f"STDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _run_matrix(*, disable_autotune: bool, tmp_path: Path, label: str) -> None:
    summary: dict[str, dict] = {}
    base_port = int(os.environ.get("DIFFULEX_MATRIX_BASE_PORT", "2800"))
    if not disable_autotune:
        base_port += 200

    port_offset = 0
    for block_size, page_sizes in _matrix_cases():
        group = {}
        for page_size in page_sizes:
            payload = _run_case(
                block_size=block_size,
                page_size=page_size,
                master_port=base_port + port_offset,
                disable_autotune=disable_autotune,
            )
            port_offset += 1
            group[page_size] = payload

        baseline_page = max(page_sizes)
        baseline = group[baseline_page]
        compare = {
            page_size: {
                "same_text_as_baseline": payload["text_sha256"] == baseline["text_sha256"],
                "same_tokens_as_baseline": payload["token_sha256"] == baseline["token_sha256"],
                "same_steps_as_baseline": payload["nfe"] == baseline["nfe"],
                "num_tokens": payload["num_tokens"],
                "nfe": payload["nfe"],
            }
            for page_size, payload in group.items()
        }
        summary[str(block_size)] = {
            "baseline_page": baseline_page,
            "compare": compare,
        }

        mismatches = [
            (page_size, info)
            for page_size, info in compare.items()
            if not (
                info["same_text_as_baseline"]
                and info["same_tokens_as_baseline"]
                and info["same_steps_as_baseline"]
            )
        ]
        assert not mismatches, (
            f"Pow4 matrix mismatch for block_size={block_size}, "
            f"autotune={'off' if disable_autotune else 'on'}: {mismatches}"
        )

    out_path = _matrix_outdir(tmp_path, label)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))


def test_page_block_pow4_matrix_autotune_off(tmp_path: Path) -> None:
    _require_matrix_env("DIFFULEX_RUN_PAGE_BLOCK_POW4_MATRIX")
    _run_matrix(disable_autotune=True, tmp_path=tmp_path, label="autotune_off")


def test_page_block_pow4_matrix_autotune_on(tmp_path: Path) -> None:
    _require_matrix_env("DIFFULEX_RUN_PAGE_BLOCK_POW4_MATRIX_AUTOTUNE")
    _run_matrix(disable_autotune=False, tmp_path=tmp_path, label="autotune_on")
