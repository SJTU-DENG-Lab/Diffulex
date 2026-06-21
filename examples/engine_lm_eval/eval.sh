#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-examples/engine_lm_eval/configs/example.yml}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="python"
fi

optional_args=()
if [[ -n "${DATASET_LIMIT:-}" ]]; then optional_args+=(--dataset-limit "${DATASET_LIMIT}"); fi
if [[ -n "${MAX_TOKENS:-}" ]]; then optional_args+=(--max-tokens "${MAX_TOKENS}"); fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then optional_args+=(--output-dir "${OUTPUT_DIR}"); fi
if [[ -n "${BASE_URL:-}" ]]; then optional_args+=(--base-url "${BASE_URL}"); fi
if [[ -n "${BATCH_SIZE:-}" ]]; then optional_args+=(--batch-size "${BATCH_SIZE}"); fi

exec "${PYTHON_BIN}" -m examples.engine_lm_eval.main --config "${CONFIG_PATH}" "${optional_args[@]}"
