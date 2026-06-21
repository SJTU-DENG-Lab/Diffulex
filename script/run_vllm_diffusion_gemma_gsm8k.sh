#!/usr/bin/env bash
set -euo pipefail

# vLLM DiffusionGemma GSM8K lm-eval runner.
#
# Defaults run the smoke config. For full GSM8K:
#   CONFIG_PATH=examples/engine_lm_eval/configs/vllm_diffusion_gemma_gsm8k_full.yml \
#     script/run_vllm_diffusion_gemma_gsm8k.sh
#
# Useful overrides:
#   CUDA_VISIBLE_DEVICES=0 PORT=29998 DATASET_LIMIT=100 script/run_vllm_diffusion_gemma_gsm8k.sh
#   KEEP_SERVER=1 script/run_vllm_diffusion_gemma_gsm8k.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-examples/engine_lm_eval/configs/vllm_diffusion_gemma_gsm8k_smoke.yml}"
MODEL="${MODEL:-/data/ckpts/google/diffusiongemma-26B-A4B-it}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL}}"
HOST="${HOST:-127.0.0.1}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
PORT="${PORT:-29998}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LOG_DIR="${LOG_DIR:-logs/vllm_diffusion_gemma_gsm8k}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SERVER_LOG="${SERVER_LOG:-${LOG_DIR}/server_${STAMP}.log}"
EVAL_LOG="${EVAL_LOG:-${LOG_DIR}/eval_${STAMP}.log}"
WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-900}"
KEEP_SERVER="${KEEP_SERVER:-0}"

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

mkdir -p "${LOG_DIR}"

cleanup() {
  if [[ "${KEEP_SERVER}" != "1" && -n "${SERVER_PID:-}" ]]; then
    if kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      kill "${SERVER_PID}" >/dev/null 2>&1 || true
      wait "${SERVER_PID}" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT

echo "================================================================"
echo "vLLM DiffusionGemma GSM8K bench"
echo "================================================================"
echo "Config:       ${CONFIG_PATH}"
echo "Model:        ${MODEL}"
echo "Served name:  ${SERVED_MODEL_NAME}"
echo "Endpoint:     http://${HOST}:${PORT}"
echo "GPUs:         ${CUDA_VISIBLE_DEVICES}"
echo "Server log:   ${SERVER_LOG}"
echo "Eval log:     ${EVAL_LOG}"
echo "Keep server:  ${KEEP_SERVER}"
echo "================================================================"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
MODEL="${MODEL}" \
SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
HOST="${SERVER_HOST}" \
PORT="${PORT}" \
examples/engine_lm_eval/launch_vllm.sh >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "Started vLLM server pid=${SERVER_PID}; waiting for /health ..."
deadline=$((SECONDS + WAIT_TIMEOUT_S))
until curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "vLLM server exited during startup. Last server log lines:" >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for vLLM server after ${WAIT_TIMEOUT_S}s. Last server log lines:" >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  sleep 5
done

echo "vLLM server is healthy. Starting lm-eval ..."

CONFIG_PATH="${CONFIG_PATH}" \
PYTHON_BIN="${PYTHON_BIN}" \
examples/engine_lm_eval/eval.sh 2>&1 | tee "${EVAL_LOG}"

echo "----------------------------------------------------------------"
echo "Latest client-side TPS summary"
CFG_PATH="${CONFIG_PATH}" "${PYTHON_BIN}" - <<'PY' || true
import json
import os
import re
from pathlib import Path

from examples.engine_lm_eval.config import BenchmarkConfig


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_.")
    return value or "unknown"


cfg = BenchmarkConfig.from_yaml(os.environ["CFG_PATH"])
root = Path(cfg.eval.output_dir).expanduser() / slugify(cfg.endpoint.engine_name)
stats_files = sorted(root.glob("**/*_stats.json"), key=lambda p: p.stat().st_mtime)
if not stats_files:
    print(f"No stats file found under {root}")
    raise SystemExit(0)

stats_path = stats_files[-1]
stats = json.loads(stats_path.read_text(encoding="utf-8"))
keys = [
    "total_samples",
    "total_completion_tokens",
    "total_time",
    "aggregate_e2e_tps_tok_s",
    "avg_e2e_tps_tok_s",
    "dtps_tok_s",
    "aggregate_decode_tps_tok_s",
    "visible_decode_tps_tok_s",
    "avg_visible_decode_tps_tok_s",
    "chunk_tpf",
]
print(f"stats: {stats_path}")
for key in keys:
    if key in stats:
        print(f"{key}: {stats[key]}")
PY
echo "----------------------------------------------------------------"

echo "Eval complete. Logs:"
echo "  server: ${SERVER_LOG}"
echo "  eval:   ${EVAL_LOG}"
