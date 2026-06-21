#!/usr/bin/env bash
set -euo pipefail

# LLaDA2.0-mini GSM8K Diffulex bench.
#
# Defaults:
#   - LLaDA2.0-mini at /data/ckpts/inclusionAI/LLaDA2.0-mini
#   - gsm8k_llada2 full split
#   - single-request TP1 engine config from diffulex_bench/configs/llada2_mini_gsm8k.yml
#   - output under benchmark_results/llada2_mini_gsm8k/run_*
#
# Examples:
#   script/run_llada2_mini_gsm8k.sh
#   DATASET_LIMIT=200 script/run_llada2_mini_gsm8k.sh
#   CUDA_VISIBLE_DEVICES=0 OUTPUT_DIR=benchmark_results/my_llada2_run script/run_llada2_mini_gsm8k.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-diffulex_bench/configs/llada2_mini_gsm8k.yml}"
if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-benchmark_results/llada2_mini_gsm8k}"
LOG_DIR="${LOG_DIR:-logs/llada2_mini_gsm8k}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_${STAMP}.log}"

MASTER_PORT="${MASTER_PORT:-$("${PYTHON_BIN}" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')}"

# Keep CUDA/NCCL runtime resolution consistent with the project venv when present.
if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import site
print(site.getsitepackages()[0])
PY
then
  VENV_SITE="$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')"
  export LD_LIBRARY_PATH="${VENV_SITE}/nvidia/cu13/lib:${VENV_SITE}/nvidia/nccl/lib:${VENV_SITE}/nvidia/nvshmem/lib:${VENV_SITE}/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
fi

export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET="${NCCL_NET:-Socket}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IGNORE_DISABLED_P2P="${NCCL_IGNORE_DISABLED_P2P:-1}"
export VLLM_TUNED_CONFIG_FOLDER="${VLLM_TUNED_CONFIG_FOLDER:-${PROJECT_ROOT}/diffulex_bench/vllm_tuned_configs}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-ERROR}"
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"

optional_args=()
if [[ -n "${DATASET_LIMIT:-}" ]]; then optional_args+=(--dataset-limit "${DATASET_LIMIT}"); fi
if [[ -n "${MAX_TOKENS:-}" ]]; then optional_args+=(--max-tokens "${MAX_TOKENS}"); fi
if [[ -n "${MAX_NFE:-}" ]]; then optional_args+=(--max-nfe "${MAX_NFE}"); fi
if [[ -n "${MODEL_PATH:-}" ]]; then optional_args+=(--model-path "${MODEL_PATH}"); fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "================================================================"
echo "LLaDA2.0-mini GSM8K bench"
echo "================================================================"
echo "Config:        ${CONFIG_PATH}"
echo "Python:        ${PYTHON_BIN}"
echo "GPUs:          ${CUDA_VISIBLE_DEVICES}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Log file:      ${LOG_FILE}"
echo "Master port:   ${MASTER_PORT}"
echo "Dataset limit: ${DATASET_LIMIT:-full}"
echo "Max tokens:    ${MAX_TOKENS:-config}"
echo "Max NFE:       ${MAX_NFE:-config}"
echo "================================================================"

exec "${PYTHON_BIN}" -m diffulex_bench.main \
  --log-file "${LOG_FILE}" \
  --log-level INFO \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --engine-arg "master_port=${MASTER_PORT}" \
  "${optional_args[@]}"
