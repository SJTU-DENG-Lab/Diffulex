#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${MODEL:-/data/ckpts/google/diffusiongemma-26B-A4B-it}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-29998}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_ATTN}"
CANVAS_LENGTH="${CANVAS_LENGTH:-256}"
DIFFUSION_SAMPLER="${DIFFUSION_SAMPLER:-entropy_bound}"
DIFFUSION_ENTROPY_BOUND="${DIFFUSION_ENTROPY_BOUND:-0.1}"
VLLM_ENV_DIR="${VLLM_ENV_DIR:-/data/jyj/vllm-env}"
if [[ -z "${VLLM_BIN:-}" ]]; then
  if [[ -x "${VLLM_ENV_DIR}/.venv/bin/vllm" ]]; then
    VLLM_BIN="${VLLM_ENV_DIR}/.venv/bin/vllm"
  elif [[ -x "${PROJECT_ROOT}/.venv/bin/vllm" ]]; then
    VLLM_BIN="${PROJECT_ROOT}/.venv/bin/vllm"
  else
    VLLM_BIN="vllm"
  fi
fi
VLLM_EXEC="${VLLM_BIN}"

export CUDA_VISIBLE_DEVICES

if [[ "${SKIP_DIFFUSION_GEMMA_PREFLIGHT:-0}" != "1" ]]; then
  if [[ -n "${VLLM_PYTHON:-}" ]]; then
    :
  elif [[ "${VLLM_BIN}" == */bin/vllm && -x "$(dirname "$(dirname "${VLLM_BIN}")")/bin/python" ]]; then
    VLLM_PYTHON="$(dirname "$(dirname "${VLLM_BIN}")")/bin/python"
  elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    VLLM_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
  else
    VLLM_PYTHON="python"
  fi
  "${VLLM_PYTHON}" - <<'PY'
import sys
try:
    from vllm.model_executor.models.registry import ModelRegistry
    archs = set(ModelRegistry.get_supported_archs())
except Exception as exc:
    print(f"Failed to inspect vLLM model registry: {exc}", file=sys.stderr)
    raise SystemExit(2)
if "DiffusionGemmaForBlockDiffusion" not in archs:
    print(
        "This vLLM install does not support DiffusionGemmaForBlockDiffusion. "
        "Install or activate a vLLM build with diffusion_gemma support, then rerun.",
        file=sys.stderr,
    )
    raise SystemExit(2)
PY
fi

EXTRA_ARGS=()
if "${VLLM_EXEC}" serve --help=all 2>/dev/null | grep -q -- "--diffusion-config"; then
  EXTRA_ARGS+=(--diffusion-config "{\"canvas_length\": ${CANVAS_LENGTH}}")
fi

unset VLLM_BIN VLLM_PYTHON VLLM_ENV_DIR

exec "${VLLM_EXEC}" serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --generation-config vllm \
  --hf-overrides "{\"diffusion_sampler\": \"${DIFFUSION_SAMPLER}\", \"diffusion_entropy_bound\": ${DIFFUSION_ENTROPY_BOUND}}" \
  --enable-chunked-prefill \
  "${EXTRA_ARGS[@]}"
