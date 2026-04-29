#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-/root/data/ckpts/inclusionAI/LLaDA2.0-mini}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TP_SIZE="${SGLANG_TP_SIZE:-1}"
PORT="${SGLANG_PORT:-29998}"
SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-1}"

SGLANG_GRAPH_ARGS=()
if [[ "${SGLANG_DISABLE_CUDA_GRAPH}" == "1" ]]; then
  SGLANG_GRAPH_ARGS+=(--disable-cuda-graph)
else
  SGLANG_GRAPH_ARGS+=(--cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS:-16}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-benchmark_results/sglang_torch_profiler}" \
SGLANG_PROFILE_WITH_STACK="${SGLANG_PROFILE_WITH_STACK:-0}" \
SGLANG_PROFILE_RECORD_SHAPES="${SGLANG_PROFILE_RECORD_SHAPES:-0}" \
python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --dllm-algorithm "${SGLANG_DLLM_ALGORITHM:-LowConfidence}" \
  --dllm-algorithm-config "${SGLANG_DLLM_ALGORITHM_CONFIG:-examples/engine_lm_eval/configs/low_confidence.yml}" \
  --trust-remote-code \
  --tp-size "${TP_SIZE}" \
  "${SGLANG_GRAPH_ARGS[@]}" \
  --max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-16}" \
  --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.7}" \
  --host "${SGLANG_HOST:-0.0.0.0}" \
  --port "${PORT}"
