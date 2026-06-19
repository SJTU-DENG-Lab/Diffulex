#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
PROFILE_ROOT="${PROFILE_ROOT:-benchmark_results/profile_diffulex_multi_bd_${STAMP}}"
DATASET="${DATASET:-gsm8k_diffulex_dmax_chat}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-10}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DIFFULEX_TP_SIZE="${DIFFULEX_TP_SIZE:-1}"

DEFAULT_MODEL="/root/workspace/jyj/workspaces/MultiBD/multi_bd_train/multibd_ckpt/llada2_dmax_mbd_oput_coder_v2_60k_e4_bufsz2/checkpoints/global_step_8000/hf_ckpt_convert"
ALT_MODEL="/inspire/hdd/global_user/yangyi-253108120173/inspire_shared/mount/advanced-machine-learning-and-deep-learning-applications/jyj/multibd/llada2_dmax_mbd_oput_coder_v2_60k_e4_bufsz2/checkpoints/global_step_8000/hf_ckpt_convert"
if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ -d "${DEFAULT_MODEL}" ]]; then
    MODEL_PATH="${DEFAULT_MODEL}"
  else
    MODEL_PATH="${ALT_MODEL}"
  fi
fi
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
MASTER_PORT="${MASTER_PORT:-$(.venv/bin/python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')}"

mkdir -p "${PROFILE_ROOT}" logs

export VLLM_TUNED_CONFIG_FOLDER="${VLLM_TUNED_CONFIG_FOLDER:-${REPO_ROOT}/diffulex_bench/vllm_tuned_configs}"

DIFFULEX_ENABLE_CUDAGRAPH="${DIFFULEX_ENABLE_CUDAGRAPH:-1}"
DIFFULEX_ENABLE_TORCH_COMPILE="${DIFFULEX_ENABLE_TORCH_COMPILE:-1}"

DIFFULEX_RUNTIME_ARGS=(
  --prefill-cudagraph-max-len "${PREFILL_CUDAGRAPH_MAX_LEN:-0}"
  --no-enable-cudagraph-torch-compile
)
if [[ "${DIFFULEX_ENABLE_CUDAGRAPH}" == "1" ]]; then
  DIFFULEX_RUNTIME_ARGS+=(--enable-prefill-cudagraph --enable-full-static-runner)
else
  DIFFULEX_RUNTIME_ARGS+=(--no-enable-prefill-cudagraph --no-enable-full-static-runner)
fi
if [[ "${DIFFULEX_ENABLE_TORCH_COMPILE}" == "1" ]]; then
  DIFFULEX_RUNTIME_ARGS+=(--enable-torch-compile)
else
  DIFFULEX_RUNTIME_ARGS+=(--no-enable-torch-compile)
fi
if [[ "${DIFFULEX_PROFILE_EAGER:-0}" == "1" ]]; then
  DIFFULEX_RUNTIME_ARGS=(
    --no-enable-prefill-cudagraph
    --no-enable-full-static-runner
    --no-enable-torch-compile
  )
fi

echo "[diffulex] profiling ${SAMPLE_LIMIT} samples, dataset=${DATASET}, model=${MODEL_PATH}"
DIFFULEX_PROFILE_DIR="${PROFILE_ROOT}/diffulex" \
DIFFULEX_PROFILE_RUN_ID="diffulex_multi_bd_buf1" \
DIFFULEX_PROFILE_ACTIVE_STEPS="${DIFFULEX_PROFILE_ACTIVE_STEPS:-256}" \
DIFFULEX_PROFILE_RECORD_SHAPES="${DIFFULEX_PROFILE_RECORD_SHAPES:-0}" \
DIFFULEX_PROFILE_MEMORY="${DIFFULEX_PROFILE_MEMORY:-0}" \
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
HF_ALLOW_CODE_EVAL=1 \
.venv/bin/python -m diffulex_bench.main \
  --log-file "${PROFILE_ROOT}/diffulex_multi_bd.log" \
  --log-level INFO \
  --config "${DIFFULEX_CONFIG_PATH:-diffulex_bench/configs/llada2_mini_gsm8k.yml}" \
  --model-path "${MODEL_PATH}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --model-name llada2_mini \
  --decoding-strategy multi_bd \
  --sampling-mode "${SAMPLING_MODE:-naive}" \
  --tensor-parallel-size "${DIFFULEX_TP_SIZE}" \
  --data-parallel-size 1 \
  --dataset "${DATASET}" \
  --dataset-limit "${SAMPLE_LIMIT}" \
  --temperature 0.0 \
  --max-tokens "${MAX_TOKENS:-4096}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4096}" \
  --max-num-reqs 1 \
  --block-size "${BLOCK_SIZE:-32}" \
  --buffer-size 1 \
  --attn-impl "${ATTN_IMPL:-triton_grouped}" \
  --moe-gemm-impl "${MOE_GEMM_IMPL:-vllm_modular}" \
  --moe-dispatcher-backend "${MOE_DISPATCHER_BACKEND:-standard}" \
  --enable-vllm-layers \
  --kv-cache-layout "${KV_CACHE_LAYOUT:-unified}" \
  --engine-arg "distributed_backend=${DISTRIBUTED_BACKEND:-nccl}" \
  --engine-arg "master_port=${MASTER_PORT}" \
  "${DIFFULEX_RUNTIME_ARGS[@]}" \
  --output-dir "${PROFILE_ROOT}/diffulex_results" \
  --no-use-run-subdirectory \
  --max-nfe "${MAX_NFE:-1024}"

.venv/bin/python script/summarize_torch_profiles.py "${PROFILE_ROOT}" --top-n "${PROFILE_TOP_N:-60}"
echo "[done] DiffuLEx profile root: ${PROFILE_ROOT}"
