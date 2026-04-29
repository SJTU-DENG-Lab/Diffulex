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

mkdir -p "${PROFILE_ROOT}" logs

DIFFULEX_RUNTIME_ARGS=(
  --enable-prefill-cudagraph
  --prefill-cudagraph-max-len "${PREFILL_CUDAGRAPH_MAX_LEN:-0}"
  --enable-full-static-runner
  --enable-torch-compile
  --no-enable-cudagraph-torch-compile
)
if [[ "${DIFFULEX_PROFILE_EAGER:-1}" == "1" ]]; then
  DIFFULEX_RUNTIME_ARGS=(
    --no-enable-prefill-cudagraph
    --no-enable-full-static-runner
    --no-enable-torch-compile
  )
fi

echo "[diffulex] profiling ${SAMPLE_LIMIT} samples, dataset=${DATASET}, model=${MODEL_PATH}"
DIFFULEX_DMAX_SAMPLER_FAST=1 \
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
  --config diffulex_bench/configs/llada2_mini_dmax_gsm8k.yml \
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
  --attn-impl "${ATTN_IMPL:-triton}" \
  --moe-gemm-impl "${MOE_GEMM_IMPL:-vllm_modular}" \
  "${DIFFULEX_RUNTIME_ARGS[@]}" \
  --output-dir "${PROFILE_ROOT}/diffulex_results" \
  --no-use-run-subdirectory \
  --max-nfe "${MAX_NFE:-1024}"

.venv/bin/python script/summarize_torch_profiles.py "${PROFILE_ROOT}" --top-n "${PROFILE_TOP_N:-60}"
echo "[done] DiffuLEx profile root: ${PROFILE_ROOT}"
