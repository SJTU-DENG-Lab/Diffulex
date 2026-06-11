#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DIR_NAME="hf_ckpt"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/inspire/hdd/global_user/yangyi-253108120173/inspire_shared/mount/advanced-machine-learning-and-deep-learning-applications/jyj/multibd/sdar_8Bb4_mbd_v2_sft_code_sdar60k_br2_bufsz4/checkpoints/}"
CONFIG_PATH="${CONFIG_PATH:-diffulex_bench/configs/sdar_chat_gsm8k.yml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_results/sdar_code_b4}"
LOG_DIR="${LOG_DIR:-logs/dmax_all_convert}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_REQS="${MAX_NUM_REQS:-128}"
BLOCK_SIZE="${BLOCK_SIZE:-4}"
BUFFER_SIZE="${BUFFER_SIZE:-4}"
MAX_NFE="${MAX_NFE:-1024}"
TOKEN_STABILITY_THRESHOLD="${TOKEN_STABILITY_THRESHOLD:-0.0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

DATASETS=(
  humaneval_plus_sdar                                                                                                                                                                                                            
  mbpp_plus_sdar
)

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

mapfile -t MODEL_DIRS < <(
  find "${CHECKPOINTS_DIR}" -maxdepth 2 -type d -name $DIR_NAME \
    | sort -V
)

if [[ "${#MODEL_DIRS[@]}" -eq 0 ]]; then
  echo "No $DIR_NAME directories found under ${CHECKPOINTS_DIR}" >&2
  exit 1
fi

echo "Found ${#MODEL_DIRS[@]} converted checkpoints:"
printf '  %s\n' "${MODEL_DIRS[@]}"
echo

for model_path in "${MODEL_DIRS[@]}"; do
  step_name="$(basename "$(dirname "${model_path}")")"

  for dataset in "${DATASETS[@]}"; do
    output_dir="${OUTPUT_ROOT}/${dataset}/${step_name}"
    log_file="${LOG_DIR}/${dataset}_${step_name}.log"
    stats_file="${output_dir}/diffulex_stats.json"

    if [[ "${FORCE_RERUN}" != "1" && -f "${stats_file}" ]]; then
      echo "[skip] ${dataset} ${step_name}: ${stats_file} exists"
      continue
    fi

    mkdir -p "${output_dir}" "$(dirname "${log_file}")"

    echo "================================================================"
    echo "[run] dataset=${dataset}"
    echo "[run] step=${step_name}"
    echo "[run] model_path=${model_path}"
    echo "[run] output_dir=${output_dir}"
    echo "[run] log_file=${log_file}"
    echo "================================================================"

    DIFFULEX_DMAX_SAMPLER_FAST=1 \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    HF_ALLOW_CODE_EVAL=1 \
    "${PYTHON_BIN}" -m diffulex_bench.main \
      --log-file "${log_file}" \
      --log-level INFO \
      --config "${CONFIG_PATH}" \
      --model-path "${model_path}" \
      --model-name sdar \
      --decoding-strategy multi_bd \
      --sampling-mode naive \
      --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
      --data-parallel-size "${DATA_PARALLEL_SIZE}" \
      --dataset "${dataset}" \
      --temperature 0.0 \
      --max-tokens "${MAX_TOKENS}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --max-num-reqs "${MAX_NUM_REQS}" \
      --block-size "${BLOCK_SIZE}" \
      --buffer-size "${BUFFER_SIZE}" \
      --token-stability-threshold "${TOKEN_STABILITY_THRESHOLD}" \
      --attn-impl triton \
      --moe-gemm-impl vllm_modular \
      --enable-prefill-cudagraph \
      --prefill-cudagraph-max-len 0 \
      --enable-full-static-runner \
      --enable-torch-compile \
      --no-enable-cudagraph-torch-compile \
      --output-dir "${output_dir}" \
      --no-use-run-subdirectory \
      --max-nfe "${MAX_NFE}"
  done
done

echo "All requested DMax code-task benchmarks finished."
