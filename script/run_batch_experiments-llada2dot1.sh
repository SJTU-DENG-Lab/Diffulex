#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# LLaDA2.1-mini batch experiments — JointThreshold edit sampler
# ============================================================
#
# Usage:
#   ./script/run_batch_experiments-llada2dot1.sh
#
#   DRY_RUN=1 ./script/run_batch_experiments-llada2dot1.sh


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ============================================================
# Experiment grid
# ============================================================

# Datasets: use existing llada2 task YAMLs
DATASETS=(
  gsm8k_llada2
  math500_llada2
  # humaneval_llada2
  # humaneval_plus_llada2
  # mbpp_sanitized_llada2
  # mbpp_plus_llada2
)

MODEL_PATHS=(
  /root/data/ckpts/inclusionAI/LLaDA2.1-mini
)

# --- Model name (must be llada2dot1_mini for edit mode) ---
MODEL_NAME="${MODEL_NAME:-llada2dot1_mini}"

# --- JointThreshold edit sampler ---
DECODING_STRATEGY="${DECODING_STRATEGY:-multi_bd}"
SAMPLING_MODE="${SAMPLING_MODE:-edit}"

# --- Block / buffer sizes ---
BLOCK_SIZES=(32)
BUFFER_SIZES=(1)

# --- Thresholds ---
ADD_BLOCK_THRESHOLDS=(0.1)
SEMI_COMPLETE_THRESHOLDS=(0.9)
ACCEPT_THRESHOLDS=(0.95)
EDIT_THRESHOLDS=(0.5)
REMASK_THRESHOLDS=(0.4)
TOKEN_STABILITY_THRESHOLDS=(0.0)

# ============================================================
# Fixed parameters
# ============================================================

CONFIG_PATH="${CONFIG_PATH:-diffulex_bench/configs/llada2_mini_gsm8k.yml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"

MAX_TOKENS="${MAX_TOKENS:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_REQS="${MAX_NUM_REQS:-128}"
MAX_NFE="${MAX_NFE:-2048}"

MAX_POST_EDIT_STEPS="${MAX_POST_EDIT_STEPS:-16}"

ATTN_IMPL="${ATTN_IMPL:-triton}"
MOE_GEMM_IMPL="${MOE_GEMM_IMPL:-vllm_modular}"
ENABLE_PREFILL_CUDAGRAPH="${ENABLE_PREFILL_CUDAGRAPH:-true}"
ENABLE_FULL_STATIC_RUNNER="${ENABLE_FULL_STATIC_RUNNER:-true}"
ENABLE_CUDAGRAPH_TORCH_COMPILE="${ENABLE_CUDAGRAPH_TORCH_COMPILE:-false}"

# --- Output ---
OUTPUT_BASE="${OUTPUT_BASE:-benchmark_results/batch_experiments_${MODEL_NAME}}"
RUN_ID="${RUN_ID:-run_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_BASE}/${RUN_ID}"
LOG_DIR="${LOG_DIR:-logs/batch_experiments_${MODEL_NAME}/${RUN_ID}}"

# --- Misc ---
FORCE_RERUN="${FORCE_RERUN:-0}"
DRY_RUN="${DRY_RUN:-0}"
CARTESIAN_MODE="${CARTESIAN_MODE:-cross}"

# ============================================================
# Build experiment list
# ============================================================

EXPERIMENTS=()

if [[ "${CARTESIAN_MODE}" == "cross" ]]; then
  for model_path in "${MODEL_PATHS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      for block_size in "${BLOCK_SIZES[@]}"; do
        for buffer_size in "${BUFFER_SIZES[@]}"; do
          for add_th in "${ADD_BLOCK_THRESHOLDS[@]}"; do
            for semi_th in "${SEMI_COMPLETE_THRESHOLDS[@]}"; do
              for acc_th in "${ACCEPT_THRESHOLDS[@]}"; do
                for edit_th in "${EDIT_THRESHOLDS[@]}"; do
                  for rem_th in "${REMASK_THRESHOLDS[@]}"; do
                    for stab_th in "${TOKEN_STABILITY_THRESHOLDS[@]}"; do
                      EXPERIMENTS+=("${model_path}|${dataset}|${block_size}|${buffer_size}|${add_th}|${semi_th}|${acc_th}|${edit_th}|${rem_th}|${stab_th}")
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
else
  num=${#MODEL_PATHS[@]}
  for i in $(seq 0 $((num - 1))); do
    model_path="${MODEL_PATHS[$i]}"
    dataset="${DATASETS[$i]:-${DATASETS[0]}}"
    block_size="${BLOCK_SIZES[$i]:-${BLOCK_SIZES[0]}}"
    buffer_size="${BUFFER_SIZES[$i]:-${BUFFER_SIZES[0]}}"
    add_th="${ADD_BLOCK_THRESHOLDS[$i]:-${ADD_BLOCK_THRESHOLDS[0]}}"
    semi_th="${SEMI_COMPLETE_THRESHOLDS[$i]:-${SEMI_COMPLETE_THRESHOLDS[0]}}"
    acc_th="${ACCEPT_THRESHOLDS[$i]:-${ACCEPT_THRESHOLDS[0]}}"
    edit_th="${EDIT_THRESHOLDS[$i]:-${EDIT_THRESHOLDS[0]}}"
    rem_th="${REMASK_THRESHOLDS[$i]:-${REMASK_THRESHOLDS[0]}}"
    stab_th="${TOKEN_STABILITY_THRESHOLDS[$i]:-${TOKEN_STABILITY_THRESHOLDS[0]}}"
    EXPERIMENTS+=("${model_path}|${dataset}|${block_size}|${buffer_size}|${add_th}|${semi_th}|${acc_th}|${edit_th}|${rem_th}|${stab_th}")
  done
fi

# ============================================================
# Print plan
# ============================================================

echo "================================================================"
echo "LLaDA2.1-mini JointThreshold Batch Experiment Plan"
echo "================================================================"
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo "Cartesian mode:    ${CARTESIAN_MODE}"
echo "Model name:        ${MODEL_NAME}"
echo "Decoding strategy: ${DECODING_STRATEGY}"
echo "Sampling mode:     ${SAMPLING_MODE}"
echo "Config:            ${CONFIG_PATH}"
echo "GPUs:              ${CUDA_VISIBLE_DEVICES}"
echo "TP size:           ${TENSOR_PARALLEL_SIZE}"
echo "Max post-edit:     ${MAX_POST_EDIT_STEPS}"
echo "Output base:       ${OUTPUT_BASE}"
echo "Run ID:            ${RUN_ID}"
echo "Output root:       ${OUTPUT_ROOT}"
echo "Log dir:           ${LOG_DIR}"
echo "Dry run:           ${DRY_RUN}"
echo "Force rerun:       ${FORCE_RERUN}"
echo "----------------------------------------------------------------"
for i in "${!EXPERIMENTS[@]}"; do
  IFS='|' read -r mp ds bs buf ath sth acth edth rth stth <<< "${EXPERIMENTS[$i]}"
  model_short="$(basename "${mp}")"
  printf "  [%2d] model=%-28s ds=%-30s bs=%s buf=%s ath=%s sth=%s acth=%s edth=%s\n" \
    "$i" "${model_short}" "${ds}" "${bs}" "${buf}" "${ath}" "${sth}" "${acth}" "${edth}"
done
echo "================================================================"
echo ""

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] Done. Set DRY_RUN=0 to execute."
  exit 0
fi

# ============================================================
# Helper: write experiment config JSON
# ============================================================

_write_exp_config() {
  local dir="$1" mp="$2" ds="$3" bs="$4" buf="$5" ath="$6" sth="$7" acth="$8" edth="$9" rth="${10}" stth="${11}"
  cat > "${dir}/experiment_config.json" <<JSON
{
  "model_path": "${mp}",
  "model_name": "${MODEL_NAME}",
  "decoding_strategy": "${DECODING_STRATEGY}",
  "sampling_mode": "${SAMPLING_MODE}",
  "dataset": "${ds}",
  "block_size": ${bs},
  "buffer_size": ${buf},
  "thresholds": {
    "add_block_threshold": ${ath},
    "semi_complete_threshold": ${sth},
    "accept_threshold": ${acth},
    "edit_threshold": ${edth},
    "remask_threshold": ${rth},
    "token_stability_threshold": ${stth}
  },
  "engine": {
    "tensor_parallel_size": ${TENSOR_PARALLEL_SIZE},
    "data_parallel_size": ${DATA_PARALLEL_SIZE},
    "max_tokens": ${MAX_TOKENS},
    "max_model_len": ${MAX_MODEL_LEN},
    "max_nfe": ${MAX_NFE},
    "max_post_edit_steps": ${MAX_POST_EDIT_STEPS},
    "max_num_batched_tokens": ${MAX_NUM_BATCHED_TOKENS},
    "max_num_reqs": ${MAX_NUM_REQS},
    "attn_impl": "${ATTN_IMPL}",
    "moe_gemm_impl": "${MOE_GEMM_IMPL}",
    "enable_prefill_cudagraph": ${ENABLE_PREFILL_CUDAGRAPH},
    "enable_full_static_runner": ${ENABLE_FULL_STATIC_RUNNER},
    "enable_cudagraph_torch_compile": ${ENABLE_CUDAGRAPH_TORCH_COMPILE}
  }
}
JSON
}

# ============================================================
# Run
# ============================================================

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

run_count=0
skip_count=0
fail_count=0

for i in "${!EXPERIMENTS[@]}"; do
  IFS='|' read -r model_path dataset block_size buffer_size add_th semi_th acc_th edit_th rem_th stab_th <<< "${EXPERIMENTS[$i]}"

  model_short="$(basename "${model_path}")"

  exp_dir="${OUTPUT_ROOT}/${dataset}/${model_short}/bs${block_size}_buf${buffer_size}"
  log_file="${LOG_DIR}/${dataset}__${model_short}__bs${block_size}_buf${buffer_size}.log"
  stats_file="${exp_dir}/diffulex_stats.json"

  if [[ "${FORCE_RERUN}" != "1" && -f "${stats_file}" ]]; then
    echo "[skip] #${i}: ${stats_file} exists"
    ((skip_count++)) || true
    continue
  fi

  echo "========================================================================"
  echo "[run]  #${i}/${#EXPERIMENTS[@]}"
  echo "[run]  model      = ${model_short}"
  echo "[run]  model_path = ${model_path}"
  echo "[run]  dataset    = ${dataset}"
  echo "[run]  block_size = ${block_size}  buffer_size = ${buffer_size}"
  echo "[run]  thresholds = ath=${add_th} sth=${semi_th} acth=${acc_th} edth=${edit_th} rth=${rem_th} stth=${stab_th}"
  echo "[run]  output_dir = ${exp_dir}"
  echo "[run]  log_file   = ${log_file}"
  echo "========================================================================"

  mkdir -p "${exp_dir}" "$(dirname "${log_file}")"

  _write_exp_config "${exp_dir}" "${model_path}" "${dataset}" \
    "${block_size}" "${buffer_size}" "${add_th}" "${semi_th}" "${acc_th}" "${edit_th}" "${rem_th}" "${stab_th}"

  CUDAGRAPH_FLAG=()
  if [[ "${ENABLE_PREFILL_CUDAGRAPH}" == "true" ]]; then
    CUDAGRAPH_FLAG=(--enable-prefill-cudagraph)
  else
    CUDAGRAPH_FLAG=(--no-enable-prefill-cudagraph)
  fi

  STATIC_RUNNER_FLAG=()
  if [[ "${ENABLE_FULL_STATIC_RUNNER}" == "true" ]]; then
    STATIC_RUNNER_FLAG=(--enable-full-static-runner)
  else
    STATIC_RUNNER_FLAG=(--no-enable-full-static-runner)
  fi

  TORCH_COMPILE_FLAG=()
  if [[ "${ENABLE_CUDAGRAPH_TORCH_COMPILE}" == "true" ]]; then
    TORCH_COMPILE_FLAG=(--enable-cudagraph-torch-compile)
  else
    TORCH_COMPILE_FLAG=(--no-enable-cudagraph-torch-compile)
  fi

  set +e
  HF_ALLOW_CODE_EVAL=1 \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${PYTHON_BIN}" -m diffulex_bench.main \
    --log-file "${log_file}" \
    --log-level INFO \
    --config "${CONFIG_PATH}" \
    --model-path "${model_path}" \
    --model-name "${MODEL_NAME}" \
    --decoding-strategy "${DECODING_STRATEGY}" \
    --sampling-mode "${SAMPLING_MODE}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --data-parallel-size "${DATA_PARALLEL_SIZE}" \
    --dataset "${dataset}" \
    --temperature 0.0 \
    --max-tokens "${MAX_TOKENS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-reqs "${MAX_NUM_REQS}" \
    --block-size "${block_size}" \
    --buffer-size "${buffer_size}" \
    --add-block-threshold "${add_th}" \
    --semi-complete-threshold "${semi_th}" \
    --accept-threshold "${acc_th}" \
    --edit-threshold "${edit_th}" \
    --remask-threshold "${rem_th}" \
    --token-stability-threshold "${stab_th}" \
    --max-post-edit-steps "${MAX_POST_EDIT_STEPS}" \
    --attn-impl "${ATTN_IMPL}" \
    --moe-gemm-impl "${MOE_GEMM_IMPL}" \
    "${CUDAGRAPH_FLAG[@]}" \
    "${STATIC_RUNNER_FLAG[@]}" \
    "${TORCH_COMPILE_FLAG[@]}" \
    --prefill-cudagraph-max-len 0 \
    --output-dir "${exp_dir}" \
    --no-use-run-subdirectory \
    --max-nfe "${MAX_NFE}"
  rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    echo "[done] #${i}: ${dataset}/${model_short}/bs${block_size}_buf${buffer_size}"
    ((run_count++)) || true
  else
    echo "[FAIL] #${i}: exit code ${rc} — see ${log_file}"
    ((fail_count++)) || true
  fi

  echo ""
done

echo "================================================================"
echo "Batch finished: ${run_count} run, ${skip_count} skipped, ${fail_count} failed (of ${#EXPERIMENTS[@]} total)"
echo "Results root: ${OUTPUT_ROOT}"
echo "Logs:         ${LOG_DIR}"
echo "================================================================"
