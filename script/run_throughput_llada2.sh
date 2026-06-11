#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# LLaDA2 MultiBD throughput benchmark
# Grid: batch × bufsize × blocksize × {math, code}
# Style: matches run_throughput_dmax.sh pattern
# ============================================================

export DIFFULEX_DMAX_SAMPLER_FAST=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ============================================================
# Model paths (math / code — separate models)
# ============================================================

# LLADA2_MATH_MODEL="${LLADA2_MATH_MODEL:-/root/workspace/jyj/workspaces/MultiBD/multi_bd_train/multibd_ckpt/chosen_ckpt/llada2_mini/MBD-Math-LLaDA2-mini-16B}"
# LLADA2_CODE_MODEL="${LLADA2_CODE_MODEL:-/root/workspace/jyj/workspaces/MultiBD/multi_bd_train/multibd_ckpt/chosen_ckpt/llada2_mini/MBD-Coder-LLaDA2-mini-16B}"

LLADA2_MATH_MODEL="${LLADA2_MATH_MODEL:-/root/data/ckpts/inclusionAI/LLaDA2.0-mini}"
LLADA2_CODE_MODEL="${LLADA2_CODE_MODEL:-/root/data/ckpts/inclusionAI/LLaDA2.0-mini}"

# ============================================================
# Model name
# ============================================================
MODEL_NAME="${MODEL_NAME:-llada2_mini}"

# ============================================================
# Decoding strategy & sampling mode
# ============================================================
DECODING_STRATEGY="${DECODING_STRATEGY:-multi_bd}"
SAMPLING_MODE="${SAMPLING_MODE:-naive}"

# ============================================================
# Grid
# ============================================================
BLOCK_SIZES=(32)
BUFFER_SIZES=(1)
BATCH_SIZES=(1)

# Thresholds — math
MATH_ADD_BLOCK_THRESHOLDS=(0.1)
MATH_SEMI_COMPLETE_THRESHOLDS=(0.9)
MATH_ACCEPT_THRESHOLDS=(0.95)
MATH_REMASK_THRESHOLDS=(0.4)
MATH_TOKEN_STABILITY_THRESHOLDS=(0.0)

# Thresholds — code
CODE_ADD_BLOCK_THRESHOLDS=(0.9)
CODE_SEMI_COMPLETE_THRESHOLDS=(0.9)
CODE_ACCEPT_THRESHOLDS=(0.95)
CODE_REMASK_THRESHOLDS=(0.4)
CODE_TOKEN_STABILITY_THRESHOLDS=(0.5)

# ============================================================
# Fixed parameters
# ============================================================
CONFIG_PATH="${CONFIG_PATH:-diffulex_bench/configs/llada2_mini_gsm8k.yml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"

MAX_TOKENS_MATH="${MAX_TOKENS_MATH:-4096}"
MAX_TOKENS_CODE="${MAX_TOKENS_CODE:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NFE="${MAX_NFE:-1024}"

ATTN_IMPL="${ATTN_IMPL:-triton}"
MOE_GEMM_IMPL="${MOE_GEMM_IMPL:-vllm_modular}"
ENABLE_PREFILL_CUDAGRAPH="${ENABLE_PREFILL_CUDAGRAPH:-true}"
ENABLE_FULL_STATIC_RUNNER="${ENABLE_FULL_STATIC_RUNNER:-true}"
ENABLE_CUDAGRAPH_TORCH_COMPILE="${ENABLE_CUDAGRAPH_TORCH_COMPILE:-false}"

# --- Output ---
OUTPUT_BASE="${OUTPUT_BASE:-benchmark_results/throughput_llada2}"
RUN_ID="${RUN_ID:-run_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_BASE}/${RUN_ID}"
LOG_DIR="${LOG_DIR:-logs/throughput_llada2/${RUN_ID}}"

# --- Misc ---
FORCE_RERUN="${FORCE_RERUN:-0}"
DRY_RUN="${DRY_RUN:-0}"

# ============================================================
# Build experiment list
# ============================================================
LLADA2_MATH_DATASETS=(
  gsm8k_llada2
  math500_llada2
)
LLADA2_CODE_DATASETS=(
  humaneval_plus_llada2
  mbpp_plus_llada2
)

EXPERIMENTS=()

for mp in "${LLADA2_MATH_MODEL}"; do
  for ds in "${LLADA2_MATH_DATASETS[@]}"; do
    for bat in "${BATCH_SIZES[@]}"; do
      for bs in "${BLOCK_SIZES[@]}"; do
        for buf in "${BUFFER_SIZES[@]}"; do
          for ath in "${MATH_ADD_BLOCK_THRESHOLDS[@]}"; do
            for sth in "${MATH_SEMI_COMPLETE_THRESHOLDS[@]}"; do
              for acth in "${MATH_ACCEPT_THRESHOLDS[@]}"; do
                for rth in "${MATH_REMASK_THRESHOLDS[@]}"; do
                  for stth in "${MATH_TOKEN_STABILITY_THRESHOLDS[@]}"; do
                    EXPERIMENTS+=("${mp}|${ds}|${bat}|${bs}|${buf}|${ath}|${sth}|${acth}|${rth}|${stth}|${MAX_TOKENS_MATH}")
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

for mp in "${LLADA2_CODE_MODEL}"; do
  for ds in "${LLADA2_CODE_DATASETS[@]}"; do
    for bat in "${BATCH_SIZES[@]}"; do
      for bs in "${BLOCK_SIZES[@]}"; do
        for buf in "${BUFFER_SIZES[@]}"; do
          for ath in "${CODE_ADD_BLOCK_THRESHOLDS[@]}"; do
            for sth in "${CODE_SEMI_COMPLETE_THRESHOLDS[@]}"; do
              for acth in "${CODE_ACCEPT_THRESHOLDS[@]}"; do
                for rth in "${CODE_REMASK_THRESHOLDS[@]}"; do
                  for stth in "${CODE_TOKEN_STABILITY_THRESHOLDS[@]}"; do
                    EXPERIMENTS+=("${mp}|${ds}|${bat}|${bs}|${buf}|${ath}|${sth}|${acth}|${rth}|${stth}|${MAX_TOKENS_CODE}")
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

# ============================================================
# Print plan
# ============================================================
echo "================================================================"
echo "LLaDA2 MultiBD Throughput Benchmark"
echo "================================================================"
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo "Model name:        ${MODEL_NAME}"
echo "Decoding strategy: ${DECODING_STRATEGY}"
echo "Sampling mode:     ${SAMPLING_MODE}"
echo "Config:            ${CONFIG_PATH}"
echo "GPUs:              ${CUDA_VISIBLE_DEVICES}"
echo "TP size:           ${TENSOR_PARALLEL_SIZE}"
echo "Output root:       ${OUTPUT_ROOT}"
echo "Log dir:           ${LOG_DIR}"
echo "----------------------------------------------------------------"
for i in "${!EXPERIMENTS[@]}"; do
  IFS='|' read -r mp ds bat bs buf ath sth acth rth stth mtok <<< "${EXPERIMENTS[$i]}"
  model_short="$(basename "${mp}")"
  printf "  [%2d] model=%-28s ds=%-25s bat=%s bs=%s buf=%s max_tok=%s\n" "$i" "${model_short}" "${ds}" "${bat}" "${bs}" "${buf}" "${mtok}"
done
echo "================================================================"
echo ""

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] Done. Set DRY_RUN=0 to execute."
  exit 0
fi

# ============================================================
# Run
# ============================================================
mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

run_count=0; skip_count=0; fail_count=0

for i in "${!EXPERIMENTS[@]}"; do
  IFS='|' read -r model_path dataset batch_size block_size buffer_size add_th semi_th acc_th rem_th stab_th max_tok <<< "${EXPERIMENTS[$i]}"
  model_short="$(basename "${model_path}")"

  exp_dir="${OUTPUT_ROOT}/${dataset}/${model_short}/bat${batch_size}_bs${block_size}_buf${buffer_size}"
  log_file="${LOG_DIR}/${dataset}__${model_short}__bat${batch_size}_bs${block_size}_buf${buffer_size}.log"
  stats_file="${exp_dir}/diffulex_stats.json"

  if [[ "${FORCE_RERUN}" != "1" && -f "${stats_file}" ]]; then
    echo "[skip] #${i}: ${stats_file} exists"
    ((skip_count++)) || true
    continue
  fi

  echo "========================================================================"
  echo "[run]  #${i}/${#EXPERIMENTS[@]}"
  echo "[run]  model      = ${model_short}"
  echo "[run]  dataset    = ${dataset}"
  echo "[run]  batch=${batch_size}  block=${block_size}  buf=${buffer_size}  max_tok=${max_tok}"
  echo "========================================================================"

  mkdir -p "${exp_dir}" "$(dirname "${log_file}")"

  max_num_batched_tokens=$((batch_size * MAX_MODEL_LEN))
  [[ ${max_num_batched_tokens} -lt 2048 ]] && max_num_batched_tokens=2048

  CUDAGRAPH_FLAG=(--enable-prefill-cudagraph)
  STATIC_RUNNER_FLAG=(--enable-full-static-runner)
  TORCH_COMPILE_FLAG=(--no-enable-cudagraph-torch-compile)

  set +e
  HF_ALLOW_CODE_EVAL=1 \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${PYTHON_BIN}" -m diffulex_bench.main \
    --log-file "${log_file}" --log-level INFO \
    --config "${CONFIG_PATH}" \
    --model-path "${model_path}" \
    --model-name "${MODEL_NAME}" \
    --decoding-strategy "${DECODING_STRATEGY}" \
    --sampling-mode "${SAMPLING_MODE}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --data-parallel-size "${DATA_PARALLEL_SIZE}" \
    --dataset "${dataset}" \
    --temperature 0.0 \
    --max-tokens "${max_tok}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${max_num_batched_tokens}" \
    --max-num-reqs "${batch_size}" \
    --block-size "${block_size}" \
    --buffer-size "${buffer_size}" \
    --add-block-threshold "${add_th}" \
    --semi-complete-threshold "${semi_th}" \
    --accept-threshold "${acc_th}" \
    --remask-threshold "${rem_th}" \
    --token-stability-threshold "${stab_th}" \
    --attn-impl "${ATTN_IMPL}" \
    --moe-gemm-impl "${MOE_GEMM_IMPL}" \
    "${CUDAGRAPH_FLAG[@]}" "${STATIC_RUNNER_FLAG[@]}" "${TORCH_COMPILE_FLAG[@]}" \
    --prefill-cudagraph-max-len 0 \
    --output-dir "${exp_dir}" \
    --no-use-run-subdirectory \
    --max-nfe "${MAX_NFE}"
  rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    echo "[done] #${i}"
    ((run_count++)) || true
  else
    echo "[FAIL] #${i}: exit code ${rc} — see ${log_file}"
    ((fail_count++)) || true
  fi
  echo ""
done

echo "================================================================"
echo "LLaDA2 MultiBD finished: ${run_count} run, ${skip_count} skipped, ${fail_count} failed (of ${#EXPERIMENTS[@]} total)"
echo "Results root: ${OUTPUT_ROOT}"
echo "================================================================"
