#!/usr/bin/env bash
set -euo pipefail

# LLaDA2 batch benchmark with the current vLLM-aligned kernel stack.
#
# This script intentionally does not override decoding thresholds, decoding
# strategy, sampling mode, block size, or buffer size unless you pass the
# corresponding environment variables. Those remain owned by CONFIG_PATH.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-diffulex_bench/configs/llada2_mini_gsm8k.yml}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

DATASETS=(${DATASETS:-gsm8k_llada2})
MODEL_PATHS=(${MODEL_PATHS:-/data1/ckpts/inclusionAI/LLaDA2.0-mini})

MODEL_NAME="${MODEL_NAME:-llada2_mini}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
EXPERT_PARALLEL_SIZE="${EXPERT_PARALLEL_SIZE:-1}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
# Engine validation requires max_num_batched_tokens >= max_model_len. Keep the
# token cap aligned with the config and control 24GB memory pressure primarily
# through max_num_reqs.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_REQS="${MAX_NUM_REQS:-32}"

ATTN_IMPL="${ATTN_IMPL:-triton_grouped}"
MOE_GEMM_IMPL="${MOE_GEMM_IMPL:-vllm_modular}"
MOE_DISPATCHER_BACKEND="${MOE_DISPATCHER_BACKEND:-standard}"
KV_CACHE_LAYOUT="${KV_CACHE_LAYOUT:-unified}"
ENABLE_VLLM_LAYERS="${ENABLE_VLLM_LAYERS:-true}"
ENABLE_PREFILL_CUDAGRAPH="${ENABLE_PREFILL_CUDAGRAPH:-true}"
ENABLE_FULL_STATIC_RUNNER="${ENABLE_FULL_STATIC_RUNNER:-true}"
ENABLE_CUDAGRAPH_TORCH_COMPILE="${ENABLE_CUDAGRAPH_TORCH_COMPILE:-false}"
PREFILL_CUDAGRAPH_MAX_LEN="${PREFILL_CUDAGRAPH_MAX_LEN:-0}"
DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-nccl}"
DISTRIBUTED_TIMEOUT_SECONDS="${DISTRIBUTED_TIMEOUT_SECONDS:-3600}"
MASTER_PORT="${MASTER_PORT:-$("${PYTHON_BIN}" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')}"

# Optional eval overrides. Leave unset to use CONFIG_PATH.
MAX_TOKENS="${MAX_TOKENS:-}"
MAX_NFE="${MAX_NFE:-}"
DATASET_LIMIT="${DATASET_LIMIT:-}"

OUTPUT_BASE="${OUTPUT_BASE:-benchmark_results/llada2_modular}"
RUN_ID="${RUN_ID:-run_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_BASE}/${RUN_ID}"
LOG_DIR="${LOG_DIR:-logs/llada2_modular/${RUN_ID}}"

FORCE_RERUN="${FORCE_RERUN:-0}"
DRY_RUN="${DRY_RUN:-0}"
CARTESIAN_MODE="${CARTESIAN_MODE:-cross}"

# Keep the CUDA/NCCL runtime resolved from the venv. The system CUDA path can
# otherwise win for some libraries and mix with torch's cu13 wheels.
VENV_SITE="$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="${VENV_SITE}/nvidia/cu13/lib:${VENV_SITE}/nvidia/nccl/lib:${VENV_SITE}/nvidia/nvshmem/lib:${VENV_SITE}/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

# NCCL 2.28/cu13 segfaults on this RTX 3090 PCIe box with the default cuMem/NVLS
# path. These settings are validated by a 4-rank torch.distributed all_reduce.
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

EXPERIMENTS=()
if [[ "${CARTESIAN_MODE}" == "cross" ]]; then
  for model_path in "${MODEL_PATHS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      EXPERIMENTS+=("${model_path}|${dataset}")
    done
  done
else
  num=${#MODEL_PATHS[@]}
  for i in $(seq 0 $((num - 1))); do
    EXPERIMENTS+=("${MODEL_PATHS[$i]}|${DATASETS[$i]:-${DATASETS[0]}}")
  done
fi

bool_flag() {
  local name="$1" value="$2" positive="$3" negative="$4"
  if [[ "${value}" == "true" ]]; then
    printf '%s\n' "${positive}"
  elif [[ "${value}" == "false" ]]; then
    printf '%s\n' "${negative}"
  else
    echo "Invalid boolean for ${name}: ${value}" >&2
    exit 2
  fi
}

echo "================================================================"
echo "LLaDA2 modular benchmark plan"
echo "================================================================"
echo "Total experiments:       ${#EXPERIMENTS[@]}"
echo "Config:                  ${CONFIG_PATH}"
echo "Model name:              ${MODEL_NAME}"
echo "GPUs:                    ${CUDA_VISIBLE_DEVICES}"
echo "TP/DP/EP:                ${TENSOR_PARALLEL_SIZE}/${DATA_PARALLEL_SIZE}/${EXPERT_PARALLEL_SIZE}"
echo "Kernel stack:            attn=${ATTN_IMPL}, moe=${MOE_GEMM_IMPL}, vllm_layers=${ENABLE_VLLM_LAYERS}"
echo "CUDA graph:              prefill=${ENABLE_PREFILL_CUDAGRAPH}, full_static=${ENABLE_FULL_STATIC_RUNNER}, compile_capture=${ENABLE_CUDAGRAPH_TORCH_COMPILE}"
echo "Distributed backend:     ${DISTRIBUTED_BACKEND} (master_port=${MASTER_PORT})"
echo "Output root:             ${OUTPUT_ROOT}"
echo "Log dir:                 ${LOG_DIR}"
echo "Dry run:                 ${DRY_RUN}"
echo "Force rerun:             ${FORCE_RERUN}"
echo "----------------------------------------------------------------"
for i in "${!EXPERIMENTS[@]}"; do
  IFS='|' read -r mp ds <<< "${EXPERIMENTS[$i]}"
  printf "  [%2d] model=%-32s dataset=%s\n" "$i" "$(basename "${mp}")" "${ds}"
done
echo "================================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

run_count=0
skip_count=0
fail_count=0

for i in "${!EXPERIMENTS[@]}"; do
  IFS='|' read -r model_path dataset <<< "${EXPERIMENTS[$i]}"
  model_short="$(basename "${model_path}")"
  exp_dir="${OUTPUT_ROOT}/${dataset}/${model_short}"
  log_file="${LOG_DIR}/${dataset}__${model_short}.log"
  stats_file="${exp_dir}/diffulex_stats.json"

  if [[ "${FORCE_RERUN}" != "1" && -f "${stats_file}" ]]; then
    echo "[skip] #${i}: ${stats_file} exists"
    ((skip_count++)) || true
    continue
  fi

  mkdir -p "${exp_dir}" "$(dirname "${log_file}")"

  VLLM_LAYERS_FLAG="$(bool_flag ENABLE_VLLM_LAYERS "${ENABLE_VLLM_LAYERS}" "--enable-vllm-layers" "--no-enable-vllm-layers")"
  PREFILL_FLAG="$(bool_flag ENABLE_PREFILL_CUDAGRAPH "${ENABLE_PREFILL_CUDAGRAPH}" "--enable-prefill-cudagraph" "--no-enable-prefill-cudagraph")"
  STATIC_FLAG="$(bool_flag ENABLE_FULL_STATIC_RUNNER "${ENABLE_FULL_STATIC_RUNNER}" "--enable-full-static-runner" "--no-enable-full-static-runner")"
  COMPILE_CAPTURE_FLAG="$(bool_flag ENABLE_CUDAGRAPH_TORCH_COMPILE "${ENABLE_CUDAGRAPH_TORCH_COMPILE}" "--enable-cudagraph-torch-compile" "--no-enable-cudagraph-torch-compile")"

  optional_args=()
  if [[ -n "${MAX_TOKENS}" ]]; then optional_args+=(--max-tokens "${MAX_TOKENS}"); fi
  if [[ -n "${MAX_NFE}" ]]; then optional_args+=(--max-nfe "${MAX_NFE}"); fi
  if [[ -n "${DATASET_LIMIT}" ]]; then optional_args+=(--dataset-limit "${DATASET_LIMIT}"); fi

  echo "========================================================================"
  echo "[run] #${i}/${#EXPERIMENTS[@]} dataset=${dataset} model=${model_short}"
  echo "[run] output_dir=${exp_dir}"
  echo "[run] log_file=${log_file}"
  echo "========================================================================"

  set +e
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${PYTHON_BIN}" -m diffulex_bench.main \
    --log-file "${log_file}" \
    --log-level INFO \
    --config "${CONFIG_PATH}" \
    --model-path "${model_path}" \
    --model-name "${MODEL_NAME}" \
    --dataset "${dataset}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --data-parallel-size "${DATA_PARALLEL_SIZE}" \
    --expert-parallel-size "${EXPERT_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-reqs "${MAX_NUM_REQS}" \
    --kv-cache-layout "${KV_CACHE_LAYOUT}" \
    --attn-impl "${ATTN_IMPL}" \
    --moe-dispatcher-backend "${MOE_DISPATCHER_BACKEND}" \
    --moe-gemm-impl "${MOE_GEMM_IMPL}" \
    "${VLLM_LAYERS_FLAG}" \
    "${PREFILL_FLAG}" \
    "${STATIC_FLAG}" \
    "${COMPILE_CAPTURE_FLAG}" \
    --prefill-cudagraph-max-len "${PREFILL_CUDAGRAPH_MAX_LEN}" \
    --engine-arg "distributed_backend=${DISTRIBUTED_BACKEND}" \
    --engine-arg "distributed_timeout_seconds=${DISTRIBUTED_TIMEOUT_SECONDS}" \
    --engine-arg "master_port=${MASTER_PORT}" \
    --output-dir "${exp_dir}" \
    --no-use-run-subdirectory \
    "${optional_args[@]}"
  rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    echo "[done] #${i}: ${dataset}/${model_short}"
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
