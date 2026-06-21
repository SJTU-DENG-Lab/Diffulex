#!/usr/bin/env bash
set -euo pipefail

# Experiment config runner.
#
# Usage:
#   ./script/run_batch_experiments.sh
#   DRY_RUN=1 ./script/run_batch_experiments.sh
#   CONFIG_FILES=llada2_mini.yml,sdar_8b_chat_b32.yml ./script/run_batch_experiments.sh
#   FILTER=multibd_math DATASET_LIMIT=10 ./script/run_batch_experiments.sh

export DIFFULEX_DMAX_SAMPLER_FAST="${DIFFULEX_DMAX_SAMPLER_FAST:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_DIR="${CONFIG_DIR:-diffulex_bench/configs/experiment}"
DEFAULTS_CONFIG="${DEFAULTS_CONFIG:-${CONFIG_DIR}/_defaults.yml}"
CONFIG_PATTERN="${CONFIG_PATTERN:-*.yml}"
CONFIG_FILES="${CONFIG_FILES:-}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

RUN_ID="${RUN_ID:-run_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-benchmark_results/experiment}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUT_BASE}/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-logs/experiment/${RUN_ID}}"
PLAN_TSV="${OUTPUT_ROOT}/plan.tsv"

FILTER="${FILTER:-}"
DATASET_LIMIT="${DATASET_LIMIT:-}"
MAX_NUM_REQS="${MAX_NUM_REQS:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_MISSING_MODELS="${SKIP_MISSING_MODELS:-0}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/resolve_experiment_configs.py" \
  --config-dir "${CONFIG_DIR}" \
  --defaults-config "${DEFAULTS_CONFIG}" \
  --config-pattern "${CONFIG_PATTERN}" \
  --config-files "${CONFIG_FILES}" \
  --output-root "${OUTPUT_ROOT}" \
  --log-dir "${LOG_DIR}" \
  --plan-tsv "${PLAN_TSV}" \
  --filter "${FILTER}" \
  --dataset-limit "${DATASET_LIMIT}" \
  --max-num-reqs "${MAX_NUM_REQS}"

total=$(( $(wc -l < "${PLAN_TSV}") - 1 ))

echo "================================================================"
echo "Experiment plan"
echo "================================================================"
echo "Config dir:          ${CONFIG_DIR}"
echo "Defaults:            ${DEFAULTS_CONFIG}"
echo "Config files:        ${CONFIG_FILES:-${CONFIG_PATTERN}}"
echo "Total experiments:   ${total}"
echo "Filter:              ${FILTER:-<none>}"
echo "Dataset limit:       ${DATASET_LIMIT:-<none>}"
echo "Max num reqs:        ${MAX_NUM_REQS:-config default}"
echo "CUDA_VISIBLE_DEVICES:${CUDA_VISIBLE_DEVICES:-<inherited>}"
echo "Output root:         ${OUTPUT_ROOT}"
echo "Log dir:             ${LOG_DIR}"
echo "Dry run:             ${DRY_RUN}"
echo "Force rerun:         ${FORCE_RERUN}"
echo "Skip missing models: ${SKIP_MISSING_MODELS}"
echo "----------------------------------------------------------------"
awk -F '\t' 'NR > 1 { printf "  [%02d] %-22s %-38s task=%-12s model=%-22s exists=%s\n", $1, $2, $3, $5, $6, $8 }' "${PLAN_TSV}"
echo "================================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] Resolved configs are under ${OUTPUT_ROOT}/resolved_configs"
  exit 0
fi

run_count=0
skip_count=0
fail_count=0

while IFS=$'\t' read -r index group name variant task model model_path model_exists config output_dir log_file; do
  stats_file="${output_dir}/diffulex_stats.json"

  if [[ "${model_exists}" != "1" ]]; then
    if [[ "${SKIP_MISSING_MODELS}" == "1" ]]; then
      echo "[skip] #${index} ${group}/${name}: model path does not exist: ${model_path}"
      ((skip_count++)) || true
      continue
    fi
    echo "[fail] #${index} ${group}/${name}: model path does not exist: ${model_path}"
    ((fail_count++)) || true
    continue
  fi

  if [[ "${FORCE_RERUN}" != "1" && -f "${stats_file}" ]]; then
    echo "[skip] #${index} ${group}/${name}: ${stats_file} exists"
    ((skip_count++)) || true
    continue
  fi

  echo "========================================================================"
  echo "[run] #${index}/${total} ${group}/${name}"
  echo "[run] variant    = ${variant}"
  echo "[run] task       = ${task}"
  echo "[run] model      = ${model}"
  echo "[run] model_path = ${model_path}"
  echo "[run] config     = ${config}"
  echo "[run] output_dir = ${output_dir}"
  echo "[run] log_file   = ${log_file}"
  echo "========================================================================"

  mkdir -p "$(dirname "${log_file}")" "${output_dir}"

  set +e
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    HF_ALLOW_CODE_EVAL=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" -m diffulex_bench.main \
      --log-file "${log_file}" \
      --log-level INFO \
      --config "${config}"
  else
    HF_ALLOW_CODE_EVAL=1 "${PYTHON_BIN}" -m diffulex_bench.main \
      --log-file "${log_file}" \
      --log-level INFO \
      --config "${config}"
  fi
  rc=$?
  set -e

  if [[ ${rc} -eq 0 ]]; then
    echo "[done] #${index} ${group}/${name}"
    ((run_count++)) || true
  else
    echo "[fail] #${index} ${group}/${name}: exit code ${rc}; see ${log_file}"
    ((fail_count++)) || true
  fi
  echo ""
done < <(tail -n +2 "${PLAN_TSV}")

echo "================================================================"
echo "Batch finished: ${run_count} run, ${skip_count} skipped, ${fail_count} failed (of ${total} total)"
echo "Results root: ${OUTPUT_ROOT}"
echo "Logs:         ${LOG_DIR}"
echo "Plan:         ${PLAN_TSV}"
echo "================================================================"

if [[ ${fail_count} -ne 0 ]]; then
  exit 1
fi
