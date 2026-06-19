#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
PROFILE_ROOT="${PROFILE_ROOT:-benchmark_results/profile_multi_bd_vs_sglang_${STAMP}}"
DATASET="${DATASET:-humaneval_dmax_reference_chat}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-10}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DIFFULEX_TP_SIZE="${DIFFULEX_TP_SIZE:-1}"
SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://127.0.0.1:29998}"
RUN_DIFFULEX="${RUN_DIFFULEX:-1}"
RUN_SGLANG_CLIENT="${RUN_SGLANG_CLIENT:-1}"
RUN_SGLANG_SERVER_PROFILE="${RUN_SGLANG_SERVER_PROFILE:-1}"
SGLANG_PROFILE_STEPS="${SGLANG_PROFILE_STEPS:-5}"
SGLANG_PROFILE_BY_STAGE="${SGLANG_PROFILE_BY_STAGE:-1}"
SGLANG_PROFILE_MERGE="${SGLANG_PROFILE_MERGE:-0}"
SGLANG_PROFILE_CPU="${SGLANG_PROFILE_CPU:-1}"
SGLANG_PROFILE_GPU="${SGLANG_PROFILE_GPU:-1}"
SGLANG_PROFILE_PREFIX="${SGLANG_PROFILE_PREFIX:-sglang_server}"
SGLANG_REQUIRE_SERVER="${SGLANG_REQUIRE_SERVER:-0}"

DEFAULT_MODEL="/root/data/ckpts/inclusionAI/LLaDA2.0-mini"
ALT_MODEL="/root/data/ckpts/inclusionAI/LLaDA2.0-mini"
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

check_sglang_server() {
  .venv/bin/python - "$SGLANG_BASE_URL" <<'PY'
import sys
import requests

url = sys.argv[1].rstrip("/")
for endpoint in ("/get_server_info", "/health"):
    try:
        response = requests.get(url + endpoint, timeout=5)
        if response.status_code < 500:
            raise SystemExit(0)
    except requests.RequestException:
        pass
raise SystemExit(1)
PY
}

if [[ "${RUN_DIFFULEX}" == "1" ]]; then
  echo "[diffulex] profiling ${SAMPLE_LIMIT} samples, dataset=${DATASET}, model=${MODEL_PATH}"
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
    --sampling-mode naive \
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
fi

if [[ "${RUN_SGLANG_CLIENT}" == "1" ]]; then
  echo "[sglang-client] profiling client-side request path against ${SGLANG_BASE_URL}"
  if ! check_sglang_server; then
    echo "[sglang-client] no sglang server is listening at ${SGLANG_BASE_URL}"
    echo "[sglang-client] start it first, or set RUN_SGLANG_CLIENT=0 to profile DiffuLEx only."
    if [[ "${SGLANG_REQUIRE_SERVER}" == "1" ]]; then
      exit 1
    fi
    RUN_SGLANG_CLIENT=0
  fi
fi

if [[ "${RUN_SGLANG_CLIENT}" == "1" ]]; then
  SGLANG_PROFILER_PID=""
  if [[ "${RUN_SGLANG_SERVER_PROFILE}" == "1" ]]; then
    echo "[sglang-server] starting server-side torch profiler for ${SGLANG_PROFILE_STEPS} forward steps"
    .venv/bin/python - "${SGLANG_BASE_URL}" "${PROFILE_ROOT}/sglang_server" "${SGLANG_PROFILE_STEPS}" "${SGLANG_PROFILE_BY_STAGE}" "${SGLANG_PROFILE_MERGE}" "${SGLANG_PROFILE_CPU}" "${SGLANG_PROFILE_GPU}" "${SGLANG_PROFILE_PREFIX}" <<'PY' &
import json
import sys
import time
from pathlib import Path

import requests

url, output_root, num_steps, by_stage, merge, cpu, gpu, prefix = sys.argv[1:9]
output_dir = Path(output_root).expanduser().resolve() / str(time.time())
output_dir.mkdir(parents=True, exist_ok=True)
try:
    server_info = requests.get(f"{url}/get_server_info", timeout=30)
    if server_info.ok:
        (output_dir / "server_args.json").write_text(json.dumps(server_info.json(), indent=2), encoding="utf-8")
except requests.RequestException:
    pass
activities = []
if cpu == "1":
    activities.append("CPU")
if gpu == "1":
    activities.append("GPU")
payload = {
    "output_dir": str(output_dir),
    "num_steps": str(num_steps),
    "activities": activities,
    "profile_by_stage": by_stage == "1",
    "merge_profiles": merge == "1",
    "profile_prefix": prefix,
}
print(f"Dump sglang server profiling traces to {output_dir}", flush=True)
response = requests.post(f"{url}/start_profile", json=payload, timeout=None)
response.raise_for_status()
PY
    SGLANG_PROFILER_PID="$!"
    sleep "${SGLANG_PROFILE_STARTUP_SLEEP:-3}"
  fi

  TMP_CFG="$(mktemp "${PROFILE_ROOT}/sglang_profile_XXXXXX.yml")"
  cat > "${TMP_CFG}" <<EOF
endpoint:
  engine_name: "sglang"
  base_url: "${SGLANG_BASE_URL}"
  model: "${MODEL_PATH}"
  api_key: "EMPTY"
  tokenizer_path: "${TOKENIZER_PATH}"
  trust_remote_code: true
  apply_chat_template: true
  chat_completions: false
  timeout: 600.0
  verify: true

eval:
  dataset_name: "${DATASET}"
  dataset_limit: ${SAMPLE_LIMIT}
  include_path: null
  dataset_data_files: null
  temperature: 0.0
  max_tokens: ${MAX_TOKENS:-4096}
  ignore_eos: false
  add_bos_token: false
  output_dir: "${PROFILE_ROOT}/sglang_client_results"
  use_run_subdirectory: false
  save_results: true
EOF

  DIFFULEX_PROFILE_DIR="${PROFILE_ROOT}/sglang_client" \
  DIFFULEX_PROFILE_RUN_ID="sglang_client" \
  DIFFULEX_PROFILE_ACTIVE_STEPS="${SAMPLE_LIMIT}" \
  DIFFULEX_PROFILE_RECORD_SHAPES="${DIFFULEX_PROFILE_RECORD_SHAPES:-0}" \
  DIFFULEX_PROFILE_MEMORY="${DIFFULEX_PROFILE_MEMORY:-0}" \
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  HF_ALLOW_CODE_EVAL=1 \
  .venv/bin/python -m examples.engine_lm_eval.main --config "${TMP_CFG}"

  if [[ -n "${SGLANG_PROFILER_PID}" ]]; then
    echo "[sglang-server] waiting for server-side profile flush"
    wait "${SGLANG_PROFILER_PID}"
  fi
fi

.venv/bin/python script/summarize_torch_profiles.py "${PROFILE_ROOT}" --top-n "${PROFILE_TOP_N:-60}"

echo "[done] profile root: ${PROFILE_ROOT}"
echo "[note] sglang_client trace only covers the OpenAI-compatible client wait/stream path; CUDA kernels require profiling inside the sglang server process."
