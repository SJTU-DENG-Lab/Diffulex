#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
PROFILE_ROOT="${PROFILE_ROOT:-benchmark_results/profile_sglang_${STAMP}}"
DATASET="${DATASET:-gsm8k_diffulex_dmax_chat}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-10}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://127.0.0.1:29998}"
RUN_SGLANG_SERVER_PROFILE="${RUN_SGLANG_SERVER_PROFILE:-1}"
SGLANG_PROFILE_STEPS="${SGLANG_PROFILE_STEPS:-64}"

MODEL_PATH="${MODEL_PATH:-/root/data/ckpts/inclusionAI/LLaDA2.0-mini}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"

mkdir -p "${PROFILE_ROOT}" logs

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

if ! check_sglang_server; then
  echo "[sglang] no server is listening at ${SGLANG_BASE_URL}"
  echo "[sglang] start sglang.launch_server first, then rerun this script."
  exit 1
fi

SGLANG_PROFILER_PID=""
if [[ "${RUN_SGLANG_SERVER_PROFILE}" == "1" ]]; then
  echo "[sglang-server] starting server-side torch profiler for ${SGLANG_PROFILE_STEPS} forward steps"
  .venv/bin/python - "${SGLANG_BASE_URL}" "${PROFILE_ROOT}/sglang_server" "${SGLANG_PROFILE_STEPS}" <<'PY' &
import json
import sys
import time
from pathlib import Path

import requests

url, output_root, num_steps = sys.argv[1], sys.argv[2], sys.argv[3]
output_dir = Path(output_root).expanduser().resolve() / str(time.time())
output_dir.mkdir(parents=True, exist_ok=True)
try:
    server_info = requests.get(f"{url}/get_server_info", timeout=30)
    if server_info.ok:
        (output_dir / "server_args.json").write_text(json.dumps(server_info.json(), indent=2), encoding="utf-8")
except requests.RequestException:
    pass
payload = {
    "output_dir": str(output_dir),
    "num_steps": str(num_steps),
    "activities": ["CPU", "GPU"],
    "profile_by_stage": True,
    "merge_profiles": True,
    "profile_prefix": "sglang_server",
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

.venv/bin/python script/summarize_torch_profiles.py "${PROFILE_ROOT}" --top-n "${PROFILE_TOP_N:-60}"
echo "[done] sglang profile root: ${PROFILE_ROOT}"
