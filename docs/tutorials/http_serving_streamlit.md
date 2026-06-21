# HTTP Serving and Streamlit Client Usage

Diffulex can run as an HTTP service and can be paired with the sample Streamlit
client for local interactive validation. This workflow is useful when you want
to test request handling, server lifecycle, or a chat-style frontend without
writing a custom client.

## Start the Server

Start with conservative engine limits:

```bash
python -m diffulex.server \
  --model /path/to/LLaDA2.0-mini \
  --model-name llada2_mini \
  --decoding-strategy multi_bd \
  --sampling-mode naive \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-reqs 1 \
  --block-size 32 \
  --buffer-size 1 \
  --page-size 32 \
  --host 0.0.0.0 \
  --port 8000
```

The launch command starts a backend worker and an HTTP frontend. The backend
owns the Diffulex engine. The frontend communicates with it over ZMQ addresses
that are generated automatically unless provided with explicit flags.

## Choose Runtime Settings

During debugging, prefer predictable behavior:

| Flag | Suggested setting | Why |
| --- | --- | --- |
| `--enforce-eager` | Add it while validating a new model or strategy. | Disables CUDA Graph capture so failures are easier to localize. |
| `--max-num-reqs` | Start with a small value. | Keeps request state and memory pressure low during startup checks. |
| `--tensor-parallel-size`, `--data-parallel-size` | Start with `1` for both. | Separates serving and model compatibility issues from distributed topology issues. |
| `--device-ids` | Set it when only selected GPUs should be visible to the server. | Makes device selection explicit for local validation. |

For throughput checks, remove `--enforce-eager` and increase request and token
limits gradually.

## Start the Streamlit Client

After the server reports that it is ready, start the sample frontend:

```bash
streamlit run examples/streamlit_block_append_chat.py -- --base-url http://localhost:8000
```

The Streamlit app talks to the server API. It is intended for local validation,
not as a production UI.

## Basic Health Checks

If the client cannot connect:

1. Confirm the server process is still running.
2. Confirm the client `--base-url` matches the server host and port.
3. Check the server log for backend startup failures.
4. Reduce model and batch limits if the backend exits during load.

If generation starts but stalls, check GPU memory, request limits, and decoding
strategy settings before changing the frontend.

## When to Use This Workflow

Use HTTP serving for integration tests, demos, and interactive validation. Use
offline `Diffulex.generate` when you only need local batched inference. Use
`diffulex_bench` when you need dataset metrics and reproducible evaluation
artifacts.
