# Server

Use the HTTP server when an application, UI, or integration test needs to send
requests to Diffulex over HTTP instead of calling the Python API in-process.

The server starts a FastAPI frontend and a synchronous backend worker that owns
the Diffulex engine. ZMQ addresses are generated automatically for local runs.

## Start a LLaDA2-Mini Server

```bash
export MODEL_PATH=/path/to/LLaDA2.0-mini

CUDA_VISIBLE_DEVICES=0 python -m diffulex.server \
  --model "$MODEL_PATH" \
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
  --gpu-memory-utilization 0.45 \
  --attn-impl triton_grouped \
  --host 127.0.0.1 \
  --port 8000
```

Use `--attn-impl triton_grouped` for normal serving and demos. Other attention
backends are compatibility/debug fallbacks and are not recommended for
performance reporting.

## Generate Endpoint

Non-streaming request:

```bash
curl -s http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Solve: 12 + 30.","temperature":0.0,"max_tokens":64,"max_nfe":256}' \
  | python -m json.tool
```

The response contains:

| Field | Meaning |
| --- | --- |
| `text` | Generated completion text. |
| `token_ids` | Generated token IDs. |
| `nfe` | Number of forward evaluations used by the request. |
| `finish_reason` | Stop reason when available. |
| `full_text` | Prompt plus generated text when available. |

Streaming request:

```bash
curl -N http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Solve: 12 + 30.","temperature":0.0,"max_tokens":64,"stream":true,"stream_mode":"denoise"}'
```

`stream_mode="denoise"` emits editable buffer snapshots.
`stream_mode="block_append"` emits stable appended text.

## Chat Endpoint

The server also exposes an OpenAI-style chat path:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Solve: 12 + 30."}],
    "temperature": 0.0,
    "max_tokens": 64,
    "stream": true,
    "stream_mode": "block_append"
  }'
```

## Demo Visualization

The repository includes a local Streamlit demo for visualizing server responses.
Start it after the HTTP server is ready:

```bash
streamlit run examples/streamlit_block_append_chat.py -- --base-url http://127.0.0.1:8000
```

The demo talks to the server API and is intended for local validation and video
capture, not production serving or throughput measurement.

## Important Server Flags

| Flag | Notes |
| --- | --- |
| `--model` | Required local checkpoint path. |
| `--model-name` | Registered Diffulex model name such as `llada2_mini`, `sdar`, or `diffusion_gemma`. |
| `--decoding-strategy` | Use a strategy compatible with the model. |
| `--sampling-mode` | Usually `naive`; use `edit` only for compatible LLaDA2 edit/DMax paths. |
| `--tensor-parallel-size`, `--data-parallel-size` | Must fit the visible CUDA devices. |
| `--device-ids` | Logical CUDA IDs after `CUDA_VISIBLE_DEVICES` is applied. |
| `--max-model-len`, `--max-num-batched-tokens`, `--max-num-reqs` | Capacity controls. Start small. |
| `--block-size`, `--buffer-size`, `--page-size` | Strategy/model layout controls. DiffusionGemma uses `256/1/256`. |
| `--disable-full-static-runner`, `--disable-torch-compile`, `--enforce-eager` | Debugging toggles for optimized paths. |
| `--use-lora`, `--lora-path`, `--pre-merge-lora` | LoRA loading and merge controls. |

Run `python -m diffulex.server --help` for the complete current option
list.
