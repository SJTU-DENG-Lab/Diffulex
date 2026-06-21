---
orphan: true
---

# Streamlit

Diffulex includes a sample Streamlit frontend for chatting with the HTTP server.
Use this page as a quick command reference for that local validation path. The
longer serving walkthrough lives under Tutorials.

## Start the Server First

The Streamlit app expects an already-running Diffulex HTTP server. Start the
server with the model, strategy, and GPU settings you want to test:

```bash
python -m diffulex.server.launch \
  --model /path/to/model \
  --model-name llada \
  --decoding-strategy d2f \
  --host 0.0.0.0 \
  --port 8000
```

Keep request limits small while validating a new model or strategy.

## Start the Client

Run the sample client from the repository root:

```bash
streamlit run examples/streamlit_block_append_chat.py -- --base-url http://localhost:8000
```

The `--base-url` value must match the server host and port. If the app cannot
connect, verify the server process is still running and check its logs for
backend startup errors.

## Intended Use

The Streamlit client is meant for local interactive validation. It is useful for
checking that the HTTP server accepts requests, returns streamed or final
responses, and behaves sensibly in a chat-like workflow. It is not a benchmark
tool and should not be used to measure throughput.
