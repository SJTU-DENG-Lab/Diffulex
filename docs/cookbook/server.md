# Server

Use the HTTP server entry point when you want an interactive service instead of a benchmark run.

Generic form:

```bash
python -m diffulex.server.launch \
  --model /path/to/model \
  --model-name <model_name> \
  --decoding-strategy <strategy> \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-batched-tokens 8192 \
  --max-num-reqs 128 \
  --gpu-memory-utilization 0.9
```

For interactive chat serving, keep `max_num_batched_tokens` larger than `max_model_len`. If you increase `max_model_len` for longer conversations, increase `max_num_batched_tokens` with it and reduce `max_num_reqs` if GPU memory becomes the bottleneck.

The server process accepts the same core engine arguments as the benchmark path, plus HTTP-specific flags:

- `--host`
- `--port`
- `--log-level`
- `--device-ids`
- `--zmq-command-addr`
- `--zmq-event-addr`

## Model recipes

Use the model pages for recommended server configurations:

- [LLaDA-8B-Instruct (D2F)](models/llada_instruct)
- [Dream-v0-Base-7B (D2F)](models/dream_base)
- [Fast_dLLM_v2_7B (MultiBD)](models/fast_dllm_v2)
- [SDAR-1.7B-Chat-b32 (MultiBD)](models/sdar_chat)
