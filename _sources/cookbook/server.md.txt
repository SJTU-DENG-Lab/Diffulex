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
  --max-model-len 2048 \
  --max-num-batched-tokens 4096 \
  --max-num-reqs 128 \
  --gpu-memory-utilization 0.9
```

The server process accepts the same core engine arguments as the benchmark path, plus HTTP-specific flags:

- `--host`
- `--port`
- `--log-level`
- `--device-ids`
- `--zmq-command-addr`
- `--zmq-event-addr`

## Supported models

### Fast-dLLM-v2

```bash
python -m diffulex.server.launch \
  --model /YOUR-CKPT-PATH/Efficient-Large-Model/Fast_dLLM_v2_7B \
  --model-name fast_dllm_v2 \
  --decoding-strategy multi_bd \
  --sampling-mode naive \
  --tensor-parallel-size 2 \
  --data-parallel-size 1 \
  --max-model-len 1024 \
  --max-num-batched-tokens 1024 \
  --max-num-reqs 24 \
  --gpu-memory-utilization 0.4 \
  --block-size 32 \
  --buffer-size 1 \
  --accept-threshold 0.95 \
  --semi-complete-threshold 0.9 \
  --add-block-threshold 0.1 \
  --enforce-eager
```

### D2F-LLaDA

```bash
python -m diffulex.server.launch \
  --model /YOUR-CKPT-PATH/GSAI-ML/LLaDA-8B-Instruct \
  --model-name llada \
  --decoding-strategy d2f \
  --tensor-parallel-size 2 \
  --data-parallel-size 1 \
  --use-lora \
  --lora-path /YOUR-CKPT-PATH/SJTU-DENG-Lab/D2F_LLaDA_Instruct_8B_Lora \
  --pre-merge-lora \
  --max-model-len 2048 \
  --max-num-batched-tokens 2048 \
  --max-num-reqs 32 \
  --accept-threshold 0.95 \
  --semi-complete-threshold 0.9 \
  --add-block-threshold 0.1 \
  --enforce-eager
```

### SDAR

```bash
python -m diffulex.server.launch \
  --model /YOUR-CKPT-PATH/JetLM/SDAR-1.7B-Chat-b32 \
  --host 0.0.0.0 \
  --port 8000 \
  --model-name sdar \
  --decoding-strategy multi_bd \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --device-ids 1 \
  --block-size 32 \
  --buffer-size 4 \
  --page-size 32 \
  --max-num-batched-tokens 4096 \
  --max-num-reqs 128 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.5 \
  --kv-cache-layout unified \
  --add-block-threshold 0.1 \
  --semi-complete-threshold 0.9 \
  --accept-threshold 0.95
```
