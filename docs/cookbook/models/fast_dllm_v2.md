# Fast_dLLM_v2_7B (MultiBD)

## 1. Model Introduction

Fast_dLLM_v2_7B uses `multi_bd` decoding and is a throughput-oriented baseline for Diffulex.

## 2. Diffulex Installation

```bash
git clone https://github.com/SJTU-DENG-Lab/Diffulex.git
cd Diffulex
uv pip install -e .
```

## 3. Model Deployment

### 3.1 Basic Configuration

```bash
python -m diffulex.server.launch \
  --model /YOUR-CKPT-PATH/Efficient-Large-Model/Fast_dLLM_v2_7B \
  --model-name fast_dllm_v2 \
  --decoding-strategy multi_bd \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 1024 \
  --max-num-batched-tokens 1024 \
  --max-num-reqs 24 \
  --block-size 32 \
  --buffer-size 1
```

### 3.2 Configuration Tips

There is a trade-off between balanced serving and maximum throughput.

For balanced serving with strong accuracy, use moderate concurrency:

```bash
python -m diffulex.server.launch \
  --model /YOUR-CKPT-PATH/Efficient-Large-Model/Fast_dLLM_v2_7B \
  --model-name fast_dllm_v2 \
  --decoding-strategy multi_bd \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 1024 \
  --max-num-batched-tokens 1024 \
  --max-num-reqs 24 \
  --block-size 32 \
  --buffer-size 1
```

For throughput-oriented serving, use higher concurrency:

```bash
python -m diffulex.server.launch \
  --model /YOUR-CKPT-PATH/Efficient-Large-Model/Fast_dLLM_v2_7B \
  --model-name fast_dllm_v2 \
  --decoding-strategy multi_bd \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 1024 \
  --max-num-batched-tokens 1024 \
  --max-num-reqs 48 \
  --block-size 32 \
  --buffer-size 1
```

Keep `max_num_batched_tokens >= max_model_len`.

## 4. Model Startup

### 4.1 Server Startup

```bash
python -m diffulex.server.launch \
  --model /YOUR-CKPT-PATH/Efficient-Large-Model/Fast_dLLM_v2_7B \
  --model-name fast_dllm_v2 \
  --decoding-strategy multi_bd \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 1024 \
  --max-num-batched-tokens 1024 \
  --max-num-reqs 24 \
  --block-size 32 \
  --buffer-size 1
```

### 4.2 Benchmark Startup

```bash
python -m diffulex_bench.main \
  --config diffulex_bench/configs/fast_dllm_v2_gsm8k.yml \
  --model-path /YOUR-CKPT-PATH/Efficient-Large-Model/Fast_dLLM_v2_7B \
  --tokenizer-path /YOUR-CKPT-PATH/Efficient-Large-Model/Fast_dLLM_v2_7B \
  --dataset-limit 400 \
  --max-model-len 1024 \
  --max-num-batched-tokens 1024 \
  --max-num-reqs 24 \
  --block-size 32 \
  --engine-arg buffer_size=1 \
  --output-dir /YOUR-OUTPUT-PATH/fast_dllm_v2_gsm8k
```

## 5. Benchmark

### 5.1 Accuracy Benchmark

Use the benchmark command above for GSM8K exact-match evaluation. For balanced serving, use the moderate-concurrency preset.

### 5.2 Speed Benchmark

For throughput-focused evaluation, increase `max_num_reqs` and monitor accuracy. The throughput-oriented preset uses higher concurrency while keeping `buffer_size=1`.

