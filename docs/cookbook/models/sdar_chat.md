# SDAR-1.7B-Chat-b32 (MultiBD)

## 1. Model Introduction

SDAR-1.7B-Chat-b32 uses `multi_bd` decoding and is a stable low-concurrency baseline.

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
  --model /YOUR-CKPT-PATH/SDAR/SDAR-1.7B-Chat-b32 \
  --model-name sdar \
  --decoding-strategy multi_bd \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-batched-tokens 8192 \
  --max-num-reqs 1 \
  --block-size 32 \
  --buffer-size 1
```

### 3.2 Configuration Tips

There is a trade-off between stable baseline quality and higher throughput.

For a stable quality-oriented baseline, use low concurrency:

```bash
python -m diffulex.server.launch \
  --model /YOUR-CKPT-PATH/SDAR/SDAR-1.7B-Chat-b32 \
  --model-name sdar \
  --decoding-strategy multi_bd \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-batched-tokens 8192 \
  --max-num-reqs 1 \
  --block-size 32 \
  --buffer-size 1
```

For throughput-oriented serving, use higher concurrency:

```bash
python -m diffulex.server.launch \
  --model /YOUR-CKPT-PATH/SDAR/SDAR-1.7B-Chat-b32 \
  --model-name sdar \
  --decoding-strategy multi_bd \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-batched-tokens 8192 \
  --max-num-reqs 4 \
  --block-size 32 \
  --buffer-size 1
```

For interactive serving, keep `max_num_batched_tokens` larger than `max_model_len`; the recommended server preset uses `8192` with `max_model_len` `4096` to leave room for longer chat turns. If GPU memory becomes the bottleneck, reduce `max_num_reqs` before lowering the context length.

## 4. Model Startup

### 4.1 Server Startup

```bash
python -m diffulex.server.launch \
  --model /YOUR-CKPT-PATH/SDAR/SDAR-1.7B-Chat-b32 \
  --model-name sdar \
  --decoding-strategy multi_bd \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-batched-tokens 8192 \
  --max-num-reqs 1 \
  --block-size 32 \
  --buffer-size 1
```

### 4.2 Benchmark Startup

```bash
python -m diffulex_bench.main \
  --config diffulex_bench/configs/sdar_chat_gsm8k.yml \
  --model-path /YOUR-CKPT-PATH/SDAR/SDAR-1.7B-Chat-b32 \
  --dataset-limit 400 \
  --max-model-len 2048 \
  --max-num-batched-tokens 4096 \
  --max-num-reqs 1 \
  --block-size 32 \
  --engine-arg buffer_size=1 \
  --output-dir /YOUR-OUTPUT-PATH/sdar_chat_gsm8k
```

## 5. Benchmark

### 5.1 Accuracy Benchmark

Use the benchmark command above for GSM8K exact-match evaluation. For stable baseline comparisons, use the low-concurrency preset.

### 5.2 Speed Benchmark

For throughput-focused evaluation, increase `max_num_reqs` and monitor accuracy. The throughput-oriented preset uses higher concurrency while keeping `buffer_size=1`.
