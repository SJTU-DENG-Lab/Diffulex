# LLaDA-8B-Instruct (D2F)

## 1. Model Introduction

LLaDA-8B-Instruct uses `d2f` decoding with the D2F LoRA adapter. This recipe covers serving and GSM8K benchmark commands, with separate recommendations for high-concurrency serving and low-concurrency speed.

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
  --model /YOUR-CKPT-PATH/GSAI-ML/LLaDA-8B-Instruct \
  --model-name llada \
  --decoding-strategy d2f \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-batched-tokens 8192 \
  --max-num-reqs 24 \
  --block-size 32 \
  --buffer-size 1 \
  --accept-threshold 0.95 \
  --semi-complete-threshold 0.9 \
  --add-block-threshold 0.1 \
  --use-lora \
  --lora-path /YOUR-CKPT-PATH/SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora
```

### 3.2 Configuration Tips

There is a serving trade-off between high concurrency and low-concurrency speed.

For general high-concurrency serving, use a smaller active block buffer:

```bash
python -m diffulex.server.launch \
  --model /YOUR-CKPT-PATH/GSAI-ML/LLaDA-8B-Instruct \
  --model-name llada \
  --decoding-strategy d2f \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-batched-tokens 8192 \
  --max-num-reqs 24 \
  --block-size 32 \
  --buffer-size 1 \
  --accept-threshold 0.95 \
  --semi-complete-threshold 0.9 \
  --add-block-threshold 0.1 \
  --use-lora \
  --lora-path /YOUR-CKPT-PATH/SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora
```

For low-concurrency speed, use a slightly larger active block buffer:

```bash
python -m diffulex.server.launch \
  --model /YOUR-CKPT-PATH/GSAI-ML/LLaDA-8B-Instruct \
  --model-name llada \
  --decoding-strategy d2f \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-batched-tokens 8192 \
  --max-num-reqs 2 \
  --block-size 32 \
  --buffer-size 2 \
  --accept-threshold 0.95 \
  --semi-complete-threshold 0.9 \
  --add-block-threshold 0.1 \
  --use-lora \
  --lora-path /YOUR-CKPT-PATH/SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora
```

For interactive serving, keep `max_num_batched_tokens` larger than `max_model_len`; the recommended server preset uses `8192` with `max_model_len` `4096` to leave room for longer chat turns. If GPU memory becomes the bottleneck, reduce `max_num_reqs` before lowering the context length. `block_size` must be one of `4`, `8`, `16`, or `32`.

## 4. Model Startup

### 4.1 Server Startup

```bash
python -m diffulex.server.launch \
  --model /YOUR-CKPT-PATH/GSAI-ML/LLaDA-8B-Instruct \
  --model-name llada \
  --decoding-strategy d2f \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-batched-tokens 8192 \
  --max-num-reqs 24 \
  --block-size 32 \
  --buffer-size 1 \
  --accept-threshold 0.95 \
  --semi-complete-threshold 0.9 \
  --add-block-threshold 0.1 \
  --use-lora \
  --lora-path /YOUR-CKPT-PATH/SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora
```

### 4.2 Benchmark Startup

```bash
python -m diffulex_bench.main \
  --config diffulex_bench/configs/llada_instruct_gsm8k.yml \
  --model-path /YOUR-CKPT-PATH/GSAI-ML/LLaDA-8B-Instruct \
  --tokenizer-path /YOUR-CKPT-PATH/GSAI-ML/LLaDA-8B-Instruct \
  --lora-path /YOUR-CKPT-PATH/SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora \
  --dataset-limit 400 \
  --max-model-len 1024 \
  --max-num-batched-tokens 1024 \
  --max-num-reqs 24 \
  --block-size 32 \
  --engine-arg buffer_size=1 \
  --output-dir /YOUR-OUTPUT-PATH/llada_instruct_gsm8k
```

## 5. Benchmark

### 5.1 Accuracy Benchmark

Use the benchmark command above for GSM8K exact-match evaluation. For general serving, prefer the high-concurrency preset. For low-concurrency throughput experiments, compare it with the low-concurrency speed preset.

### 5.2 Speed Benchmark

For throughput-focused evaluation, sweep `max_num_reqs` under the two presets. In our tests, `buffer_size=1` is the safer default for high-concurrency serving, while `buffer_size=2` is useful as a low-concurrency speed option.
