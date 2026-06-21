# Quickstart

This page gives the shortest working path for the current codebase:

- install Diffulex;
- run a small LLaDA2-mini benchmark;
- run one in-process Python generation;
- start the HTTP server;
- optionally run the vLLM DiffusionGemma baseline.

For reproducing the MBD LMS experiments, use the `mbd-lms` branch. The current
main branch contains ongoing runtime and model-specific optimizations.

## Prerequisites

- Diffulex is installed in a Python environment. See [Installation](installation.md).
- At least one NVIDIA GPU is visible to PyTorch.
- The model checkpoint exists locally.

The examples below use LLaDA2-mini:

```bash
export MODEL_PATH=/data/ckpts/inclusionAI/LLaDA2.0-mini
```

Replace this path with the location of the checkpoint on your machine.

## 1. Install

From the repository root:

```bash
uv venv --python 3.11 --seed
source .venv/bin/activate
uv pip install -e .
uv pip install vllm
```

Verify the install:

```bash
python -c "from diffulex import Diffulex, SamplingParams; print('ok')"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## 2. Run a Small Benchmark

Use the maintained LLaDA2-mini GSM8K runner. Start with a small limit:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH="$MODEL_PATH" \
DATASET_LIMIT=10 \
script/run_llada2_mini_gsm8k.sh
```

The script wraps:

```bash
python -m diffulex_bench.main \
  --config diffulex_bench/configs/llada2_mini_gsm8k.yml \
  --model-path "$MODEL_PATH" \
  --dataset-limit 10
```

Results are written under `benchmark_results/llada2_mini_gsm8k/` by default.
Remove `DATASET_LIMIT` only after the limited run loads the model, generates
answers, and writes results correctly.

## 3. Run Python Inference

For a direct in-process call:

```python
from diffulex import Diffulex, SamplingParams

model_path = "/data/ckpts/inclusionAI/LLaDA2.0-mini"

llm = Diffulex(
    model=model_path,
    model_name="llada2_mini",
    decoding_strategy="multi_bd",
    sampling_mode="naive",
    mask_token_id=156895,
    tensor_parallel_size=1,
    data_parallel_size=1,
    gpu_memory_utilization=0.45,
    max_model_len=4096,
    max_num_batched_tokens=4096,
    max_num_reqs=1,
    block_size=32,
    buffer_size=1,
    page_size=32,
    attn_impl="triton_grouped",
    enable_prefix_caching=True,
    enable_full_static_runner=True,
    enable_vllm_layers=True,
)

outputs = llm.generate(
    ["Solve: Natalia sold clips to 48 friends in April, and half as many in May. How many clips did she sell in May?"],
    SamplingParams(temperature=0.0, max_tokens=256, max_nfe=1024),
)

for item in outputs.trajectories:
    print(item.text)

llm.exit()
```

Use `attn_impl="naive"` and `enforce_eager=True` only when debugging
correctness. Use the optimized settings when measuring throughput.

## 4. Start the HTTP Server

The server uses the same engine configuration, exposed as CLI flags. A minimal
single-GPU LLaDA2-mini command is:

```bash
CUDA_VISIBLE_DEVICES=0 python -m diffulex.server.launch \
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
  --attn-impl triton \
  --host 127.0.0.1 \
  --port 8000
```

Send a request:

```bash
curl -s http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Solve: 12 + 30.","temperature":0.0,"max_tokens":64,"max_nfe":256}' \
  | python -m json.tool
```

For streaming chat validation:

```bash
streamlit run examples/streamlit_block_append_chat.py -- --base-url http://127.0.0.1:8000
```

## 5. Run DiffusionGemma or vLLM Baselines

Diffulex has a native DiffusionGemma benchmark config:

```bash
CUDA_VISIBLE_DEVICES=0 python -m diffulex_bench.main \
  --config diffulex_bench/configs/diffusion_gemma_gsm8k.yml \
  --model-path /data/ckpts/google/diffusiongemma-26B-A4B-it \
  --dataset-limit 10
```

The repository also keeps a vLLM DiffusionGemma baseline runner. This is for
comparison, not for starting Diffulex:

```bash
CUDA_VISIBLE_DEVICES=0 \
CONFIG_PATH=examples/engine_lm_eval/configs/vllm_diffusion_gemma_gsm8k_smoke.yml \
script/run_vllm_diffusion_gemma_gsm8k.sh
```

Use the `*_full.yml` config only after the smoke run succeeds.

## Decoding Strategy Cheatsheet

| Strategy | Typical models | Notes |
| --- | --- | --- |
| `d2f` | D2F LoRA-style LLaDA, Dream, DiffuCoder paths | Full-prefix block decoding; disables prefix caching. |
| `multi_bd` | LLaDA2-mini, SDAR, Fast-dLLM-v2, stable DiffuCoder/Dream reasoner paths | Block-causal multi-block decoding; supports prefix caching. |
| `dmax` | Supported LLaDA2 edit-sampling experiments | Requires `sampling_mode="edit"`. |
| `diffusion_gemma` | DiffusionGemma | Native DiffusionGemma canvas/block decoder. |

For more detail, read [Configuration](../user_guide/configuration.md) and
[Benchmark](../user_guide/benchmark.md).
