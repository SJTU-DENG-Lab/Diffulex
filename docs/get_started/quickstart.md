# Quickstart

This quickstart takes the shortest path through the main Diffulex workflows:

- Offline batched inference
- Online serving
- Benchmarking
- choosing a decoding strategy

For deeper workflow explanations, see the [Tutorials](../tutorials/index.md).

## Prerequisites

- Diffulex is [installed](installation.md) in a Python environment.
- NVIDIA GPUs are available for inference.
- The model checkpoint and optional LoRA checkpoint are available locally.

The examples below use D2F-LLaDA. Replace the checkpoint paths with paths from
your own environment.

## Installation

If you have not installed Diffulex yet, use the NVIDIA CUDA path:

`````{tabs}

````{tab} NVIDIA CUDA

```bash
uv venv --python 3.11 --seed
source .venv/bin/activate
uv pip install -e .
uv pip install vllm==0.19.1 --torch-backend=auto
```

````

````{tab} Other Environment

Currently, only CUDA Linux environments are fully supported.

````

`````

For more details, see [Installation](installation.md).

## Offline Batched Inference

With Diffulex installed, the simplest workflow is offline generation inside
your own Python process.

Import the engine class and per-request sampling parameters:

```python
from diffulex import Diffulex, SamplingParams
```

Construct the engine from a local model directory. The example starts in eager
mode to make first-run debugging easier:

```python
llm = Diffulex(
    model="/YOUR-CKPT-PATH/GSAI-ML/LLaDA-8B-Instruct",
    model_name="llada",
    decoding_strategy="d2f",
    tensor_parallel_size=2,
    data_parallel_size=1,
    max_model_len=2048,
    max_num_batched_tokens=2048,
    max_num_reqs=32,
    use_lora=True,
    lora_path="/YOUR-CKPT-PATH/SJTU-DENG-Lab/D2F_LLaDA_Instruct_8B_Lora",
    pre_merge_lora=True,
    enforce_eager=True,
)
```

Define prompts and generation parameters:

```python
prompts = [
    "Solve: Natalia sold clips to 48 friends in April, and half as many in May.",
]
sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
```

Run generation and read the generated text:

```python
outputs = llm.generate(prompts, sampling_params)

for output in outputs.trajectories:
    print(output.text)

llm.exit()
```

`Diffulex.generate` returns a `GenerationOutputs` object. Each item in
`outputs.trajectories` stores the generated text, token IDs, request trajectory,
and timing data.

## Online Serving

For interactive use, run Diffulex behind the HTTP server.

Start the server:

```bash
python -m diffulex.server.launch \
  --model /YOUR-CKPT-PATH/GSAI-ML/LLaDA-8B-Instruct \
  --model-name llada \
  --decoding-strategy d2f \
  --tensor-parallel-size 2 \
  --data-parallel-size 1 \
  --max-model-len 2048 \
  --max-num-batched-tokens 2048 \
  --max-num-reqs 32 \
  --use-lora \
  --lora-path /YOUR-CKPT-PATH/SJTU-DENG-Lab/D2F_LLaDA_Instruct_8B_Lora \
  --pre-merge-lora \
  --enforce-eager
```

For a local chat UI, start the sample Streamlit frontend after the server is
running:

```bash
streamlit run examples/streamlit_block_append_chat.py -- --base-url http://localhost:8000
```

## Benchmarking

Use `diffulex_bench` for dataset-backed evaluation workloads such as GSM8K,
HumanEval, or MBPP.

Run a D2F-LLaDA GSM8K evaluation:

```bash
python -m diffulex_bench.main \
  --config diffulex_bench/configs/llada_instruct_gsm8k.yml \
  --model-path /YOUR-CKPT-PATH/GSAI-ML/LLaDA-8B-Instruct \
  --tokenizer-path /YOUR-CKPT-PATH/GSAI-ML/LLaDA-8B-Instruct \
  --model-name llada \
  --decoding-strategy d2f \
  --use-lora \
  --lora-path /YOUR-CKPT-PATH/SJTU-DENG-Lab/D2F_LLaDA_Instruct_8B_Lora \
  --dataset gsm8k_diffulex \
  --dataset-limit 100 \
  --temperature 0.0 \
  --max-tokens 256
```

The config file provides default engine and evaluation settings. Command line
flags override matching config values.

## On Decoding Strategies

Set `decoding_strategy` in Python or `--decoding-strategy` on the command line.
The built-in strategies are:

| Strategy | Use it for |
| --- | --- |
| `d2f` | D2F-style block diffusion. |
| `multi_bd` | Multi-block diffusion strategies such as Fast-dLLM-v2 and SDAR. |
| `dmax` | DMax-style token merging on supported LLaDA2 models. |

Some model names force compatible defaults during config validation. For
example, `model_name="diffusion_gemma"` uses `multi_bd` and a larger block/page
size.
