# Get Started

Start here if you want the shortest path to using Diffulex.

## Engine

The core engine is constructed from a local model directory:

```python
from diffulex import Diffulex

llm = Diffulex(
    model_path="/YOUR-CKPT-PATH/your-model",
    model_name="your_model_name",
    decoding_strategy="d2f",
    mask_token_id=151666,
)
```

## Benchmark

Use `diffulex_bench` to run evaluation jobs:

```bash
python -m diffulex_bench.main --config diffulex_bench/configs/example.yml
```

For model-specific benchmark launch patterns, see the [Cookbook](../cookbook/index.md).

## Server

Use the HTTP server when you want to serve requests interactively:

```bash
python -m diffulex.server.launch --model /YOUR-CKPT-PATH/your-model --model-name your_model_name
```

For server launch patterns and the Streamlit sample frontend, see the [Cookbook](../cookbook/index.md).
