# Model Loading and Configuration Walkthrough

This tutorial walks through the path from user configuration to a constructed
`Diffulex` engine. It focuses on what happens when you call `Diffulex(...)` and
which configuration fields affect the early load path.

## Starting Point

The public API is intentionally small:

```python
from diffulex import Diffulex, SamplingParams

llm = Diffulex(
    model="/path/to/model",
    model_name="llada",
    decoding_strategy="d2f",
    tensor_parallel_size=1,
    data_parallel_size=1,
)
```

`Diffulex` returns a `DiffulexEngine` instance. The constructor separates keyword
arguments that match `diffulex.config.Config` fields and ignores unrelated
keywords. This keeps the public constructor aligned with the engine config
without requiring a separate wrapper class.

## Configuration Creation

The first major step is building `Config(model, **config_kwargs)`. The `model`
path must be an existing local directory. Diffulex then validates model family,
decoding strategy, sampling mode, page and block sizes, cache layout, parallel
topology, LoRA settings, and runtime optimization flags.

Important model-specific behavior happens here:

| Condition | Normalized behavior |
| --- | --- |
| `decoding_strategy="d2f"` | Forces `multi_block_prefix_full=True` and disables prefix caching. |
| `decoding_strategy="multi_bd"` | Forces `multi_block_prefix_full=False`. |
| `decoding_strategy="dmax"` | Forces `multi_block_prefix_full=False` and requires an edit-sampling model with `sampling_mode="edit"`. |
| `model_name="diffusion_gemma"` | Forces `multi_bd`, `block_size=256`, `page_size=256`, and `buffer_size=1`. |

If a validation error is raised, fix the configuration before debugging model
weights or kernels.

## Tokenizer and HF Config

After config validation, the engine loads the tokenizer with
`auto_tokenizer_from_pretrained`. It records the tokenizer vocabulary size and
EOS token ID on the config. If the tokenizer exposes `mask_token_id`, Diffulex
uses that value instead of the default mask token.

`Config` also loads the Hugging Face config through `AutoConfig.from_pretrained`
with `trust_remote_code=True`. The effective `max_model_len` is clamped to the
model config's maximum sequence length.

## Worker Processes

Diffulex computes the model-parallel world size from tensor, expert, and data
parallel sizes. Rank 0 runs in the main process. Additional ranks are spawned as
worker processes with Python multiprocessing. Each worker constructs a model
runner from the same validated config.

If startup fails, the engine calls `exit()` to clean up worker processes before
re-raising the exception.

## Strategy Components

The decoding strategy selects several registered components:

- request state through `AutoReq`;
- scheduler through `AutoScheduler`;
- KV cache manager through `AutoKVCacheManager`;
- model runner through `AutoModelRunner`;
- attention metadata functions through the strategy model runner.

Built-in strategies are imported from `diffulex.strategy`, which triggers their
registry decorators.

## First Generation

Once the engine is constructed, use `generate` for normal offline inference:

```python
outputs = llm.generate(
    ["Solve: 12 + 30."],
    SamplingParams(temperature=0.0, max_tokens=32),
)

for output in outputs.trajectories:
    print(output.text)
```

Call `llm.exit()` when the process is done with the engine.

## Practical Debugging

Use a tiny prompt, `tensor_parallel_size=1`, and `data_parallel_size=1` while
validating a new model family. Once the model loads and one request completes,
increase parallelism and batch limits.
