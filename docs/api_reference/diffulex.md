# diffulex

The `diffulex` package is the public Python entry point for in-process
inference. Importing the package is intentionally lightweight: the root module
uses lazy imports so that `from diffulex import Diffulex, SamplingParams` does
not eagerly import the full engine, CUDA kernels, or model stack until the
engine is constructed.

## Public Symbols

- `Diffulex`
- `SamplingParams`
- `get_logger`
- `setup_logger`
- `LoggerMixin`

## Diffulex

`Diffulex` is a thin public alias for `diffulex.engine.engine.DiffulexEngine`.
Constructing it validates the engine configuration, loads tokenizer metadata,
initializes the model runner, and creates the strategy-specific scheduler.

```python
from diffulex import Diffulex

llm = Diffulex(
    model="/path/to/model",
    model_name="llada",
    decoding_strategy="d2f",
    tensor_parallel_size=1,
    data_parallel_size=1,
)
```

The `model` argument must point to a local model directory. Most keyword
arguments are passed through to `diffulex.config.Config`; unsupported keyword
arguments are ignored by the engine constructor.

## Generation

Use `generate` for offline batched inference:

```python
from diffulex import SamplingParams

outputs = llm.generate(
    ["Solve 2 + 2."],
    SamplingParams(temperature=0.0, max_tokens=32),
)

for item in outputs.trajectories:
    print(item.text)
```

`generate` accepts a list of strings or a list of token ID lists. When prompts
are strings, the engine tokenizes them with the model tokenizer. The return
value records generated text, token IDs, request trajectories, and timing data.

## Request Lifecycle

Lower-level callers can use the step API:

- `add_request(prompt, sampling_params)` adds a request and returns its request ID.
- `step()` advances the scheduler and model runner by one engine step.
- `is_finished()` reports whether all queued requests are complete.
- `abort_request(req_id)` asks the scheduler to stop a request.
- `exit()` tears down model workers and profiling sessions.

Call `exit()` when embedding Diffulex in a long-running process. The engine also
registers an `atexit` hook and signal handlers, but explicit shutdown makes
resource ownership clearer in tests and services.

## SamplingParams

`SamplingParams` controls generation behavior for each request:

| Parameter | How to set it | What it does |
| --- | --- | --- |
| `temperature` | Use `0.0` for deterministic runs, or a higher float for sampling. | Controls generation randomness. |
| `max_tokens` | Use a positive output-token limit. | Caps the number of generated tokens. |
| `max_nfe` | Use a positive integer, or leave it unset. | Caps the number of forward evaluations when the strategy supports that limit. |
| `max_repetition_run` | Use a positive integer, or leave it unset. | Stops generation after a long repeated-token run. |
| `ignore_eos` | Leave `False` for normal generation; set `True` only when a task requires it. | Lets generation continue after EOS. |

When set, `max_nfe` and `max_repetition_run` must be positive.

## Logging Helpers

`get_logger`, `setup_logger`, and `LoggerMixin` are re-exported for callers that
want logging behavior consistent with Diffulex internals.

## Root Modules

| Module | Source |
| --- | --- |
| `diffulex.config` | `diffulex/config.py` |
| `diffulex.diffulex` | `diffulex/diffulex.py` |
| `diffulex.logger` | `diffulex/logger.py` |
| `diffulex.profiling` | `diffulex/profiling.py` |
| `diffulex.sampling_params` | `diffulex/sampling_params.py` |
| `diffulex.vllm_compat` | `diffulex/vllm_compat.py` |

## Subpackages

The package map below is intentionally limited to two levels: each page covers
one direct `diffulex.*` package and lists only its direct children.

:::{toctree}
:maxdepth: 1

diffulex.attention
diffulex.distributed
diffulex.engine
diffulex.layer
diffulex.mixin
diffulex.model
diffulex.moe
diffulex.sampler
diffulex.server
diffulex.strategy
diffulex.strategy_template
diffulex.utils
:::
