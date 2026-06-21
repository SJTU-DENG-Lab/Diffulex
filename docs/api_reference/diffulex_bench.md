# diffulex_bench

The `diffulex_bench` package wraps Diffulex engine configuration and
lm-evaluation-harness execution. It is the preferred path for repeatable
evaluation runs because it keeps engine arguments, dataset settings, output
paths, and logging in one configuration flow.

## Public Symbols

- `BenchmarkRunner`
- `load_benchmark_dataset`
- `compute_metrics`
- `BenchmarkConfig`
- `EngineConfig`
- `EvalConfig`
- `DiffulexLM`

## Configuration Classes

`BenchmarkConfig` groups two main sections:

| Config class | What it contains |
| --- | --- |
| `EngineConfig` | Model path, tokenizer path, model family, decoding strategy, parallelism, LoRA settings, cache layout, thresholds, and runtime options. |
| `EvalConfig` | Dataset name, split, limits, generation limits, output directory, result saving, and lm-eval include path. |

Configs can be loaded from YAML or JSON and then overridden from the command
line. This is useful for keeping a reusable base config while changing model
paths or dataset limits per run.

```python
from diffulex_bench.config import BenchmarkConfig

config = BenchmarkConfig.from_yaml("diffulex_bench/configs/example.yml")
```

## lm-eval Integration

`diffulex_bench.main` converts `BenchmarkConfig` into lm-eval model arguments
and then invokes lm-evaluation-harness with the `diffulex` model adapter. The
adapter is registered by importing `diffulex_bench.lm_eval_model`.

The benchmark entry point also handles:

- encoding and decoding complex engine values for lm-eval model args;
- default task include paths under `diffulex_bench/tasks`;
- optional task data file overrides for local JSON datasets;
- per-run output directories with timestamped names;
- sample logging when result saving is enabled.

## Dataset Helpers

`load_benchmark_dataset` and `compute_metrics` support the older local runner
path. New evaluation workflows should prefer the lm-eval based CLI unless a
custom runner needs direct access to dataset records or metrics.

## Typical Usage

Run from the command line for normal evaluations:

```bash
python -m diffulex_bench.main \
  --config diffulex_bench/configs/example.yml \
  --model-path /path/to/model \
  --dataset gsm8k_diffulex \
  --dataset-limit 100
```

Use the Python API when writing tests or custom scripts that need to construct
or inspect benchmark configuration before launching a run.
