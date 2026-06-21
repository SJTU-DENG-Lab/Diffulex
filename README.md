# Diffulex

Diffulex is a flexible and extensible inference engine for block-style and
canvas-style diffusion language models. It is the runtime layer for turning new
dLLM decoding ideas into runnable, measurable systems, rather than a collection
of one-off benchmark scripts.

Researchers can use Diffulex to prototype a decoding strategy, connect it to
real scheduling and KV-cache behavior, serve it through the engine, profile its
systems cost, and compare it under aligned benchmark scripts without rebuilding
the whole inference stack from scratch.

The engine follows a strategy-oriented design. A decoding paradigm is expressed
through coordinated runtime components:

- request state;
- scheduler;
- KV cache manager;
- model runner;
- sampler;
- attention metadata;
- benchmark and serving entry points.

This separation makes Diffulex suitable for rapid research iteration and
agent-assisted engineering. With the existing strategies as references,
developers can efficiently use coding agents such as Claude Code or Codex to add
a new decoding algorithm, wire it through the engine stack, and immediately
evaluate correctness, throughput, and model quality.

## Why Diffulex

MBD-LMs show that block diffusion inference is not a single fixed algorithm.
The same broad paradigm can cover multiple block-style dLLM inference modes:

| Paradigm / mechanism | What Diffulex is meant to support |
|---|---|
| SingleBD | Native one-block-at-a-time block diffusion decoding. |
| MultiBD | A bounded running-set of active blocks with Block Buffer-style execution. |
| DualCache | Future cache designs that need separate cache views or cache lifecycles. |
| TokenMerge | Token-merge decoding paths such as DMax-style parallel decoding. |
| Edit | Edit/remask refinement paths for compatible diffusion models. |
| Uniform DLM | Non-block or canvas-style denoising models such as DiffusionGemma. |

Diffulex gives these algorithms a common systems substrate: paged KV cache,
prefix reuse, block scheduling, static-shape execution, optimized attention,
optional vLLM-backed layers, MoE paths, benchmark tooling, and HTTP serving.

The intended workflow is:

1. define the decoding state and acceptance rule;
2. implement the scheduler/cache/runner/sampler hooks by following the closest
   existing strategy;
3. run the benchmark or serving entry point;
4. inspect throughput, per-request statistics, generated outputs, and profiles;
5. iterate with normal code review or with Claude Code, Codex, and similar
   coding agents.

## Branches and Use Cases

| Branch | Use case |
|---|---|
| `mbd-lms` | Reproduce the MBD-LMs experiments with the aligned configs and scripts below. |
| `main` | Active engine development, open-source contribution, and new dLLM decoding algorithms. |

If your goal is to reproduce reported MBD-LMs results, stay on this branch. If
your goal is to build new runtime features or new decoding strategies, start
from `main`.

## Extending the Engine

The fastest way to add a new algorithm is to start from the closest existing
strategy:

| New idea | Closest reference |
|---|---|
| Single-block BD-LM inference | SingleBD / native block-diffusion configs |
| Multi-block decoding | `multi_bd` |
| Token merging or DMax-like decoding | `dmax` / TokenMerge paths |
| Edit/remask refinement | edit sampling paths |
| DiffusionGemma-like denoising | `diffusion_gemma` |

Implement the strategy-specific request state, scheduler behavior, cache
metadata, runner preparation, and sampler logic, then validate it with the
benchmark scripts. The existing code structure is intentionally regular so that
Claude Code, Codex, or similar coding agents can help propagate a new strategy
through the engine consistently.

## Run Experiments

Experiment configs live in:

```bash
diffulex_bench/configs/experiment/
```

Core hyperparameters are mirrored in those config files. Common values for all rows:

```text
max_model_len=4096, max_new_tokens=4096, max_nfe=1024
```

| Configuration | Variant | Task | Buffer | Block | tau_add | tau_semi | tau_stable | tau_M2T | tau_T2T |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaDA2-Mini-DMax | SingleBD Native | Math | 1 | 32 | - | - | - | 0.50 | - |
| LLaDA2-Mini-DMax | SingleBD Native | Code | 1 | 32 | - | - | - | 0.65 | - |
| LLaDA2-Mini-DMax | MultiBD training-free | Math | 2 | 32 | 0.10 | 0.90 | 0.50 | 0.50 | - |
| LLaDA2-Mini-DMax | MultiBD training-free | Code | 2 | 32 | 0.90 | 0.90 | 0.50 | 0.65 | - |
| MBD-LLaDA2-Mini-DMax | MBD | Math | 2 | 32 | 0.10 | 0.90 | 0.50 | 0.50 | - |
| MBD-LLaDA2-Mini-DMax | MBD | Code | 2 | 32 | 0.90 | 0.90 | 0.50 | 0.65 | - |
| LLaDA2-Mini | SingleBD Native | Math | 1 | 32 | - | - | - | 0.95 | - |
| LLaDA2-Mini | SingleBD Native | Code | 1 | 32 | - | - | - | 0.95 | - |
| LLaDA2-Mini | MultiBD training-free | Math | 2 | 32 | 0.10 | 0.90 | - | 0.95 | - |
| LLaDA2-Mini | MultiBD training-free | Code | 2 | 32 | 0.90 | 0.90 | - | 0.95 | - |
| MBD-LLaDA2-Mini | MBD | Math | 2 | 32 | 0.10 | 0.90 | - | 0.95 | - |
| MBD-LLaDA2-Mini | MBD | Code | 2 | 32 | 0.90 | 0.90 | - | 0.95 | - |
| SDAR-8B-Chat-b32 | SingleBD Native | Math | 1 | 32 | - | - | - | 0.95 | - |
| SDAR-8B-Chat-b32 | SingleBD Native | Code | 1 | 32 | - | - | - | 0.95 | - |
| SDAR-8B-Chat-b32 | MultiBD training-free | Math | 4 | 32 | 0.10 | 0.90 | - | 0.95 | - |
| SDAR-8B-Chat-b32 | MultiBD training-free | Code | 4 | 32 | 0.90 | 0.90 | - | 0.95 | - |
| MBD-SDAR-8B-Chat-b32 | MBD | Math | 4 | 32 | 0.10 | 0.90 | - | 0.95 | - |
| MBD-SDAR-8B-Chat-b32 | MBD | Code | 4 | 32 | 0.90 | 0.90 | - | 0.95 | - |
| SDAR-8B-Chat-b4 | SingleBD Native | Math | 1 | 4 | - | - | - | 0.95 | - |
| SDAR-8B-Chat-b4 | SingleBD Native | Code | 1 | 4 | - | - | - | 0.95 | - |
| SDAR-8B-Chat-b4 | MultiBD training-free | Math | 4 | 4 | 0.10 | 0.25 | - | 0.95 | - |
| SDAR-8B-Chat-b4 | MultiBD training-free | Code | 4 | 4 | 0.75 | 0.75 | - | 0.95 | - |
| MBD-SDAR-8B-Chat-b4 | MBD | Math | 4 | 4 | 0.10 | 0.25 | - | 0.95 | - |
| MBD-SDAR-8B-Chat-b4 | MBD | Code | 4 | 4 | 0.75 | 0.75 | - | 0.95 | - |
| LLaDA2-Mini-CAP | SingleBD Native | Math | 1 | 32 | - | - | - | 0.95 | - |
| LLaDA2-Mini-CAP | SingleBD Native | Code | 1 | 32 | - | - | - | 0.95 | - |
| LLaDA2-Mini-CAP | MultiBD training-free | Math | 2 | 32 | 0.10 | 0.90 | - | 0.95 | - |
| LLaDA2-Mini-CAP | MultiBD training-free | Code | 2 | 32 | 0.90 | 0.90 | - | 0.95 | - |
| LLaDA2.1-Mini | SingleBD Native | Math | 1 | 32 | - | - | - | 0.70 | 0.50 |
| LLaDA2.1-Mini | SingleBD Native | Code | 1 | 32 | - | - | - | 0.70 | 0.50 |
| LLaDA2.1-Mini | MultiBD training-free | Math | 2 | 32 | 0.10 | 0.90 | - | 0.70 | 0.50 |
| LLaDA2.1-Mini | MultiBD training-free | Code | 2 | 32 | 0.90 | 0.90 | - | 0.70 | 0.50 |

The only experiment entrypoint is:

```bash
./script/run_batch_experiments.sh
```

Preview the run plan without launching models:

```bash
DRY_RUN=1 ./script/run_batch_experiments.sh
```

Run all experiment configs:

```bash
./script/run_batch_experiments.sh
```

Run selected config files:

```bash
CONFIG_FILES=llada2_mini.yml ./script/run_batch_experiments.sh
CONFIG_FILES=llada2_mini.yml,sdar_8b_chat_b32.yml ./script/run_batch_experiments.sh
```

Filter selected experiments by name/group/task/model:

```bash
FILTER=multibd_math ./script/run_batch_experiments.sh
FILTER=llada2_mini DATASET_LIMIT=10 ./script/run_batch_experiments.sh
```

Override capacity or checkpoint paths:

```bash
MAX_NUM_REQS=256 ./script/run_batch_experiments.sh
LLADA2_MINI_MODEL=/data/ckpts/inclusionAI/LLaDA2.0-mini ./script/run_batch_experiments.sh
SDAR_B32_MODEL=/path/to/SDAR-8B-Chat-b32 ./script/run_batch_experiments.sh
```

If some configured checkpoints are unavailable and you want to run only the available ones:

```bash
SKIP_MISSING_MODELS=1 ./script/run_batch_experiments.sh
```

Outputs are written to:

```bash
benchmark_results/experiment/<run_id>/
logs/experiment/<run_id>/
```

Each run also writes resolved per-experiment benchmark YAMLs under:

```bash
benchmark_results/experiment/<run_id>/resolved_configs/
```
