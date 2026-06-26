# Benchmark

Use `diffulex_bench` for dataset-backed evaluation workloads such as GSM8K,
HumanEval, MBPP, and MATH500. The benchmark path wraps Diffulex as an
lm-evaluation-harness model and writes logs, samples, trajectories, and metrics
under the configured output directory.

## Recommended Workflow

1. Pick a config under `diffulex_bench/configs/`.
2. Override the local model path from the command line.
3. Start with `--dataset-limit`.
4. Inspect generated text and metrics.
5. Remove the limit only after the limited run is correct.

## LLaDA2-Mini GSM8K

The maintained LLaDA2-mini path is:

```bash
CUDA_VISIBLE_DEVICES=0 python -m diffulex_bench.main \
  --config diffulex_bench/configs/llada2_mini_gsm8k.yml \
  --model-path /path/to/LLaDA2.0-mini \
  --dataset-limit 10 \
  --output-dir benchmark_results/llada2_mini_gsm8k
```

The convenience wrapper is equivalent for most runs:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/path/to/LLaDA2.0-mini \
DATASET_LIMIT=10 \
script/run_llada2_mini_gsm8k.sh
```

The config defaults to single-request TP1 execution. Keep that path for
single-sample speed comparisons. Increase `max_num_reqs` or data parallelism
only when measuring aggregate throughput.

## DiffusionGemma

Native Diffulex DiffusionGemma benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 python -m diffulex_bench.main \
  --config diffulex_bench/configs/diffusion_gemma_gsm8k.yml \
  --model-path /path/to/diffusiongemma-26B-A4B-it \
  --dataset-limit 10
```

DiffusionGemma uses a 256-token block/page size and model-specific entropy-bound
sampling controls. Keep the provided config as the baseline unless you are
profiling a specific runtime change.

## vLLM DiffusionGemma Baseline

The vLLM runner is a baseline for comparison, not a Diffulex engine launcher:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL=/path/to/diffusiongemma-26B-A4B-it \
CONFIG_PATH=examples/engine_lm_eval/configs/vllm_diffusion_gemma_gsm8k_smoke.yml \
script/run_vllm_diffusion_gemma_gsm8k.sh
```

Use `examples/engine_lm_eval/configs/vllm_diffusion_gemma_gsm8k_full.yml` for
the full run after the smoke run passes.

## Other Configs

| Config | Typical use |
| --- | --- |
| `diffucoder_instruct_gsm8k.yml` | DiffuCoder D2F-style GSM8K. |
| `dream_base_gsm8k.yml` | Dream D2F-style GSM8K. |
| `fast_dllm_v2_gsm8k.yml` | Fast-dLLM-v2 multi-block GSM8K. |
| `sdar_chat_gsm8k.yml` | SDAR dense GSM8K. |
| `sdar_moe_chat_gsm8k.yml` | SDAR-MoE GSM8K. |
| `llada2_mini_dmax_gsm8k.yml` | LLaDA2-mini DMax/edit sampling. |

Most configs include development-cluster paths. Override them with
`--model-path`, or copy the YAML and edit the path for repeated runs.

## Output Files

A normal run writes to `output_dir/run_<timestamp>_<task>/` unless
`use_run_subdirectory` is disabled. Inspect:

| Artifact | Use |
| --- | --- |
| Benchmark log | Engine args, progress logs, startup errors. |
| `diffulex_stats.json` | Token counts, NFE counts, aggregate throughput, per-sample throughput. |
| Response JSON files | Full, truncated, and extracted responses. |
| Decode trajectory | Per-step output trajectory when enabled by the benchmark path. |
| lm-eval result files | Task metrics and per-sample records. |

Diffulex reports both aggregate throughput and per-sample mean throughput. For
engine comparison, aggregate throughput is usually the better capacity metric;
per-sample mean remains useful for single-request latency behavior.

## Common Overrides

| Override | Example | Notes |
| --- | --- | --- |
| Model path | `--model-path /path/to/model` | Prefer CLI override over editing checked-in configs. |
| Dataset limit | `--dataset-limit 10` | Use for smoke tests. Omit for full evaluation. |
| Token cap | `--max-tokens 2048` | Raise only if outputs are truncated. |
| NFE cap | `--max-nfe 1024` | Use when comparing diffusion step budgets. |
| Output dir | `--output-dir benchmark_results/my_run` | Keeps runs grouped by experiment. |
| Progress bars | `eval.use_tqdm: false` in YAML | Avoids terminal stalls during long runs. |

For exact CLI options, run:

```bash
python -m diffulex_bench.main --help
```
