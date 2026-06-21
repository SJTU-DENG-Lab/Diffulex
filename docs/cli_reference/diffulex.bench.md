# diffulex.bench

The benchmark CLI runs Diffulex through lm-evaluation-harness tasks. It is the
main command-line interface for GSM8K, MATH500, HumanEval, MBPP, and custom
lm-eval compatible task YAMLs.

The entry point is:

```bash
python -m diffulex_bench.main --help
```

## Config-First Usage

Use a YAML config when you want repeatable settings:

```bash
python -m diffulex_bench.main \
  --config diffulex_bench/configs/llada2_mini_gsm8k.yml \
  --model-path /path/to/LLaDA2.0-mini \
  --dataset-limit 10
```

Command line flags override matching config fields. This makes it practical to
keep one config per model family and vary paths, limits, and output directories
per run.

## Model and Strategy Arguments

| Flag | How to set it | What it does |
| --- | --- | --- |
| `--model-path` | Point to the local base-model checkpoint directory. Required unless the YAML already sets `engine.model_path`. | Passes the model weights path into Diffulex. |
| `--tokenizer-path` | Point to a tokenizer directory, or omit it to fall back to the model path in the benchmark config flow. | Lets lm-eval use a tokenizer stored separately from the weights. |
| `--model-name` | Use a registered model key: `dream`, `sdar`, `sdar_moe`, `fast_dllm_v2`, `llada`, `llada2`, `llada2_moe`, `llada2_mini`, `llada2dot1_mini`, `llada2_mini_dmax`, or `diffusion_gemma`. The default is `dream`. | Selects the model adapter and sampler defaults. |
| `--decoding-strategy` | Use `d2f`, `multi_bd`, `dmax`, or `diffusion_gemma` where supported by the selected model/config. | Chooses the strategy-specific request, scheduler, cache, runner, and attention metadata path. |
| `--sampling-mode` | Use `naive`, `edit`, or omit it and let config/model defaults apply. | Selects sampler behavior. `edit` is restricted to compatible LLaDA2-family names. |
| `--mask-token-id` | Use the tokenizer's mask token ID. The default is `151666`. | Supplies the mask token when tokenizer metadata does not override it. |

## Parallelism and Capacity Arguments

| Flag | How to set it | What it does |
| --- | --- | --- |
| `--tensor-parallel-size` | Use `1` to `8` ranks. The CLI default is `1`. | Splits one model replica across multiple GPUs. |
| `--data-parallel-size` | Use `1` to `1024` groups. The CLI default is `1`. | Runs independent evaluation groups when enough devices are available. |
| `--expert-parallel-size` | Omit it or keep the effective value at `1` for the current validated path. | Reserves the MoE expert-parallel dimension. |
| `--gpu-memory-utilization` | Use a fraction such as `0.9`. | Guides GPU memory planning for engine allocation. |
| `--max-model-len` | Use a positive sequence length. The default is `2048`, and the HF config may clamp it. | Sets the requested prompt-plus-output length limit. |
| `--max-num-batched-tokens` | Use a positive token budget. The default is `4096`; it must cover the effective model length. | Limits scheduler batch size by token count. |
| `--max-num-reqs` | Use a positive request count, or omit it to let config defaults apply. | Caps active requests and replaces deprecated `--max-num-seqs`. |
| `--max-num-seqs` | Deprecated; use `--max-num-reqs` for new commands. | Keeps older benchmark commands working when `--max-num-reqs` is unset. |

## Sampling Arguments

| Flag | How to set it | What it does |
| --- | --- | --- |
| `--temperature` | Use `0.0` for deterministic evaluation or a higher float for sampling. | Sets generation randomness for benchmark requests. |
| `--max-tokens` | Use a positive output-token limit. The default is `256`. | Caps generated tokens per request. |
| `--max-nfe` | Use a positive number of forward evaluations, or omit it. | Adds a hard evaluation-step bound when the strategy supports it. |
| `--max-repetition-run` | Use a positive run length, or omit it. | Stops a request after the generated suffix repeats one token for too long. |
| `--ignore-eos` | Add the flag only when a task should continue after EOS. | Prevents EOS from ending generation. |

## Dataset Arguments

| Flag | How to set it | What it does |
| --- | --- | --- |
| `--dataset` | Use an lm-eval task name. The default is `gsm8k_diffulex`. | Selects the benchmark task; bundled tasks live under `diffulex_bench/tasks`. |
| `--include-path` | Point to a directory of task YAMLs, omit it for bundled tasks, or pass an empty string to disable the bundled include path. | Controls where lm-eval looks for task definitions. |
| `--dataset-split` | Use the split name expected by the dataset. The default is `test`. | Passes the dataset split through config. |
| `--dataset-limit` | Use a positive number for smoke tests or partial runs, and omit it for the full task. | Limits how many examples are evaluated. |
| `--dataset-data-files` | Point to a local JSON file, or omit it. | Overrides `dataset_kwargs.data_files` in the task YAML. |
| `--confirm-run-unsafe-code` | Enable only for lm-eval tasks where executing generated code is expected and acceptable. | Allows code-execution tasks to run. |

## Output and Logging

| Flag | How to set it | What it does |
| --- | --- | --- |
| `--output-dir` | Point to an output directory. The default is `benchmark_results`. | Sets the base directory for benchmark artifacts. |
| `--use-run-subdirectory` / `--no-use-run-subdirectory` | Leave run subdirectories on for normal runs, or disable them when a fixed output path is needed. | Writes each run under `run_<timestamp>_<task>/`. |
| `--save-results` / `--no-save-results` | Leave saving on unless only logs are needed. | Controls result and sample output files. |
| `--log-file` | Point to a log file, or omit it for console logging. | Adds persistent benchmark logs. |
| `--log-level` | Use `DEBUG`, `INFO`, `WARNING`, or `ERROR`. The default is `INFO`. | Controls benchmark log verbosity. |

## LoRA and Runtime Controls

| Flag | How to set it | What it does |
| --- | --- | --- |
| `--use-lora` | Add the flag when benchmarking with an adapter. | Enables LoRA adapter loading. |
| `--lora-path` | Point to the adapter checkpoint directory. Required with `--use-lora`. | Loads the adapter weights. |
| `--pre-merge-lora` | Add when the adapter should be merged into the base model at load time. Benchmark YAML defaults to pre-merge on. | Avoids per-forward adapter compute when merging is supported. |
| `--enforce-eager` / `--no-enforce-eager` | Use eager mode for debugging, or explicitly allow optimized graph paths for measurement. | Overrides config-driven eager/optimized execution behavior. |
| `--kv-cache-layout` | Use `unified` for the default cache layout or `distinct` for strategy experiments. | Chooses KV cache storage layout. |
| `--attn-impl` | Use `triton` or `triton_grouped` for optimized runs, `naive` for debugging, or omit to keep config defaults. | Overrides the attention backend. |
| `--page-size` | Use `4`, `8`, `16`, or `32` for most models; DiffusionGemma uses `256`. | Sets the KV cache page size. |
| `--block-size` | Use `4`, `8`, `16`, or `32` for most models; DiffusionGemma uses `256`. Keep it no larger than `--page-size`. | Sets the token span of one diffusion block. |
| `--buffer-size` | Use a positive block count, or omit it and keep the config default of `4`. | Controls how many diffusion blocks can remain active. |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | Use the pair to make prefix-cache behavior explicit in an experiment command. | Enables or disables compatible prefix cache reuse. |

## Threshold and Token-Merge Controls

| Flag | How to set it | What it does |
| --- | --- | --- |
| `--add-block-threshold` | Omit it to use `0.1`, or pass a float for block-add tuning. | Controls when another decoding block can be added. |
| `--semi-complete-threshold` | Omit it to use `0.9`, or pass a float for block advancement tuning. | Controls when semi-complete block state can advance. |
| `--accept-threshold` | Use a confidence value from `0` to `1`. The default is `0.9`. | Accepts mask-to-token updates once confidence is high enough. |
| `--edit-threshold` | Use a confidence value from `0` to `1`. The default is `0.0`. | Accepts token-to-token edits in edit-style decoding. |
| `--remask-threshold` | Use a confidence value from `0` to `1`. The default is `0.4`. | Remasks filled tokens that fall below the confidence threshold. |
| `--token-stability-threshold` | Use a stability ratio from `0` to `1`. The default is `0.0`. | Controls when the next DMax edit block can be added. |
| `--token-merge-mode` | Use `dmax_topk`, `iter_smooth_topk`, or omit it and keep config defaults. | Selects token-merge metadata behavior for DMax-style strategies. |
| `--token-merge-top-k` | Use a positive candidate count, or omit it and keep the config default of `1`. | Keeps this many candidates in token-merge metadata. |
| `--token-merge-renormalize` | Add when the experiment should explicitly renormalize token-merge probabilities. | Controls probability normalization after token candidate filtering. |
| `--token-merge-weight` | Use a non-negative interpolation weight, or omit it and keep the config default of `1.0`. | Weights token-merge interpolation. |

For long-lived values, prefer YAML config. Use CLI overrides for experiment
specific changes that should be visible in the command history.
