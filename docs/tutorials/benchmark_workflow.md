# Benchmark Workflow Deep Dive

Use `diffulex_bench` for evaluation workloads such as GSM8K, HumanEval, MBPP,
and MATH500. The benchmark path wraps Diffulex as an lm-evaluation-harness
model, so task definitions and result files follow lm-eval conventions while
engine settings remain Diffulex-specific.

## Workflow

1. Select a benchmark config under `diffulex_bench/configs/`.
2. Override model paths and engine settings from the command line.
3. Run `python -m diffulex_bench.main`.
4. Inspect the generated run directory and log file.
5. Repeat with one controlled change at a time.

## Config Files

Configs group engine and evaluation settings. Use them for values that should
stay stable across runs, such as decoding strategy, model family, cache layout,
thresholds, and default dataset.

Command line flags override matching config values. This lets you reuse a config
while changing paths or dataset limits:

```bash
python -m diffulex_bench.main \
  --config diffulex_bench/configs/example.yml \
  --model-path /path/to/model \
  --tokenizer-path /path/to/tokenizer \
  --dataset gsm8k_diffulex \
  --dataset-limit 100 \
  --output-dir outputs/bench
```

Start with `--dataset-limit` while validating a new model or strategy. Remove
the limit only after model loading, generation, and result writing are stable.

## Task Resolution

By default, Diffulex uses bundled lm-eval task YAML files under
`diffulex_bench/tasks`. Use `--include-path` when running task definitions from
another directory. Use `--dataset-data-files` to override a task YAML's
`data_files` field with a local JSON file for one run.

Code-generation tasks may require `--confirm-run-unsafe-code` because lm-eval
executes generated code for scoring. Keep this explicit when running untrusted
models or unreviewed prompts.

## Output Layout

The benchmark runner writes results under `--output-dir`. By default, each run
uses a timestamped subdirectory with the task name in the path. This avoids
overwriting prior runs and keeps trajectory, sample, and metric files together.

For ad hoc testing, keep the default run subdirectory behavior. Disable it only
when an external script expects a stable output directory.

## Reading Results

Inspect these artifacts first:

- the benchmark log for model args, task names, and engine startup errors;
- lm-eval result JSON for aggregate scores;
- logged samples when `--save-results` is enabled;
- Diffulex trajectory or stats output when a run saves per-request data.

If scores look wrong, check that the task name, tokenizer path, `max_tokens`,
temperature, and decoding strategy match the intended experiment.

## Common Iteration Pattern

For a new configuration, use a gradual run sequence:

1. Run one or two examples with `--dataset-limit`.
2. Increase `--max-tokens` only if outputs are truncated.
3. Run a larger but still limited sample.
4. Only then run the full dataset.

This keeps model load errors, task wiring errors, and quality regressions
separate from long-running evaluation cost.
