# Adding a New Model Family

This tutorial walks through adding a model family to Diffulex. The goal is to
make the model load through the standard engine path, decode with a compatible
strategy, and pass a focused smoke test before broad benchmarking.

## Choose a Model Name

Pick a stable `model_name` string. This key connects configuration, model
registration, sampler registration, CLI choices, and benchmark configs. Keep the
name lowercase and consistent with existing names such as `llada`, `sdar`, and
`fast_dllm_v2`.

If the model should be benchmarked from the CLI, add the name to
`MODEL_NAME_CHOICES` in `diffulex_bench/arg_parser.py`.

## Model Implementation

Add model code under `diffulex/model/`. Register the model with
`AutoModelForDiffusionLM.register`. Most factories receive `config.hf_config`;
use `use_full_config=True` only when model construction needs full Diffulex
runtime settings.

The model should match the interface expected by the selected model runner and
sampler. Start from the closest existing model family and keep the first version
minimal.

## Sampler Implementation

Add a matching sampler under `diffulex/sampler/` when the model needs
family-specific token update logic. Register it with `AutoSampler.register`.

Use `sampling_mode="naive"` unless the model needs edit-style updates. Edit
sampling is currently restricted to specific LLaDA2-family model names in
`Config._validate_sampling_mode`.

## Configuration Defaults

Only add model-specific defaults when the generic engine arguments are not
enough. Examples already in the config:

| Condition | Existing config behavior |
| --- | --- |
| DiffusionGemma | Uses the native `diffusion_gemma` strategy defaults, `block_size=256`, `page_size=256`, and `buffer_size=1`. |
| DMax | Requires edit sampling and a DMax-compatible model name. |
| D2F | Disables prefix caching and uses full-prefix multi-block behavior. |

Avoid broad validation until a real invalid state has been observed.

## Benchmark and Serving Configs

After the model loads, add a small benchmark config under
`diffulex_bench/configs/` if the model is meant to be evaluated regularly. Use
paths as placeholders and keep model-specific settings explicit.

For serving, document a minimal command with low request and token limits first.
Users can scale limits after the command succeeds.

## Verification

A staged verification path keeps wiring issues easy to isolate:

1. Import the model and sampler modules.
2. Construct a `Config` with the new `model_name`.
3. Run one tiny offline generation.
4. Run a benchmark with `--dataset-limit`.
5. Add focused tests for model loading, sampler behavior, or config validation.

Do not start with a full benchmark. Full evaluations hide basic wiring problems
behind long runtime and larger GPU memory pressure.
