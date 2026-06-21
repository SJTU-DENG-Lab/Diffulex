# Add New Models

Adding a model family means registering the model implementation, adding sampler
behavior when the update rule differs, wiring the config and CLI names, and
verifying a small engine run before broad benchmark or serving work.

## Choose the Model Name

Pick one stable `model_name` string. This value is used by:

- `Config.model_name`;
- `AutoModelForDiffusionLM`;
- `AutoSampler`;
- benchmark CLI choices;
- benchmark YAML configs;
- examples and docs.

Use lowercase names with underscores, matching existing names such as
`fast_dllm_v2` and `sdar_moe`.

## Register the Model

Add the implementation under `diffulex/model/` and register it with
`AutoModelForDiffusionLM.register`.

Most model factories receive `config.hf_config`:

```python
from diffulex.model.auto_model import AutoModelForDiffusionLM


@AutoModelForDiffusionLM.register("my_model")
class MyModel:
    def __init__(self, hf_config):
        ...
```

Use `use_full_config=True` only when construction needs Diffulex-specific
runtime settings:

```python
@AutoModelForDiffusionLM.register("my_model", use_full_config=True)
class MyModel:
    def __init__(self, config):
        ...
```

## Register the Sampler

If the model needs custom token update logic, add a sampler under
`diffulex/sampler/` and register it with `AutoSampler.register`.

```python
from diffulex.sampler.auto_sampler import AutoSampler


@AutoSampler.register("my_model")
class MySampler:
    ...
```

If the existing sampler path is enough, avoid adding a new sampler. A model
family should only add custom sampler code when the update rule differs.

## Update Config Validation

Add validation in `diffulex.config.Config` only for required constraints. Good
validation catches invalid runtime states, such as unsupported strategy/mode
combinations or model-specific block/page sizes.

Do not add model-specific defaults unless the generic defaults would produce an
incorrect engine state.

## Update CLI Choices and Configs

If the model should be used through benchmarks, add the model name to
`MODEL_NAME_CHOICES` in `diffulex_bench/arg_parser.py`.

Add a benchmark YAML under `diffulex_bench/configs/` only after a small offline
generation works. Keep checkpoint paths as placeholders and keep strategy,
sampling mode, block size, and thresholds explicit.

## Verification

A practical verification sequence is:

1. Import the model module and sampler module.
2. Print available model and sampler registry keys.
3. Construct a `Config` for the new `model_name`.
4. Run one tiny `Diffulex.generate` call.
5. Run `python -m diffulex_bench.main --dataset-limit 1`.
6. Add tests for the model-specific behavior you introduced.

Do not start with full serving or full benchmark runs. They make registration
and config mistakes slower to diagnose.
