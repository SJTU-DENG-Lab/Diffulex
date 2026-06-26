# diffulex.utils

`diffulex.utils` contains shared helpers that do not belong to one model
family, strategy, or serving path. The modules here are still part of the
runtime path: checkpoint loading, tokenizer construction, and output accounting
all flow through this package.

| Module | Main responsibility |
| --- | --- |
| `diffulex.utils.checkpoint` | Small dataclasses used by checkpoint weight resolution. |
| `diffulex.utils.loader` | Base-model and LoRA weight loading. |
| `diffulex.utils.output` | Generation trajectories, text conversion, and benchmark metrics. |
| `diffulex.utils.registry` | Display helpers for registry factories. |
| `diffulex.utils.tokenizer` | Robust Hugging Face tokenizer construction. |

## diffulex.utils.checkpoint

`checkpoint` defines the value objects used by model and layer code when a
checkpoint tensor needs custom loading behavior.

| Symbol | How to use it | What it does |
| --- | --- | --- |
| `LoadContext` | Receive it inside a module's `resolve_checkpoint_weight` hook. | Carries the engine `Config` and the full checkpoint tensor name being resolved. |
| `ResolvedWeight` | Return it from `resolve_checkpoint_weight`. | Describes where a tensor should go: a parameter, a buffer, a custom loader, a transform, a shard id, or a skip marker. |

Use `ResolvedWeight` when checkpoint names do not map cleanly to PyTorch
parameter names. It keeps family-specific mapping logic near the model or layer
that owns the weight, while `loader` handles the actual copy.

## diffulex.utils.loader

`loader` is responsible for reading `.safetensors` checkpoints, applying custom
weight resolvers, handling packed module mappings, and optionally loading LoRA
adapters.

| Symbol | How to use it | What it does |
| --- | --- | --- |
| `load_lora_config` | Pass a LoRA adapter directory. | Reads `adapter_config.json` when present; otherwise returns an empty dict. |
| `enable_lora_for_model` | Pass a model and LoRA config before loading weights. | Calls `__init_lora__` on matching modules so LoRA tensors exist. |
| `default_weight_loader` | Use as the fallback parameter loader. | Copies the loaded tensor into `param.data`. |
| `resolve_weight_spec` | Pass model, checkpoint tensor name, and config. | Walks modules from most specific prefix to root and asks `resolve_checkpoint_weight` hooks for a `ResolvedWeight`. |
| `apply_resolved_weight` | Pass a `ResolvedWeight` and loaded tensor. | Applies transforms, custom loaders, parameter shards, buffers, or skip behavior. |
| `try_load_direct` | Pass model, tensor name, and loaded tensor. | Attempts direct parameter or buffer loading by exact name. |
| `try_load_via_packed_mapping` | Pass model, packed mapping, tensor name, loaded tensor, and config. | Handles packed projections such as merged QKV or model-family-specific aliases. |
| `load_model` | Pass an initialized model and `Config`. | Loads base `.safetensors` files, enables LoRA when requested, then loads LoRA weights. |
| `load_lora_weights` | Pass a LoRA-enabled model and adapter path. | Finds LoRA A/B tensors, handles TP sharding for supported layers, and optionally pre-merges adapters. |

The load order is deliberate. Custom resolvers get the first chance to map a
checkpoint tensor, packed-module mappings run second, and exact parameter/buffer
names are tried last. This makes unusual model-family layouts explicit without
breaking the simple case.

## diffulex.utils.output

`output` stores generation results and computes the metrics shown by offline
inference and benchmarks. It keeps both token-level trajectories and aggregate
throughput counters.

| Symbol | How to use it | What it does |
| --- | --- | --- |
| `decode_token_ids_robust` | Pass a tokenizer and token IDs. | Decodes normally first, then falls back to token conversion for tokenizers with stricter decode signatures. |
| `ReqStep` | Created for each scheduled engine step. | Records step time, prefill/decode mode, generated token count, running tokens, buffer block IDs, and optional block trace. |
| `ReqTrajectory` | One item per prompt. | Stores final token IDs, full response token IDs, truncation flags, completion reason, text, and per-step trajectory. |
| `GenerationOutputs` | Created by the engine for a batch. | Accumulates trajectories and exposes metrics such as TPF, TTFT, TPOT, throughput, prefill throughput, and decode throughput. |
| `GenerationOutputs.record_step` | Call after each engine step. | Updates batch counters and appends `ReqStep` records for each active request. |
| `GenerationOutputs.convert_to_text` | Pass the tokenizer after generation finishes. | Decodes truncated and full token responses into text. |
| `GenerationOutputs.to_benchmark_format` | Call before returning benchmark-compatible data. | Produces `{text, full_text, token_ids, nfe}` dictionaries. |

Set `DIFFULEX_SAVE_TRACE=0` when block-level traces are not needed. Leaving it
enabled records per-block status and mask ratios, which is useful for debugging
decoding behavior but adds more data to each trajectory.

## diffulex.utils.registry

`registry` contains small helpers used by the registry classes for readable
errors and diagnostics.

| Symbol | How to use it | What it does |
| --- | --- | --- |
| `fetch_factory_name` | Pass a class, function, `functools.partial`, or callable object. | Returns a stable module-qualified display name after unwrapping decorators and partials. |

Use this helper when a registry needs to describe which factory is currently
bound without assuming the factory is a plain class.

## diffulex.utils.tokenizer

`tokenizer` wraps Hugging Face `AutoTokenizer.from_pretrained` with a fallback
for tokenizer configs that store `extra_special_tokens` as a list. Some
tokenizer versions expect a dict, so Diffulex coerces the list into stable
generated token names before retrying.

| Symbol | How to use it | What it does |
| --- | --- | --- |
| `auto_tokenizer_from_pretrained` | Use instead of calling `AutoTokenizer.from_pretrained` directly in Diffulex code. | Loads the tokenizer, and if necessary retries with coerced `extra_special_tokens`. |

The fallback only runs for the known `extra_special_tokens` shape error. Other
tokenizer loading failures are re-raised so configuration problems remain
visible.
