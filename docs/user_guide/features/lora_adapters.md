# LoRA Adapters

LoRA settings load adapter checkpoints for supported model families. They are
used by both offline inference and benchmark/server entry points.

## Arguments

| Key | How to set it | What it does |
| --- | --- | --- |
| `use_lora` | Set `True` when an adapter should be loaded. The default is `False`. | Enables LoRA adapter loading. |
| `lora_path` | Point to the adapter checkpoint directory. Required when `use_lora=True`. | Provides the adapter weights. |
| `pre_merge_lora` | Set `True` when the adapter should be merged into the base model at load time. Core config defaults to `False`; benchmark config defaults to `True`. | Avoids per-forward adapter compute when merging is supported. |

| Surface | Flags | Notes |
| --- | --- | --- |
| Server CLI | `--use-lora`, `--lora-path`, `--pre-merge-lora` | Use these when serving with an adapter. |
| Benchmark CLI | `--use-lora`, `--lora-path`, `--pre-merge-lora` | Use these when evaluating an adapter-backed model. |

## Pre Merge Lora

Pre-merge avoids per-forward adapter compute when the adapter and base model can
be safely merged. Use it for normal inference after confirming the LoRA path
matches the base model. Disable it while debugging adapter loading issues.

## Validation

If `use_lora=True` and no `lora_path` is provided, config construction fails.
If the path does not exist, Diffulex logs a warning.
