# CUDA Graph

CUDA Graph support reduces launch overhead by capturing stable execution paths.
Diffulex exposes the current full-static runner controls and the standard eager
debug switch.

## Main Controls

| Key | How to set it | What it does |
| --- | --- | --- |
| `enforce_eager` | Set `True` while debugging; keep `False` for optimized runs. | Disables CUDA Graph-style execution paths. |
| `enable_full_static_runner` | Leave `True` for supported multi-block optimized paths. | Enables the full-static runner for supported forward passes. |

## Debugging Workflow

Use eager mode while validating a new model, sampler, strategy, or kernel:

```bash
python -m diffulex.server ... --enforce-eager
```

After correctness is stable, remove eager mode and compare one optimization
toggle at a time.

## Related Arguments

| Surface | Flags | Notes |
| --- | --- | --- |
| Server CLI | `--enforce-eager`, `--disable-full-static-runner` | Use eager mode for debugging and full-static runner controls for optimized paths. |
| Benchmark CLI | `--enforce-eager`, `--no-enforce-eager`, `--enable-full-static-runner` | Use full-static runner controls for current experiments. |
