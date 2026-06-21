# CUDA Graph

CUDA Graph support reduces launch overhead by capturing stable execution paths.
Diffulex exposes toggles for full static runner paths and torch compile
interaction. Historical prefill CUDA Graph flags remain in the config and CLI
for compatibility, but the current benchmark CLI marks that path as a no-op.

## Main Controls

| Key | How to set it | What it does |
| --- | --- | --- |
| `enforce_eager` | Set `True` while debugging; keep `False` for optimized runs. | Disables CUDA Graph-style execution paths. |
| `enable_prefill_cudagraph` | Keep unset unless testing legacy behavior. | Deprecated compatibility toggle in current benchmark flows. |
| `enable_full_static_runner` | Leave `True` for supported multi-block optimized paths. | Enables the full-static runner for supported forward passes. |
| `prefill_cudagraph_max_len` | Use `0` to follow `max_model_len`, or set a non-negative bucket length. | Caps the maximum prefill bucket length captured. |
| `enable_cudagraph_torch_compile` | Keep `False` unless testing the experimental combined path. | Allows torch compile inside decode graph capture. |

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
| Server CLI | `--enforce-eager`, `--disable-prefill-cudagraph`, `--disable-full-static-runner`, `--prefill-cudagraph-max-len`, `--enable-cudagraph-torch-compile` | `--disable-prefill-cudagraph` is a deprecated no-op; the other flags affect current optimized paths. |
| Benchmark CLI | `--enforce-eager`, `--no-enforce-eager`, `--enable-prefill-cudagraph`, `--enable-full-static-runner`, `--prefill-cudagraph-max-len`, `--enable-cudagraph-torch-compile` | `--enable-prefill-cudagraph` is a deprecated no-op; use full-static runner and compile toggles for current experiments. |
