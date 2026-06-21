# Developer Troubleshooting

Use focused checks before running the full suite. Most development failures are
caused by registry imports, invalid config combinations, CUDA availability, or
shape/layout mismatches between model runner, attention metadata, and kernels.

## Imports

Confirm lightweight imports first:

```bash
python -c "import diffulex, diffulex_kernel; print('ok')"
```

Then import the package that should register your component:

```bash
python -c "from diffulex import strategy; print(strategy.__all__)"
```

If a strategy is missing, check that its directory has `__init__.py` and that
the package import does not raise.

## Strategy Registration

When a strategy is not found, confirm that each registry decorator uses the same
strategy key. Request, scheduler, KV cache manager, and model runner must all be
registered under the decoding strategy name selected by `Config`.

Also check that the strategy package imports registered classes from its
`__init__.py`. Registration happens as an import side effect.

## Model and Sampler Registration

When a model or sampler is not found, import the relevant module directly and
inspect available keys:

```python
from diffulex.model.auto_model import AutoModelForDiffusionLM
from diffulex.sampler.auto_sampler import AutoSampler

print(AutoModelForDiffusionLM.available_models())
print(AutoSampler.available_samplers())
```

The `model_name` in config must match both registrations when a custom sampler
is required.

## GPU-Only Failures

Reduce the model size, request count, token budget, and batch token budget to
separate logic errors from memory pressure.

Useful temporary changes:

| Setting | Temporary value |
| --- | --- |
| `tensor_parallel_size` | Set to `1`. |
| `data_parallel_size` | Set to `1`. |
| `max_num_reqs` | Lower it to reduce active request pressure. |
| `max_num_batched_tokens` | Lower it to reduce scheduler and memory pressure. |
| `enforce_eager` | Set to `True` while isolating correctness issues. |

Re-enable optimized paths only after the small case is correct.

## Cache and Attention Issues

Cache bugs often appear as incorrect tokens, CUDA memory errors, or shape
mismatches. Check these fields together:

| Field or component | Why it matters |
| --- | --- |
| `block_size` | Must match the decoding block layout expected by the strategy. |
| `page_size` | Must stay compatible with `block_size` and the KV cache manager. |
| `kv_cache_layout` | Must match attention metadata and kernel layout assumptions. |
| Strategy attention metadata | Defines the tensors passed into attention kernels. |
| Model runner tensor preparation | Produces the shapes and layouts consumed by kernels. |

Do not change kernel code until the metadata and layout assumptions are clear.
