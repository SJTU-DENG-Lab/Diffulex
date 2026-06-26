# Profiling

Use profiling when correctness is already established and you need to understand
where time or memory is spent. Keep profiling runs narrow; changing many runtime
settings at once makes results difficult to interpret.

## Pytorch Profiler

Use PyTorch Profiler when you need CPU and CUDA timing for a focused inference
path. Diffulex records named regions around major engine operations such as
scheduler work, request preparation, model runner execution, and output
recording.

Keep profiling runs small:

- use a short prompt set;
- limit generated tokens;
- record one model and strategy configuration at a time;
- save traces outside the source tree when they are large.

## What to Measure

Choose the metric before profiling:

- end-to-end latency for one request;
- throughput for a fixed prompt set;
- scheduler overhead;
- model runner execution time;
- prefill cost versus decode cost;
- kernel time for attention, top-k, or MoE operations.

Use the smallest workload that still exhibits the behavior being measured.

## Runtime Toggles

Compare optimized and debug paths deliberately:

| Setting | What it compares |
| --- | --- |
| `enforce_eager=True` | Debug-friendly eager execution against optimized paths. |
| CUDA Graph paths | Launch-overhead reduction against eager execution. |
| `enable_torch_compile` | Supported compiled execution against uncompiled execution. |
| `enable_vllm_layers` | Optional vLLM-backed layers against local layer implementations. |

Run a baseline before changing any toggle. Keep model, prompts, token limits,
and parallelism fixed across comparisons.

## Existing Scripts

Diffulex includes profiling and benchmark scripts under `script/`. Use these
when they match the target workload because they capture local conventions for
model paths, task names, and profiler output.

If you create a new profiling script, keep large trace files out of the source
tree and document the exact command used to generate a result.
