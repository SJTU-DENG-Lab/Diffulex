# The Design

This section describes the major internal boundaries in Diffulex. The design is
registry-driven: a decoding strategy selects request state, scheduling, KV cache
management, model execution, and attention metadata without changing the public
engine API.

## Engine Architecture

The public `Diffulex` symbol constructs `DiffulexEngine`. Engine startup follows
this sequence:

1. Build and validate `diffulex.config.Config`.
2. Compute the requested parallel world size.
3. Spawn model runner workers for nonzero ranks.
4. Load the tokenizer and synchronize tokenizer-derived fields.
5. Construct the rank-0 model runner.
6. Construct the strategy-specific scheduler.

The engine owns request submission, stepping, output recording, worker cleanup,
and profiling lifecycle. It delegates strategy-specific behavior to registered
components.

## Request Flow

`add_request` tokenizes string prompts, creates a request object through
`AutoReq`, assigns page size metadata, and adds the request to the scheduler.

`step` runs one scheduler/model/sampler iteration:

1. the scheduler returns executable requests and whether the step is prefill;
2. requests are prepared for execution;
3. the model runner executes and returns sample output;
4. the scheduler postprocesses request state;
5. finished request IDs are evicted from sampler state.

`generate` repeats this loop until the scheduler reports completion.

## Scheduler

The scheduler decides which requests can prefill, decode, append blocks,
preempt, abort, or finish during each engine step. Strategy templates provide
common lifecycle operations, while concrete strategies define the exact policy.

Data parallel scheduling is handled separately from model parallel execution.
The scheduler must respect memory and token budgets such as `max_num_reqs`,
`max_num_batched_tokens`, and `max_model_len`.

## KV Cache and Paged Attention

KV cache managers track page allocation, append rules, prefix reuse, and layout
metadata consumed by attention kernels. Diffulex supports `unified` and
`distinct` KV cache layouts; the layout must match attention metadata and kernel
expectations.

Paged attention lets the scheduler manage cache blocks without requiring every
request to occupy one contiguous memory region.

## Model Runner

Model runners prepare tensors, set attention metadata, execute model forward
passes, invoke samplers, and optionally capture CUDA graphs. Strategy-specific
model runners are the main boundary for changing tensor layout or attention
semantics.

## Registries

Diffulex uses registries for extensibility:

- `AutoReq`
- `AutoScheduler`
- `AutoKVCacheManager`
- `AutoModelRunner`
- `AutoModelForDiffusionLM`
- `AutoSampler`

Importing strategy, model, or sampler modules triggers decorators that populate
these registries.

## Kernels

Kernel modules provide optimized attention, KV cache, top-k, and MoE operations
used by the engine and model layers. Kernel changes should be covered by focused
numerical tests before they are wired into a strategy path.
