# diffulex.engine

`diffulex.engine` contains the core inference lifecycle: request creation,
scheduling, KV cache management, model-runner execution, sampling, output
recording, and worker cleanup. Decoding strategies plug into this package
through registries rather than by changing the public engine API.

| Module | Role |
| --- | --- |
| `diffulex.engine.dllm_block` | Block and block-buffer state used by diffusion decoding requests. |
| `diffulex.engine.engine` | Main `DiffulexEngine` implementation and worker process entry point. |
| `diffulex.engine.kv_cache_manager` | Base KV cache manager contract, page objects, and registry. |
| `diffulex.engine.model_runner` | Base model runner contract and registry. |
| `diffulex.engine.request` | Base request state and request registry. |
| `diffulex.engine.scheduler` | Scheduler base classes, data-parallel wrapper, and scheduler registry. |
| `diffulex.engine.status` | Request, block, and block-type enums. |
| `diffulex.engine.strategy_registry` | Shared strategy registry implementation used by request, scheduler, cache, and runner registries. |

## diffulex.engine.dllm_block

This module models diffusion decoding at block granularity. It tracks which
tokens are masked, filled, semi-complete, or complete and stores per-block state
needed by multi-block and edit-style strategies.

| Symbol | Purpose |
| --- | --- |
| `DllmBlock` | Per-block token state, masks, counters, and status transitions. |
| `DllmBlockBuffer` | Active block buffer used by multi-block request state. |

Strategy templates build on these classes instead of duplicating block-state
bookkeeping.

## diffulex.engine.engine

This module contains `DiffulexEngine`, the in-process engine behind the public
`diffulex.Diffulex` alias. It validates configuration, loads tokenizer metadata,
spawns nonzero-rank model runners, constructs strategy components, submits
requests, steps the scheduler/model/sampler loop, and records outputs.

| Symbol | Purpose |
| --- | --- |
| `DiffulexEngine` | Main engine implementation for offline generation and lower-level step APIs. |
| `_run_model_runner_worker` | Worker process entry point for nonzero ranks. |

Most users should import `Diffulex` from the package root. Engine internals are
useful when extending scheduling, worker lifecycle, or profiling behavior.

## diffulex.engine.kv_cache_manager

This module defines how strategies allocate, append, release, and reuse KV cache
pages. Concrete strategies register their cache manager implementations under a
decoding-strategy key.

| Symbol | Purpose |
| --- | --- |
| `Page` | KV cache page descriptor. |
| `KVCacheManagerBase` | Abstract cache manager contract used by schedulers. |
| `AutoKVCacheManager` | Strategy registry for concrete cache managers. |

Cache manager changes should be paired with attention metadata checks because
page allocation and kernel layout must agree.

## diffulex.engine.model_runner

Model runners prepare tensors, initialize attention metadata, execute model
forward passes, call samplers, and optionally capture CUDA graph paths. The base
runner owns common model/sampler construction and worker lifecycle behavior.

| Symbol | Purpose |
| --- | --- |
| `ModelRunnerBase` | Common runner functionality for model loading, sampler loading, execution setup, and worker control. |
| `AutoModelRunner` | Strategy registry for concrete model runners. |

New strategies usually subclass a strategy template model runner instead of
subclassing `ModelRunnerBase` directly.

## diffulex.engine.request

This module provides the base request object and request registry. A request
tracks prompt tokens, generated tokens, sampling parameters, output state, and
strategy-specific lifecycle fields supplied by mixins or templates.

| Symbol | Purpose |
| --- | --- |
| `DllmReq` | Base request state used by scheduler and model runner code. |
| `AutoReq` | Strategy registry for concrete request classes. |

Strategy-specific request classes should keep only request-local decoding state
here; scheduler policy belongs in scheduler classes.

## diffulex.engine.scheduler

Schedulers decide which requests can prefill, decode, append blocks, finish, or
abort during each engine step. The data-parallel wrapper coordinates multiple
request-processing groups.

| Symbol | Purpose |
| --- | --- |
| `SchedulerBase` | Abstract scheduler contract used by `DiffulexEngine`. |
| `DataParallelScheduler` | Wrapper for data-parallel scheduling. |
| `AutoScheduler` | Strategy registry for concrete schedulers. |

Scheduler changes should preserve the contract between request state, cache
manager decisions, and model runner tensor preparation.

## diffulex.engine.status

This module contains status enums shared by requests and block buffers.

| Symbol | Purpose |
| --- | --- |
| `DllmBlockType` | Identifies block categories. |
| `DllmBlockStatus` | Tracks block lifecycle state. |
| `DllmReqStatus` | Tracks request lifecycle state. |

Use these enums instead of ad hoc strings in request or scheduler code.

## diffulex.engine.strategy_registry

This module implements the registry pattern used by `AutoReq`,
`AutoScheduler`, `AutoKVCacheManager`, and `AutoModelRunner`.

| Symbol | Purpose |
| --- | --- |
| `DiffulexStrategyRegistry` | Base class for strategy-keyed registries with aliases, defaults, and factory lookup. |

When adding a strategy, every registered component should use the same decoding
strategy key so `Config.decoding_strategy` resolves a coherent component set.
