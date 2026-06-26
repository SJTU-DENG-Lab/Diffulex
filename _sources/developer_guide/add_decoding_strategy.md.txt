# Add a Decoding Strategy

Diffulex selects strategy-specific engine components through registries. A new
strategy usually registers these pieces under one `decoding_strategy` name:

- request state;
- scheduler;
- KV cache manager;
- model runner;
- attention metadata.

Use this guide when the built-in `d2f`, `multi_bd`, `dmax`, and
`diffusion_gemma` strategies do not match the decoding behavior you need.

## Start from Current Templates

Current reusable pieces live in these places:

| Area | Path |
| --- | --- |
| Core multi-block request/scheduler/cache/runner aliases | `diffulex.engine.request`, `diffulex.engine.scheduler`, `diffulex.engine.kv_cache_manager`, `diffulex.engine.model_runner` |
| Multi-block runner helpers and attention metadata mixin | `diffulex.mixin.multi_block` |
| Token-merge templates | `diffulex.strategy.templates.token_merge` |
| Dual-cache extension points | `diffulex.strategy.templates.dual_cache` |

Use `multi_bd` or `d2f` as the smallest references for normal multi-block
strategies. Use `dmax` as the reference when attention needs token-merge
metadata. Use `diffusion_gemma` as the reference for a model-specific strategy
with custom request and sampler semantics.

## Directory Layout

Create a package under `diffulex/strategy/<strategy_name>/`:

```text
diffulex/strategy/my_strategy/
  __init__.py
  attention/
    metadata.py
  engine/
    kv_cache_manager.py
    model_runner.py
    request.py
    scheduler.py
```

Import the package from `diffulex/strategy/__init__.py` so registry decorators
run during engine startup.

## Request

Register a request class with `AutoReq`:

```python
from diffulex.config import Config
from diffulex.engine.request import AutoReq, MultiBlockReqTemplate
from diffulex.sampling_params import SamplingParams


@AutoReq.register("my_strategy")
class MyStrategyReq(MultiBlockReqTemplate):
    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams = SamplingParams(),
        config: Config | None = None,
    ):
        super().__init__(token_ids, sampling_params, config)
```

For normal multi-block behavior, the base request is enough. Add fields only
when the strategy needs additional per-request state.

## Scheduler

Register a scheduler with `AutoScheduler`:

```python
from diffulex.config import Config
from diffulex.engine.scheduler import AutoScheduler, MultiBlockSchedulerTemplate


@AutoScheduler.register("my_strategy")
class MyStrategyScheduler(MultiBlockSchedulerTemplate):
    def __init__(self, config: Config):
        super().__init__(config)
```

`MultiBlockSchedulerTemplate` is a compatibility alias for the core block-aware
scheduler. Override `add`, `schedule`, `preempt`, or `postprocess` only when the
default lifecycle is not correct for the new strategy.

## KV Cache Manager

Register a KV cache manager with `AutoKVCacheManager`:

```python
from diffulex.config import Config
from diffulex.engine.kv_cache_manager import AutoKVCacheManager, MultiBlockKVCacheManagerTemplate


@AutoKVCacheManager.register("my_strategy")
class MyStrategyKVCacheManager(MultiBlockKVCacheManagerTemplate):
    def __init__(self, config: Config):
        super().__init__(config)
```

Override this class when cache growth, prefix reuse, page hashing, or append
rules differ from the standard multi-block manager.

## Attention Metadata

The model runner sets the global attention metadata fetch function used by the
attention layers. A minimal metadata module can extend the multi-block metadata
mixin:

```python
from dataclasses import dataclass

from diffulex.attention.metadata import AttnMetaDataBase
from diffulex.mixin.multi_block.attention_metadata import MultiBlockAttnMetaDataMixin


@dataclass
class MyStrategyAttnMetaData(MultiBlockAttnMetaDataMixin, AttnMetaDataBase):
    def __post_init__(self):
        self.init_multi_block()


MY_STRATEGY_ATTN_METADATA = MyStrategyAttnMetaData()


def fetch_my_strategy_attn_metadata() -> MyStrategyAttnMetaData:
    return MY_STRATEGY_ATTN_METADATA


def set_my_strategy_attn_metadata(**kwargs) -> None:
    global MY_STRATEGY_ATTN_METADATA
    MY_STRATEGY_ATTN_METADATA = MyStrategyAttnMetaData(**kwargs)


def reset_my_strategy_attn_metadata() -> None:
    global MY_STRATEGY_ATTN_METADATA
    MY_STRATEGY_ATTN_METADATA = MyStrategyAttnMetaData()
```

Use a separate metadata object when the strategy changes attention layout,
prefix handling, or page table interpretation.

## Model Runner

Register a model runner with `AutoModelRunner`:

```python
from multiprocessing.synchronize import Event

from diffulex.attention.metadata import set_fetch_fn_for_attn_metadata
from diffulex.config import Config
from diffulex.engine.model_runner import AutoModelRunner, MultiBlockModelRunnerTemplate
from diffulex.strategy.my_strategy.attention.metadata import (
    fetch_my_strategy_attn_metadata,
    reset_my_strategy_attn_metadata,
    set_my_strategy_attn_metadata,
)


@AutoModelRunner.register("my_strategy")
class MyStrategyModelRunner(MultiBlockModelRunnerTemplate):
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        set_fetch_fn_for_attn_metadata(fetch_my_strategy_attn_metadata)
        self.init_attn_metadata_fn(
            set_my_strategy_attn_metadata,
            reset_my_strategy_attn_metadata,
            fetch_my_strategy_attn_metadata,
        )
        super().__init__(config, rank, event)
```

The base runner already provides normal multi-block prepare, run, and graph
capture behavior. Override runner methods only when the strategy needs custom
tensor preparation, model execution, sampler interaction, or graph capture.

## Export and Import

Export the registered classes from the strategy package `__init__.py`:

```python
from .engine.kv_cache_manager import MyStrategyKVCacheManager
from .engine.model_runner import MyStrategyModelRunner
from .engine.request import MyStrategyReq
from .engine.scheduler import MyStrategyScheduler

__all__ = [
    "MyStrategyKVCacheManager",
    "MyStrategyModelRunner",
    "MyStrategyReq",
    "MyStrategyScheduler",
]
```

Then import the package from `diffulex/strategy/__init__.py`.

## Config Validation

Add validation in `diffulex.config.Config` only when the strategy would
otherwise run in an invalid state.

Examples already in the config:

| Strategy | Config behavior |
| --- | --- |
| `d2f` | Forces `multi_block_prefix_full=True` and disables prefix caching. |
| `multi_bd` | Forces `multi_block_prefix_full=False`. |
| `dmax` | Forces `multi_block_prefix_full=False` and requires `sampling_mode="edit"`. |
| `diffusion_gemma` | Uses DiffusionGemma-specific block/page/sampler defaults. |

## Verification Checklist

Before opening a pull request:

- Import `diffulex.strategy` and confirm the new package imports.
- Construct a `Config` with the new `decoding_strategy`.
- Run one tiny in-process generation.
- Run a benchmark with `--dataset-limit`.
- Add focused tests for request state, scheduler behavior, sampler output, or
  attention metadata.
