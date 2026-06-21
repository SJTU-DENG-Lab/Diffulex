# Add a Decoding Strategy

Diffulex selects strategy-specific engine components through registries. A new
strategy usually needs four registered components:

- request state
- scheduler
- KV cache manager
- model runner

Use this guide when the built-in `d2f`, `multi_bd`, and `dmax` strategies do not
match the decoding behavior you need.

## Start from an existing template

Most strategies should start from one of the templates under
`diffulex/strategy_template/`:

- `multi_block` for block-wise diffusion strategies.
- `token_merging_multi_block` for strategies that merge or rewrite filled
  tokens.
- `dual_cache` for experimental dual-cache designs.

The built-in `d2f` and `multi_bd` strategies are the smallest references for a
multi-block strategy. The `dmax` strategy is the reference for token merging.

## Directory layout

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

The package is imported automatically by `diffulex.strategy`. Importing the
package must import the component classes so their registry decorators run.

## Register request state

Register a request class with `AutoReq`:

```python
from diffulex.config import Config
from diffulex.engine.request import AutoReq
from diffulex.sampling_params import SamplingParams
from diffulex.strategy_template.multi_block.engine.request import MultiBlockReqTemplate


@AutoReq.register("my_strategy")
class MyStrategyReq(MultiBlockReqTemplate):
    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams = SamplingParams(),
        config: Config | None = None,
    ):
        super().__init__(token_ids, sampling_params)
```

Use the request class for per-request decoding state. For a normal multi-block
strategy, the template initializes block state when the scheduler adds the
request.

## Register the scheduler

Register a scheduler with `AutoScheduler`:

```python
from __future__ import annotations

from diffulex.config import Config
from diffulex.engine.request import DllmReq
from diffulex.engine.scheduler import AutoScheduler
from diffulex.strategy_template.multi_block.engine.scheduler import MultiBlockSchedulerTemplate


@AutoScheduler.register("my_strategy")
class MyStrategyScheduler(MultiBlockSchedulerTemplate):
    def __init__(self, config: Config):
        super().__init__(config)
        self.init_multi_block()

    def add(self, req: DllmReq) -> None:
        self.add_multi_block(req)

    def schedule(self) -> tuple[list[DllmReq], bool]:
        return self.schedule_multi_block()

    def preempt(self, req: DllmReq) -> None:
        self.preempt_multi_block(req)

    def postprocess(self, reqs: list[DllmReq], sample_output) -> None:
        self.postprocess_multi_block(reqs, sample_output)
```

Override only the methods whose behavior is different. Keep the template calls
when the default multi-block lifecycle is still correct.

## Register the KV cache manager

Register a KV cache manager with `AutoKVCacheManager`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from diffulex.config import Config
from diffulex.engine.kv_cache_manager import AutoKVCacheManager
from diffulex.strategy_template.multi_block.engine.kv_cache_manager import MultiBlockKVCacheManagerTemplate

if TYPE_CHECKING:
    from diffulex.engine.request import DllmReq


@AutoKVCacheManager.register("my_strategy")
class MyStrategyKVCacheManager(MultiBlockKVCacheManagerTemplate):
    def __init__(self, config: Config):
        super().__init__(config)

    def can_append(self, req: "DllmReq") -> bool:
        return self.can_append_multi_block(req)

    def may_append(self, req: "DllmReq") -> None:
        self.may_append_multi_block(req)
```

Change this class when the strategy needs different cache growth, prefix reuse,
or page append rules.

## Register attention metadata

The model runner sets the global attention metadata fetch function used by the
attention layers. A minimal multi-block metadata module looks like this:

```python
import torch

from dataclasses import dataclass

from diffulex.strategy_template.multi_block.attention.metadata import MultiBlockAttnMetaDataTemplate


@dataclass
class MyStrategyAttnMetaData(MultiBlockAttnMetaDataTemplate):
    def __post_init__(self):
        self.init_multi_block()


MY_STRATEGY_ATTN_METADATA = MyStrategyAttnMetaData()


def fetch_my_strategy_attn_metadata() -> MyStrategyAttnMetaData:
    return MY_STRATEGY_ATTN_METADATA


def set_my_strategy_attn_metadata(
    is_prefill: bool = False,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping: torch.Tensor | None = None,
    context_lens: torch.Tensor | None = None,
    page_tables: torch.Tensor | None = None,
    page_size: int = 32,
    block_size: int = 32,
    kv_cache_layout: str = "unified",
) -> None:
    global MY_STRATEGY_ATTN_METADATA
    MY_STRATEGY_ATTN_METADATA = MyStrategyAttnMetaData(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        page_tables=page_tables,
        page_size=page_size,
        block_size=block_size,
        kv_cache_layout=kv_cache_layout,
    )


def reset_my_strategy_attn_metadata() -> None:
    global MY_STRATEGY_ATTN_METADATA
    MY_STRATEGY_ATTN_METADATA = MyStrategyAttnMetaData()
```

Use a separate metadata object when the strategy changes attention layout,
prefix handling, or page table interpretation.

## Register the model runner

Register a model runner with `AutoModelRunner`:

```python
from __future__ import annotations

from multiprocessing.synchronize import Event

import torch

from diffulex.attention.metadata import set_fetch_fn_for_attn_metadata
from diffulex.config import Config
from diffulex.engine.model_runner import AutoModelRunner
from diffulex.engine.request import DllmReq
from diffulex.strategy.my_strategy.attention.metadata import (
    fetch_my_strategy_attn_metadata,
    reset_my_strategy_attn_metadata,
    set_my_strategy_attn_metadata,
)
from diffulex.strategy_template.multi_block.engine.model_runner import MultiBlockModelRunnerTemplate


@AutoModelRunner.register("my_strategy")
class MyStrategyModelRunner(MultiBlockModelRunnerTemplate):
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        set_fetch_fn_for_attn_metadata(fetch_my_strategy_attn_metadata)
        self.init_attn_metadata_fn(
            set_my_strategy_attn_metadata,
            reset_my_strategy_attn_metadata,
            fetch_my_strategy_attn_metadata,
        )
        self.mask_token_id = config.mask_token_id
        self.is_prefix_full = config.multi_block_prefix_full

        super().__init__(config, rank, event)

    def prepare_prefill(self, reqs: list[DllmReq]):
        self.prepare_chunked_prefill_multi_block(reqs)

    def prepare_decode(self, reqs: list[DllmReq]):
        self.prepare_decode_multi_block(reqs)

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor):
        self.run_model_multi_block(input_ids, positions)

    def run(self, reqs: list[DllmReq]) -> list[int]:
        return self.run_multi_block(reqs)

    @torch.inference_mode()
    def capture_cudagraph(self):
        self.capture_cudagraph_multi_block()
```

This class is the main place to change tensor preparation, attention metadata,
model execution, CUDA graph capture, or sampler interaction.

## Export the components

Import the registered classes from the strategy package `__init__.py`:

```python
from __future__ import annotations

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

After this, the strategy can be selected with:

```bash
python -m diffulex.server.launch \
  --model /YOUR-CKPT-PATH/your-model \
  --model-name your_model_name \
  --decoding-strategy my_strategy
```

or from Python:

```python
from diffulex import Diffulex

llm = Diffulex(
    model="/YOUR-CKPT-PATH/your-model",
    model_name="your_model_name",
    decoding_strategy="my_strategy",
)
```

## Update validation when needed

If the strategy is only valid for specific models or sampling modes, add the
smallest necessary validation in `diffulex.config.Config.__post_init__`.

Examples already in the config:

| Strategy | Config behavior |
| --- | --- |
| `d2f` | Forces `multi_block_prefix_full=True` and disables prefix caching. |
| `multi_bd` | Forces `multi_block_prefix_full=False`. |
| `dmax` | Forces `multi_block_prefix_full=False` and requires an edit-sampling model with `sampling_mode="edit"`. |

Avoid adding strategy-specific validation unless the engine would otherwise run
with an invalid state.

## Verification checklist

Before opening a pull request:

- Run `python -c "from diffulex import strategy; print(strategy.__all__)"` and
  confirm the package imports.
- Construct a `Config` with the new `decoding_strategy`.
- Run a tiny benchmark or server launch with a small dataset limit.
- Run the focused tests that cover the changed template or strategy code.
- Add a tutorial or user guide example if the strategy is intended for users.
