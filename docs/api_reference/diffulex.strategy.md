# diffulex.strategy

`diffulex.strategy` contains the built-in decoding strategies. Importing the
package imports each strategy subpackage so its registry decorators run. The
engine then resolves the selected strategy name through the request, scheduler,
KV cache manager, model runner, sampler, and attention metadata registries.

| Strategy | Template family | Registered components |
| --- | --- | --- |
| `d2f` | Full-prefix block diffusion | `D2fReq`, `D2fScheduler`, `D2fKVCacheManager`, `D2fModelRunner`, `D2fAttnMetaData` |
| `dmax` | Token-merging multi-block diffusion | `DMaxReq`, `DMaxScheduler`, `DMaxKVCacheManager`, `DMaxModelRunner`, `DMaxAttnMetaData` |
| `multi_bd` | Multi-Block Diffusion | `MultiBDReq`, `MultiBDScheduler`, `MultiBDKVCacheManager`, `MultiBDModelRunner`, `MultiBDAttnMetaData` |
| `diffusion_gemma` | DiffusionGemma canvas/block decoding | `DiffusionGemmaReq`, `DiffusionGemmaScheduler`, `DiffusionGemmaKVCacheManager`, `DiffusionGemmaModelRunner`, `DiffusionGemmaAttnMetaData` |

The package-level helpers keep the currently selected strategy name:

| Symbol | How to use it | What it does |
| --- | --- | --- |
| `fetch_decoding_strategy` | Call from code that needs to inspect the active strategy. | Returns the current strategy name or `None`. |
| `set_decoding_strategy` | Pass a strategy name before strategy-dependent setup. | Stores the active decoding strategy globally. |
| `reset_decoding_strategy` | Call during teardown or tests. | Clears the global strategy name. |

These helpers do not register components by themselves. Registration happens
inside the strategy subpackages through `AutoReq`, `AutoScheduler`,
`AutoKVCacheManager`, and `AutoModelRunner` decorators.

## diffulex.strategy.d2f

`d2f` is the default block-diffusion strategy. It uses the multi-block template
stack and keeps prefix attention full by setting `is_prefix_full=True` in the
model runner.

| Symbol | How to use it | What it does |
| --- | --- | --- |
| `D2fReq` | Created through `AutoReq.create(..., decoding_strategy="d2f")`. | Inherits `MultiBlockReqTemplate`; strategy-specific state is initialized later by the scheduler. |
| `D2fScheduler` | Created through `AutoScheduler`. | Adds requests with `init_multi_block`, schedules prefill/decode blocks, handles preemption, and postprocesses accepted token writes. |
| `D2fKVCacheManager` | Created through `AutoKVCacheManager`. | Uses multi-block page allocation and append behavior. |
| `D2fModelRunner` | Created through `AutoModelRunner`. | Prepares chunked multi-block prefill/decode batches, runs the model, samples outputs, and captures multi-block CUDA graphs. |
| `D2fAttnMetaData` | Fetched by the attention backend. | Extends `MultiBlockAttnMetaDataTemplate` and stores per-batch multi-block attention fields. |
| `fetch_d2f_attn_metadata` | Called through the global attention metadata fetch hook. | Returns the current D2F metadata instance. |
| `set_d2f_attn_metadata` | Called by the model runner before attention. | Replaces the current D2F metadata with request/page/sequence tensors for the next forward pass. |
| `reset_d2f_attn_metadata` | Called after a forward pass or capture. | Restores an empty D2F metadata object. |

Use `d2f` as the reference when adding a strategy that follows the standard
multi-block request lifecycle without token-merging metadata.

## diffulex.strategy.dmax

`dmax` is the built-in token-merging strategy. It starts from the token-merging
multi-block template and adds DMax-specific request activation behavior,
attention metadata, and graph capture buffers for token merge descriptors.

| Symbol | How to use it | What it does |
| --- | --- | --- |
| `DMaxReq` | Created through `AutoReq.create(..., decoding_strategy="dmax")`. | Inherits `TokenMergingMultiBlockReqTemplate` and initializes token-merge state from `Config`. |
| `DMaxScheduler` | Created through `AutoScheduler`. | Uses token-merge request state, schedules multi-block work, and postprocesses token-merge sampler output. |
| `DMaxKVCacheManager` | Created through `AutoKVCacheManager`. | Uses token-merging multi-block cache behavior on top of the multi-block page allocator. |
| `DMaxModelRunner` | Created through `AutoModelRunner`. | Binds DMax attention metadata, prepares token-merging metadata tensors, and captures token-merging CUDA graphs. |
| `DMaxAttnMetaData` | Fetched by the attention backend. | Extends `TokenMergingMultiBlockAttnMetaDataTemplate` and resets token-merging fields on construction. |
| `fetch_dmax_attn_metadata` | Called through the global attention metadata fetch hook. | Returns the current DMax metadata instance. |
| `set_dmax_attn_metadata` | Called by the model runner before attention. | Replaces the current DMax metadata with request/page/sequence tensors for the next forward pass. |
| `reset_dmax_attn_metadata` | Called after a forward pass or capture. | Restores an empty DMax metadata object. |

`DMaxReq` also honors `DIFFULEX_DMAX_FORCE_PREFILL_ACTIVE=1`. That environment
switch keeps active-block iterations on the prefill-style path, which is useful
when comparing against reference DMax behavior.

## diffulex.strategy.multi_bd

`multi_bd` is the built-in Multi-Block Diffusion strategy. It shares most of
the same template surface as `d2f`, but uses block-causal prefix behavior so
prefix caching can remain enabled.

| Symbol | How to use it | What it does |
| --- | --- | --- |
| `MultiBDReq` | Created through `AutoReq.create(..., decoding_strategy="multi_bd")`. | Inherits `MultiBlockReqTemplate`; request attributes are filled by `init_multi_block`. |
| `MultiBDScheduler` | Created through `AutoScheduler`. | Adds, schedules, preempts, and postprocesses multi-block requests. |
| `MultiBDKVCacheManager` | Created through `AutoKVCacheManager`. | Uses multi-block cache allocation and append behavior. |
| `MultiBDModelRunner` | Created through `AutoModelRunner`. | Prepares multi-block batches and sets attention flags from `Config`, including DiffusionGemma-specific prefix handling. |
| `MultiBDAttnMetaData` | Fetched by the attention backend. | Extends `MultiBlockAttnMetaDataTemplate` for the MultiBD strategy. |
| `fetch_multi_bd_attn_metadata` | Called through the global attention metadata fetch hook. | Returns the current MultiBD metadata instance. |
| `set_multi_bd_attn_metadata` | Called by the model runner before attention. | Replaces the current MultiBD metadata with request/page/sequence tensors for the next forward pass. |
| `reset_multi_bd_attn_metadata` | Called after a forward pass or capture. | Restores an empty MultiBD metadata object. |

Use `multi_bd` as the closer reference when a new strategy needs block-causal
prefix behavior and prefix caching.

## diffulex.strategy.diffusion_gemma

`diffusion_gemma` is the native DiffusionGemma strategy. It uses a
DiffusionGemma-specific request, sampler, model runner, and attention metadata
path with 256-token block/page defaults.

| Symbol | How to use it | What it does |
| --- | --- | --- |
| `DiffusionGemmaReq` | Created through `AutoReq.create(..., decoding_strategy="diffusion_gemma")`. | Tracks DiffusionGemma canvas state and commit timing. |
| `DiffusionGemmaScheduler` | Created through `AutoScheduler`. | Uses the DiffusionGemma request lifecycle with standard scheduling hooks. |
| `DiffusionGemmaKVCacheManager` | Created through `AutoKVCacheManager`. | Uses the DiffusionGemma page/block layout. |
| `DiffusionGemmaModelRunner` | Created through `AutoModelRunner`. | Prepares DiffusionGemma prefill/decode tensors, self-conditioning context, and model forward calls. |
| `DiffusionGemmaAttnMetaData` | Fetched by the attention backend. | Stores DiffusionGemma attention metadata for the current forward pass. |
