# diffulex.strategy_template

`diffulex.strategy_template` contains reusable strategy building blocks. A real
strategy package usually subclasses or directly reuses one of these templates,
then registers concrete request, scheduler, KV cache manager, model runner, and
attention metadata classes under a strategy name.

The templates are intentionally organized by decoding layout rather than by
public feature:

| Template package | Use it when | Core classes |
| --- | --- | --- |
| `dual_cache` | A strategy needs the multi-block lifecycle but wants a separate cache-layout variant. | `DualCacheReqTemplate`, `DualCacheSchedulerTemplate`, `DualCacheKVCacheManagerTemplate`, `DualCacheModelRunnerTemplate` |
| `multi_block` | A strategy decodes with a rolling buffer of fixed-size diffusion blocks. | `MultiBlockReqTemplate`, `MultiBlockSchedulerTemplate`, `MultiBlockKVCacheManagerTemplate`, `MultiBlockModelRunnerTemplate`, `MultiBlockAttnMetaDataTemplate`, `FullStaticRunner` |
| `token_merging_multi_block` | A strategy is multi-block and also passes token-merge distributions into attention. | `TokenMergeDescriptor`, `TokenMergingMultiBlockReqTemplate`, `TokenMergingMultiBlockSchedulerTemplate`, `TokenMergingMultiBlockKVCacheManagerTemplate`, `TokenMergingMultiBlockModelRunnerTemplate`, `TokenMergingMultiBlockAttnMetaDataTemplate` |

For extension work, start from the smallest template that already matches the
decoding lifecycle. Most new block-diffusion strategies should begin with
`multi_block`; only use `token_merging_multi_block` when attention truly needs
per-token merge descriptors.

## diffulex.strategy_template.dual_cache

`dual_cache` is a thin template family reserved for strategies that follow the
multi-block lifecycle but need a dual-cache implementation. At the moment it
inherits the multi-block behavior directly; concrete strategies can override
only the pieces that differ.

| Symbol | How to use it | What it does |
| --- | --- | --- |
| `DualCacheReqTemplate` | Subclass it when request state will diverge from normal multi-block request state. | Currently inherits `MultiBlockReqTemplate`. |
| `DualCacheSchedulerTemplate` | Subclass it when admission, preemption, or postprocessing needs dual-cache rules. | Currently inherits `MultiBlockSchedulerTemplate`. |
| `DualCacheKVCacheManagerTemplate` | Subclass it when page allocation or append behavior differs. | Currently inherits `MultiBlockKVCacheManagerTemplate`. |
| `DualCacheModelRunnerTemplate` | Subclass it when batch preparation or model execution differs. | Currently inherits `MultiBlockModelRunnerTemplate`. |

Because these classes are mostly pass-through today, treat them as named
extension points rather than runtime behavior changes.

## diffulex.strategy_template.multi_block

`multi_block` is the core template for block diffusion. It represents each
request as a chain of `DllmBlock` objects, keeps a rolling
`DllmBlockBuffer`, schedules prefill and decode work against the KV cache, and
prepares attention metadata for paged attention.

| Symbol | How to use it | What it does |
| --- | --- | --- |
| `MultiBlockReqTemplate` | Inherit from it in a concrete request class. | Initializes block state, pads prefix tokens to block boundaries, tracks generation limits, advances block status, and exposes running sequences for the runner. |
| `MultiBlockSchedulerTemplate` | Inherit from it in a concrete scheduler. | Admits waiting requests, allocates/extends KV pages, preempts requests when cache space is tight, and applies sampled token writes back into blocks. |
| `MultiBlockKVCacheManagerTemplate` | Inherit from it in a concrete KV cache manager. | Calculates missing cache pages, finalizes unhashed pages, and appends pages for newly cacheable blocks. |
| `MultiBlockModelRunnerTemplate` | Inherit from it in a concrete model runner. | Prepares prefill/decode tensors, sets attention metadata, runs model forward and sampling, supports idle ranks, and captures CUDA graphs. |
| `MultiBlockAttnMetaDataTemplate` | Extend it for strategy-specific attention metadata. | Stores valid slices, buffer size, prefix/full-prefix flags, status tables, and prefix-hole controls used by attention kernels. |
| `FullStaticRunner` | Used by `MultiBlockModelRunnerTemplate`. | Runs selected prefill/decode shapes through static buffers and CUDA graphs when the configured shape is supported. |

The request template is where most lifecycle semantics live. It decides when a
request is waiting, prefilling, decoding, completed, or preempted; it also owns
the truncation rules for EOS, `max_tokens`, `max_model_len`, `max_nfe`, and
`max_repetition_run`.

The scheduler template is intentionally narrow: it chooses which requests run
this step, asks the cache manager whether pages are available, and applies the
sampler's accepted writes. It leaves strategy-specific sampling rules to the
sampler package.

## diffulex.strategy_template.token_merging_multi_block

`token_merging_multi_block` extends the multi-block template with per-position
token-merge metadata. It is the template family used by `dmax`.

| Symbol | How to use it | What it does |
| --- | --- | --- |
| `TokenMergeDescriptor` | Store top-k token IDs, top-k probabilities, and residual probability for one absolute position. | Validates that top-k IDs are present and aligned with top-k probabilities. |
| `TokenMergingMultiBlockReqTemplate` | Inherit from it when a strategy records merge descriptors per running position. | Initializes token-merge config, stores descriptors, clears stale descriptors, and looks up descriptors during runner preparation. |
| `TokenMergingMultiBlockSchedulerTemplate` | Inherit from it in the concrete scheduler. | Reuses multi-block scheduling and adds token-merging postprocessing hooks. |
| `TokenMergingMultiBlockKVCacheManagerTemplate` | Inherit from it in the concrete cache manager. | Reuses multi-block cache behavior for token-merging strategies. |
| `TokenMergingMultiBlockModelRunnerTemplate` | Inherit from it in the concrete runner. | Builds token-merge tensors, binds them into attention metadata, handles CUDA graph replay metadata, and runs token-merging multi-block batches. |
| `TokenMergingMultiBlockAttnMetaDataTemplate` | Extend it for concrete token-merging metadata. | Stores merge masks, top-k IDs, top-k probabilities, residual probabilities, merge mode, weight, and renormalization settings. |

Token merging adds metadata but keeps the same request/block scheduling model.
That separation is important: a strategy can change how attention sees uncertain
tokens without rewriting the core multi-block scheduler.
