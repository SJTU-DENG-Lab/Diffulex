# Research Engine

Diffulex `main` is the active engine branch for researchers who want to turn a
diffusion language model idea into a runnable, profiled, and serveable system.
Use the Diffulex `mbd-lms` branch when reproducing the reported MBD-LMs
experiments; use `main` when building new decoding algorithms, cache behavior,
model support, kernels, or serving features.

## Why This Backend Fits dLLM Research

Most block-level dLLM inference algorithms can be expressed as a small set of
runtime decisions:

- what block state each request owns;
- which blocks are active in the current running set;
- when a block can be appended, committed, rewritten, or evicted;
- how the active block view maps to prefix KV cache and paged attention;
- how logits are converted back into masked-token updates.

Diffulex keeps these concerns separated. This makes it practical to implement
new algorithms on top of the same serving backend instead of rebuilding
scheduler, cache, attention, CUDA graph, benchmark, and HTTP serving paths for
each idea.

The engine is also friendly to code-agent assisted research. A coding agent such
as Codex or Claude Code can usually make a bounded patch when it is given the
strategy reference to copy from and the file map below. The important point is
that the algorithm semantics live in registered strategy components, while the
systems machinery remains reusable.

## Block Buffer Backend

Diffulex treats Block Buffer execution as a three-level backend:

| Level | What it owns | Main files |
| --- | --- | --- |
| Logical block state | Prompt blocks, noisy blocks, dummy slots, commit readiness, block progress, and request-local generation state. | `diffulex/engine/dllm_block.py`, `diffulex/engine/request.py`, `diffulex/strategy/<name>/engine/request.py` |
| Running-set policy | Which blocks are active, when new blocks enter the buffer, when completed blocks are committed, and when requests prefill, decode, preempt, or finish. | `diffulex/engine/scheduler.py`, `diffulex/engine/kv_cache_manager.py`, `diffulex/strategy/<name>/engine/{scheduler,kv_cache_manager}.py` |
| Paged KV and kernel view | Prefix reuse, page tables, cache append rules, attention metadata, and dLLM-oriented Triton kernels. | `diffulex/attention/`, `diffulex/mixin/multi_block/`, `diffulex_kernel/python/`, `diffulex/strategy/<name>/attention/metadata.py`, `diffulex/strategy/<name>/engine/model_runner.py` |

This layering is why SingleBD, MultiBD, TokenMerge/DMax, edit refinement,
DiffusionGemma-style uniform diffusion, and future DualCache-style designs can
share one backend. New algorithms usually change the running-set and sampling
semantics, not the whole engine.

## Implementation Map

Use an existing strategy as the reference before adding a new one:

| Reference | Use it when |
| --- | --- |
| `diffulex/strategy/multi_bd` | The algorithm is a standard block-causal running-set decoder. |
| `diffulex/strategy/d2f` | The algorithm is closest to native SingleBD or prefix-full block decoding. |
| `diffulex/strategy/dmax` and `diffulex/strategy/templates/token_merge` | The algorithm changes token acceptance, merge metadata, or edit-style sampling. |
| `diffulex/strategy/diffusion_gemma` | The algorithm is model-specific and has non-standard canvas, sampler, or block semantics. |
| `diffulex/strategy/templates/dual_cache` | The algorithm changes cache ownership or needs a DualCache-style design. |

A strategy-level algorithm typically covers these files:

| File | Required when | What to implement |
| --- | --- | --- |
| `diffulex/strategy/<name>/__init__.py` | Always. | Export registered request, scheduler, cache manager, and model runner classes. Strategy packages are auto-imported from `diffulex/strategy/__init__.py`. |
| `diffulex/strategy/<name>/config.py` | The strategy needs normalized defaults. | Register a `StrategyConfigRegistry` normalizer and force only the invariants required by the algorithm. |
| `diffulex/strategy/<name>/engine/request.py` | Always. | Register `AutoReq`; add per-request state such as block progress, edit windows, token-merge metadata, or custom finish conditions. |
| `diffulex/strategy/<name>/engine/scheduler.py` | Always. | Register `AutoScheduler`; define add-block, commit, prefill/decode, preemption, and finish policy. |
| `diffulex/strategy/<name>/engine/kv_cache_manager.py` | Always for cache-aware strategies. | Register `AutoKVCacheManager`; define page allocation, append, prefix reuse, and cache-commit behavior. |
| `diffulex/strategy/<name>/attention/metadata.py` | The attention mask or page interpretation differs from an existing strategy. | Define metadata consumed by `diffulex.attention.Attention`; start from `MultiBlockAttnMetaDataMixin` when possible. |
| `diffulex/strategy/<name>/engine/model_runner.py` | Always. | Register `AutoModelRunner`; prepare tensors, set attention metadata, call the model, invoke the sampler, and connect CUDA graph/full-static runner paths. |
| `diffulex/sampler/<model_or_strategy>.py` | Sampling semantics change. | Implement mask-to-token updates, edit updates, token merge, confidence thresholds, or model-specific output postprocessing. |
| `diffulex/mixin/<feature>/` | The behavior should be reused across strategies. | Put shared scheduler, sampler, request, or runner helpers here instead of duplicating strategy code. |
| `diffulex_kernel/python/*.py` | The algorithm needs a new fused operation. | Add a Triton or Python reference path for attention, KV cache, sampler, top-k, layernorm, or other dLLM-specific kernels. |
| `diffulex/config.py` and CLI/benchmark config files | A new public option is genuinely needed. | Add validation and user-facing flags only for options that users should tune. Keep compatibility-only fields out of docs and help text. |

As a planning rule, a strategy that reuses Block Buffer, grouped attention, and
an existing sampler is often a small patch across 6 to 8 files. A strategy with
new sampler semantics usually adds one or two sampler/mixin files. A strategy
with new kernel semantics needs extra reference code, Triton code, and profiling
work. The goal is to spend code on the new algorithm, not on rebuilding serving
infrastructure.

## Code-Agent Workflow

For agent-assisted implementation, give the agent a narrow instruction:

1. Name the closest reference strategy.
2. State which algorithm semantics differ.
3. Ask it to create or modify only the files in the implementation map.
4. Require `triton_grouped` attention unless the task is explicitly debugging a
   fallback path.
5. Ask for a tiny generation smoke run, then a limited benchmark run, then
   profiling only after correctness is stable.

This workflow works well because Diffulex keeps strategy registration,
attention metadata, sampler logic, and dLLM-oriented Triton kernels in explicit
locations. It lets researchers explore real high-performance algorithm variants
with much less systems boilerplate.
