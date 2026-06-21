# diffulex_kernel

The `diffulex_kernel` package exposes optional optimized kernel entry points
used by attention, KV cache, top-k, and MoE execution paths. The package root is
lazy: importing `diffulex_kernel` does not load Triton or other heavy optional
dependencies until a specific kernel symbol is requested.

## Public Symbols

- `dllm_chunked_prefill`
- `chunked_prefill_attn_unified`
- `chunked_prefill_attn_grouped_unified`
- `store_kv_cache_unified_layout`
- `store_kv_cache_distinct_layout`
- `load_kv_cache`
- `fused_moe`
- `vllm_fused_moe`
- `fused_expert_packed`
- `fused_topk`
- `fused_group_limited_topk`
- `fused_grouped_topk`

## Attention Kernels

`dllm_chunked_prefill` and `chunked_prefill_attn_unified` refer to the unified
chunked prefill attention implementation. The grouped variant is available as
`chunked_prefill_attn_grouped_unified`. Engine code chooses these paths through
strategy-specific model runner and attention metadata logic.

## KV Cache Kernels

KV cache helpers support both configured cache layouts:

- `store_kv_cache_unified_layout`
- `store_kv_cache_distinct_layout`
- `load_kv_cache`

Use the layout that matches `Config.kv_cache_layout`. Mixing layouts between
engine configuration and direct kernel calls will produce incorrect cache
interpretation.

## MoE and Top-k Kernels

Fused MoE and top-k helpers accelerate routing and expert execution for
supported models. Available entry points include `fused_moe`,
`vllm_fused_moe`, `fused_expert_packed`, `fused_topk`, and grouped top-k
aliases.

## Direct Use

Most users should not call these kernels directly. They are lower-level
building blocks expected to receive tensors in layouts prepared by Diffulex
model runners and layers. Direct calls are useful for focused kernel tests,
profiling scripts, and numerical comparisons against reference implementations.

When adding a new kernel, include a focused test under `test/python/kernel/` or
the relevant third-party kernel test directory.
