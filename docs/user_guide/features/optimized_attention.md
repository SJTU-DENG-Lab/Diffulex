# Optimized Attention

Optimized attention paths reduce memory movement and improve throughput by
combining strategy metadata, paged KV cache layout, and specialized kernels.

## Attention Implementation

`attn_impl` selects the attention backend.

The core config accepts `triton`, `triton_grouped`, and `naive`.

The server CLI and benchmark CLI expose all core choices.

Use `triton_grouped` for normal serving, benchmarking, and performance reports.
The older `triton` path and the `naive` path are kept for compatibility and
debugging; they are not recommended for optimized runs.

## Paged Attention

Paged attention stores KV cache in pages instead of requiring a single
contiguous region per request. The scheduler and KV cache manager use page
tables to map request positions to cache storage.

| Key | How to set it | What it does |
| --- | --- | --- |
| `page_size` | Use `4`, `8`, `16`, or `32` for most models; `diffusion_gemma` uses `256`. | Sets the KV cache page size used by paged attention. |
| `block_size` | Keep it less than or equal to `page_size`. | Keeps diffusion block layout compatible with KV cache pages. |
| `kv_cache_layout` | Use `unified` unless a strategy or experiment needs `distinct`. | Chooses how cache storage is organized for attention. |

## Chunked Prefill

Chunked prefill splits long prefill work into smaller chunks so the engine can
respect token budgets and cache constraints. Strategy-specific model runners
prepare chunked prefill tensors and attention metadata.

## Related Arguments

| Surface | Names | Notes |
| --- | --- | --- |
| Python/config | `attn_impl`, `page_size`, `kv_cache_layout` | Primary configuration fields for attention and cache layout. |
| CLI | `--attn-impl`, `--page-size`, `--kv-cache-layout` | Use for serving or benchmark overrides. |
| Kernel package | `diffulex_kernel` | Provides the lower-level optimized attention and cache helpers. |
