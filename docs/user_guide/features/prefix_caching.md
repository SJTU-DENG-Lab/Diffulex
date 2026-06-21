# Prefix Caching

Prefix caching reuses compatible prefix KV cache state across requests. It is a
capacity and latency optimization for workloads with shared prompts or repeated
prefixes.

## Configuration

`enable_prefix_caching` controls whether compatible strategies may use prefix
caching.

The value is boolean and defaults to `True`. Strategy normalization still has
the final say: `decoding_strategy="d2f"` forces prefix caching off, while
`multi_bd` and `dmax` leave it enabled when the rest of the cache layout is
compatible.

| Surface | How to set it | Notes |
| --- | --- | --- |
| Server CLI | Prefix caching is enabled by default; add `--disable-prefix-caching` to turn it off. | Useful when debugging cache layout or request state. |
| Benchmark CLI | Use `--enable-prefix-caching` or `--no-enable-prefix-caching`. | Makes prefix-cache behavior explicit in experiment commands. |

## When to Disable It

Disable prefix caching while debugging cache layout, request state, or strategy
changes. Once correctness is stable, re-enable it for throughput and latency
checks.

## Related Arguments

| Surface | Names | Notes |
| --- | --- | --- |
| Python/config | `enable_prefix_caching` | Main config field before strategy normalization. |
| CLI | `--disable-prefix-caching`, `--enable-prefix-caching`, `--no-enable-prefix-caching` | Server and benchmark CLIs expose the toggle with different flag names. |
| Related config | `decoding_strategy`, `multi_block_prefix_full`, `kv_cache_layout` | Strategy and cache layout decide whether prefix reuse is actually compatible. |
