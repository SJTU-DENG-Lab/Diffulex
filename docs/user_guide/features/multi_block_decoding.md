# Multi-Block Decoding

Multi-block decoding lets Diffulex manage multiple diffusion blocks during one
request. Instead of treating the full generation span as one monolithic region,
the scheduler and request state track block-level progress and decide when new
blocks can be added or released.

## When It Applies

Multi-block behavior is used by strategies such as `multi_bd` and by the core
block-aware templates in `diffulex.engine` and `diffulex.mixin.multi_block`.
It is also part of strategy specializations such as DMax-style token merging.

For `d2f`, config normalization forces `multi_block_prefix_full=True`. For
`multi_bd` and `dmax`, config normalization forces `multi_block_prefix_full=False`.

## Block Size

`block_size` controls the token span managed as one diffusion block.

For most model families, choose one of `4`, `8`, `16`, or `32`. The general
default is `32`. `model_name="diffusion_gemma"` uses `256`, and config
normalization forces that value for both block and page size.

`block_size` must not exceed `page_size`. If you change one, check the other at
the same time so the KV cache layout still matches the decoding block layout.

Larger block sizes can reduce block-management overhead but increase the amount
of work tied to one block. Smaller block sizes expose more scheduling granularity
but can increase bookkeeping pressure.

## Buffer Size

`buffer_size` controls how many active diffusion blocks the request can keep in
the multi-block buffer.

The general default is `4`. `model_name="diffusion_gemma"` is normalized to
`1`.

Increasing the buffer can improve overlap between block progress and scheduling,
but it also increases active state. When debugging, use the strategy default or a
small value before tuning throughput.

## Related Arguments

| Surface | Names | Notes |
| --- | --- | --- |
| Python/config | `block_size`, `buffer_size` | Primary knobs for block span and active block count. |
| CLI | `--block-size`, `--buffer-size` | Use for quick serving or benchmark overrides. |
| Related config | `page_size`, `multi_block_prefix_full` | `page_size` must stay compatible with `block_size`; strategy normalization controls `multi_block_prefix_full`. |
