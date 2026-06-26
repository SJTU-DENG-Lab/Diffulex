# Multi-Block Diffusion

**Multi-Block Diffusion (MultiBD)** is the inference formulation used by
Multi-Block Diffusion Language Models (MBD-LMs). Native block diffusion language
models often run **Single-Block Diffusion (SingleBD)**: one noisy block is
iteratively denoised, committed into the KV cache, and only then can the next
block start. SingleBD preserves KV caching, but it leaves inter-block
parallelism unused.

MultiBD keeps a bounded running-set of consecutive blocks active at the same
time. Earlier blocks may be complete and waiting to enter the KV cache while
later blocks are already being refined. This exposes inter-block parallelism
without giving up the clean cached prefix that makes block diffusion models
servable.

In Diffulex, the runtime option for this method is:

```yaml
decoding_strategy: multi_bd
```

This page describes the engine-side implementation. For reproducing the
MBD-LMs experiments and their training recipe, use the Diffulex
[`mbd-lms`](https://github.com/SJTU-DENG-Lab/Diffulex/tree/mbd-lms) branch.
For new runtime development and open-source contributions, use
[`main`](https://github.com/SJTU-DENG-Lab/Diffulex/tree/main).

## Method Terms

| Term | Meaning in Diffulex |
| --- | --- |
| SingleBD | One active noisy block is decoded at a time; later blocks wait for the current block to finish and enter KV cache. |
| MultiBD | A bounded active block set is decoded concurrently with block-causal visibility. |
| Running-set | The consecutive blocks that have not yet become part of the clean cached prefix. |
| Block Buffer | A fixed-size physical buffer for resident blocks. It keeps shapes stable while logical blocks enter, complete, and commit to KV cache. |
| MultiTF | The MBD-LMs post-training recipe that aligns training states with practical MultiBD inference states. This is part of the experiment branch, not a server flag. |

## Runtime Mapping

`decoding_strategy="multi_bd"` selects the block-aware request state,
scheduler, KV cache manager, model runner, and attention metadata path. The
core implementation lives in `diffulex.engine` and `diffulex.mixin.multi_block`.

At a high level:

1. The request state tracks block-level progress and the active running-set.
2. The scheduler decides when another block can enter the active set.
3. Completed front blocks are committed into the KV cache.
4. Prefix caching reuses the clean cached prefix for later steps and requests.
5. Static-shape execution can run over the configured block/buffer layout.

For `d2f`, config normalization forces `multi_block_prefix_full=True`, which
uses full-prefix multi-block behavior and disables prefix caching. For
`multi_bd` and `dmax`, config normalization forces
`multi_block_prefix_full=False`, which is the block-causal path needed by
prefix caching.

## When It Applies

Use `multi_bd` for model families that are configured for block-causal
multi-block decoding, such as LLaDA2, SDAR, SDAR-MoE, Fast-dLLM-v2, Stable-
DiffCoder, and Dream reasoner paths.

There are two common usage modes:

| Mode | What it means |
| --- | --- |
| Training-free MultiBD | Run an existing compatible BD-LM with `decoding_strategy=multi_bd`. This can improve parallelism, but quality depends on how well the checkpoint tolerates MultiBD states. |
| MBD-LM reproduction | Use checkpoints/configs from the MBD-LMs experiment setup, where MultiTF was used to align training with MultiBD inference. Reproduce these through the `mbd-lms` branch. |

DMax-style token merging composes with the same block-causal runtime ideas, but
uses `decoding_strategy="dmax"` because it has additional edit-sampling and
token-merge requirements.

## Block Size

`block_size` controls the token span managed as one diffusion block.

For most model families, choose one of `4`, `8`, `16`, or `32`. The general
default is `32`. `model_name="diffusion_gemma"` uses `256`, and config
normalization forces that value for both block and page size.

`block_size` must not exceed `page_size`. If you change one, check the other at
the same time so the KV cache layout still matches the decoding block layout.

Larger block sizes can reduce block-management overhead but increase the amount
of work tied to one block. Smaller block sizes expose more scheduling
granularity but can increase bookkeeping pressure.

## Buffer Size

`buffer_size` controls how many active diffusion blocks the request can keep in
the multi-block buffer.

The general default is `4`. `model_name="diffusion_gemma"` is normalized to
`1`.

Increasing the buffer can expose more inter-block parallelism and improve
overlap between block progress, scheduling, and KV-cache commits. It also
increases active state and per-step work. When debugging, use the strategy
default or a small value before tuning throughput.

## Related Arguments

| Surface | Names | Notes |
| --- | --- | --- |
| Python/config | `block_size`, `buffer_size` | Primary knobs for block span and active block count. |
| CLI | `--block-size`, `--buffer-size` | Use for quick serving or benchmark overrides. |
| Related config | `page_size`, `multi_block_prefix_full` | `page_size` must stay compatible with `block_size`; strategy normalization controls `multi_block_prefix_full`. |
