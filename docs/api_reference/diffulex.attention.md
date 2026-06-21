# diffulex.attention

`diffulex.attention` is the boundary between strategy-specific attention
metadata and the attention kernels used by model layers. Strategy model runners
prepare metadata for each engine step, then the attention layer reads that
metadata through the package-level fetch hook.

This package should stay small. New decoding strategies usually add their own
metadata subclasses under `diffulex.strategy.*.attention`; shared backend
selection and metadata plumbing belong here.

| Module | Role |
| --- | --- |
| `diffulex.attention.attn_impl` | Implements the `Attention` module and dispatches to reference, Triton, or grouped Triton attention paths. |
| `diffulex.attention.metadata` | Defines the shared metadata base class and global fetch/warmup helpers used by attention layers. |

## diffulex.attention.attn_impl

This module owns the common attention layer interface used by model
implementations. It keeps backend-specific calls behind a single `Attention`
module so model code can pass hidden states, QKV projections, and cache tensors
without directly choosing a kernel.

| Symbol | Purpose |
| --- | --- |
| `Attention` | PyTorch module that selects the configured attention implementation and consumes the current attention metadata. |
| `reference_torch_attention` | Debug/reference implementation for correctness checks. |
| `triton_attention` | Optimized attention path for the standard metadata layout. |
| `triton_grouped_attention` | Optimized grouped attention path for grouped metadata layouts. |

Use the reference path when debugging tensor shapes or metadata. Use Triton
paths for normal optimized runs once request scheduling and cache layout are
known to be correct.

## diffulex.attention.metadata

This module defines the metadata contract shared by attention layers and
strategy model runners. A strategy-specific runner installs a fetch function
before execution; the attention layer reads the current metadata through that
function during forward passes.

| Symbol | Purpose |
| --- | --- |
| `AttnMetaDataBase` | Base dataclass for prefill/decode lengths, page tables, slot mapping, context lengths, page size, block size, and cache layout. |
| `set_fetch_fn_for_attn_metadata` | Installs the strategy-specific metadata fetch function. |
| `set_warming_up` / `is_warming_up` / `reset_warming_up` | Track CUDA graph warmup state so attention code can distinguish warmup from normal execution. |

When adding a new strategy, define the strategy-specific metadata subclass in
the strategy package and keep only shared metadata mechanics in this module.
