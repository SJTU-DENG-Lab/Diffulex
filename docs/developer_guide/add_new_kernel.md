# Add New Kernel

Add a new kernel when Python or the existing Triton/vLLM paths are not enough
for a specific operation. Keep the kernel isolated until it matches a reference
implementation, then integrate it through the narrowest engine boundary.

## Choose the Location

Use `diffulex_kernel/python/` for Diffulex-owned Python/Triton kernel entry
points. Use the relevant third-party kernel package only when extending code
that already lives there.

Expose public kernel symbols through `diffulex_kernel/__init__.py` only when
other packages need to import them directly.

## Keep a Reference Path

Before optimizing, write or identify a reference implementation. The reference
can be a simple PyTorch implementation or an existing slower kernel path.

A good kernel test checks:

- output values against the reference;
- supported dtypes;
- boundary shapes;
- layout assumptions;
- device placement.

## Integrate Through Engine Boundaries

Most kernels should be called through one of these layers:

- attention implementation;
- KV cache helper;
- model layer;
- MoE routing or GEMM helper;
- strategy-specific model runner.

Avoid calling a new kernel directly from scheduler code. The scheduler should
decide what runs, not own tensor-level implementation details.

## Validate Layout Assumptions

Document the expected tensor layout near the call site. For KV cache and
attention kernels, verify these values together:

| Field | What to verify |
| --- | --- |
| `page_size` | Matches the KV cache paging expected by the kernel. |
| `block_size` | Fits within the selected page size and strategy layout. |
| `kv_cache_layout` | Matches the memory interpretation used by the kernel. |
| Attention metadata fields | Describe the same shape, page table, and context layout that the kernel reads. |
| dtype and device | Match the kernel's supported execution path. |

Mismatched metadata often looks like a kernel bug, so inspect metadata before
changing low-level code.

## Profiling

Profile the kernel in isolation before measuring full engine throughput. Once
the kernel is integrated, compare against the previous path with the same model,
prompt set, token limits, batch limits, and parallelism.

## Verification Checklist

1. Add a focused correctness test.
2. Add shape or dtype coverage for supported variants.
3. Run the focused kernel test.
4. Run the smallest engine path that reaches the kernel.
5. Profile only after correctness is stable.
