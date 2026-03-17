"""Standalone debug script for chunked prefill kernel - no pytest dependency."""

import torch
from test.python.kernel.test_dllm_flash_attn_chunked_prefill_unified_kernel import (
    build_test_data,
    call_chunked_prefill_kernel,
    naive_chunked_prefill_ref,
)


def debug_single_case():
    """Mixed batch: prefill + decode with various contexts."""
    torch.manual_seed(42)

    # Config: mixed batch (prefill + decode)
    q_lens = [192, 128, 128, 128]      # seq0: prefill, seq1: prefill, seq2-3: decode
    valid_q_lens = [192, 128, 96, 64]  # seq0-1: full, seq2-3: partial
    ctx_lens = [0, 64, 256, 128]       # seq0: no cache, seq1-3: with cache
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 32
    dllm_block_size = 32

    # Build data
    data = build_test_data(
        q_lens,
        valid_q_lens,
        ctx_lens,
        num_heads,
        num_kv_heads,
        head_dim,
        page_size,
        dllm_block_size=dllm_block_size,
    )
    q, k, v, k_cache, v_cache, pt, st, cl, cu, vs, pl, ppl = data
    scale = 1.0 / head_dim**0.5

    # Kernel output
    out = call_chunked_prefill_kernel(
        q, k, v, k_cache, v_cache, pt, st, cl, cu, vs, pl, ppl,
        scale, dllm_block_size, is_block_causal=True
    )

    # Reference output (with mask visualization)
    ref = naive_chunked_prefill_ref(
        q, k, v, k_cache, v_cache, pt, st, cl, cu, vs,
        [0], [0], scale, page_size, dllm_block_size,
        is_block_causal=True, visualize_mask=True
    )

    # Compare
    diff = (out - ref).abs()
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"Output shape: {out.shape}")
    print(f"Max diff: {diff.max().item():.6f}")
    print(f"Mean diff: {diff.mean().item():.6f}")
    print(f"{'='*80}\n")

    return out, ref


if __name__ == "__main__":
    debug_single_case()
