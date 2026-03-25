import os
import pytest
import torch
import torch.nn.functional as F
import triton
from einops import rearrange

from diffulex_kernel.python.chunked_prefill_triton import (
    _chunked_prefill_attn_unified_kernel,
)
from diffulex_kernel.python.kv_cache_kernels import store_kv_cache_unified


# ---------------------------------------------------------------------------
# Mask visualization helper
# ---------------------------------------------------------------------------


def _visualize_mask(mask, seq_id, ctx_len, valid_q_len, block_size, label):
    """Visualize attention mask with clear structure."""
    print(f"\n{'='*80}")
    print(f"Seq {seq_id} | {label} | ctx_len={ctx_len}, valid_q_len={valid_q_len}, block_size={block_size}")
    print(f"{'='*80}")

    mask_np = mask.cpu().numpy()
    total_kv = mask_np.shape[1]

    # Print header
    print(f"Mask shape: Q={mask_np.shape[0]} x KV={total_kv} (cache={ctx_len}, new={valid_q_len})")

    # Compact visualization for large masks
    if valid_q_len > 64 or total_kv > 64:
        print("\n[Compact view - showing block boundaries]")
        step = max(1, block_size // 4)
        sample_q = list(range(0, valid_q_len, step))
        sample_kv = list(range(0, total_kv, step))

        print(f"\n    KV→", end="")
        for kv_idx in sample_kv:
            if kv_idx < ctx_len:
                print(f" C{kv_idx:3d}", end="")
            else:
                print(f" N{kv_idx-ctx_len:3d}", end="")
        print()

        for q_idx in sample_q:
            block_id = q_idx // block_size
            print(f"Q{q_idx:3d}(B{block_id})", end="")
            for kv_idx in sample_kv:
                print(f"  {'█' if mask_np[q_idx, kv_idx] else '·'}  ", end="")
            print()
    else:
        # Full visualization for small masks
        print(f"\n    KV→", end="")
        for kv_idx in range(total_kv):
            if kv_idx < ctx_len:
                print(f"C{kv_idx%10}", end="")
            else:
                print(f"N{(kv_idx-ctx_len)%10}", end="")
        print()

        for q_idx in range(valid_q_len):
            block_id = q_idx // block_size
            print(f"Q{q_idx:2d}(B{block_id})", end="")
            for kv_idx in range(total_kv):
                print("█" if mask_np[q_idx, kv_idx] else "·", end="")
            print()

    # Statistics
    visible_per_q = mask_np.sum(axis=1)
    print(f"\nStats: min_visible={visible_per_q.min():.0f}, max_visible={visible_per_q.max():.0f}, "
          f"avg_visible={visible_per_q.mean():.1f}")
    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# Kernel invocation (direct call, bypasses the incomplete Python wrapper)
# ---------------------------------------------------------------------------


def call_chunked_prefill_kernel(
    q,
    k,
    v,
    k_cache,
    v_cache,
    page_tables,
    status_table,
    context_lens,
    cu_seqlens_q,
    valid_slices,
    prefix_lens,
    padded_prefix_lens,
    softmax_scale,
    dllm_block_size,
    is_block_causal,
    is_prefix_full=False,
    BLOCK_M=64,
    BLOCK_N=64,
):
    o = torch.zeros_like(q)
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    num_groups = num_heads // num_kv_heads
    head_dim = q.shape[-1]
    head_dim_padded = 1 << (head_dim - 1).bit_length()
    page_size = k_cache.shape[1]
    num_seqs = len(cu_seqlens_q) - 1

    q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    max_seqlen_q = int(q_lens.max().item())

    grid = (num_seqs, num_heads, triton.cdiv(max_seqlen_q, BLOCK_M))

    launch_kwargs = {}
    if os.environ.get("DIFFULEX_DISABLE_CHUNKED_PREFILL_AUTOTUNE", "0") == "1":
        launch_kwargs["BLOCK_M"] = BLOCK_M
        launch_kwargs["BLOCK_N"] = BLOCK_N

    _chunked_prefill_attn_unified_kernel[grid](
        q,
        k,
        v,
        o,
        k_cache,
        v_cache,
        page_tables,
        status_table,
        context_lens,
        cu_seqlens_q,
        valid_slices,
        prefix_lens,
        padded_prefix_lens,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        page_tables.stride(0),
        page_tables.stride(1),
        NUM_GROUPS=num_groups,
        HEAD_DIM=head_dim,
        HEAD_DIM_PADDED=head_dim_padded,
        PAGE_SIZE=page_size,
        DLLM_BLOCK_SIZE=dllm_block_size,
        IS_BLOCK_CAUSAL=is_block_causal,
        IS_PREFIX_FULL=is_prefix_full,
        **launch_kwargs,
    )
    return o


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def naive_chunked_prefill_ref(
    q,
    k,
    v,
    k_cache,
    v_cache,
    page_tables,
    statuses,
    context_lens,
    cu_seqlens_q,
    valid_slices,
    prefix_lens_list,
    padded_prefix_lens_list,
    scale,
    page_size,
    dllm_block_size,
    is_block_causal,
    is_prefix_full=False,
    visualize_mask=False,
):
    num_seqs = len(cu_seqlens_q) - 1
    output = torch.zeros_like(q)

    for seq_id in range(num_seqs):
        q_start = int(cu_seqlens_q[seq_id].item())
        valid_slice = int(valid_slices[seq_id].item())
        valid_q_len = valid_slice - q_start
        ctx_len = int(context_lens[seq_id].item())

        if valid_q_len <= 0:
            continue

        q_seq = q[q_start : q_start + valid_q_len]

        k_parts, v_parts = [], []
        for rel_page_id in range(page_tables.shape[1]):
            abs_page_id = int(page_tables[seq_id, rel_page_id].item())
            if abs_page_id < 0:
                continue
            page_start = rel_page_id * page_size
            if page_start >= ctx_len:
                break
            n = min(page_start + page_size, ctx_len) - page_start
            k_parts.append(k_cache[abs_page_id, :n])
            v_parts.append(v_cache[abs_page_id, :n])

        k_new = k[q_start : q_start + valid_q_len]
        v_new = v[q_start : q_start + valid_q_len]

        if k_parts:
            k_full = torch.cat(k_parts + [k_new], dim=0)
            v_full = torch.cat(v_parts + [v_new], dim=0)
        else:
            k_full = k_new
            v_full = v_new

        mask = None
        if is_block_causal:
            qi = torch.arange(valid_q_len, device=q.device)
            kj = torch.arange(valid_q_len, device=q.device)
            abs_q = ctx_len + qi
            abs_k = ctx_len + kj

            if is_prefix_full:
                status = statuses[seq_id]
                P = prefix_lens_list[seq_id]
                P_prime = padded_prefix_lens_list[seq_id]

                if status == 0:
                    pure_prefix = (qi[:, None] < P) & (kj[None, :] < P)
                    padded_causal = ((qi[:, None] >= P) & (qi[:, None] < P_prime)) & (kj[None, :] < P_prime)
                    block_ends = ((abs_q // dllm_block_size) + 1) * dllm_block_size
                    block_mask_extend = (abs_k[None, :] < block_ends[:, None]) & (qi[:, None] >= P_prime)

                    new_kv_mask = pure_prefix | padded_causal | block_mask_extend
                else:
                    block_ends = ((abs_q // dllm_block_size) + 1) * dllm_block_size
                    new_kv_mask = abs_k[None, :] < block_ends[:, None]
            else:
                block_ends = ((abs_q // dllm_block_size) + 1) * dllm_block_size
                new_kv_mask = abs_k[None, :] < block_ends[:, None]

            cache_mask = torch.ones(
                valid_q_len,
                ctx_len,
                dtype=torch.bool,
                device=q.device,
            )
            mask = torch.cat([cache_mask, new_kv_mask], dim=1)
            mask = mask.unsqueeze(0).unsqueeze(0)

            if visualize_mask:
                _visualize_mask(mask[0, 0], seq_id, ctx_len, valid_q_len, dllm_block_size,
                                f"block_causal{'_prefix' if is_prefix_full else ''}")

        q_sdpa = rearrange(q_seq, "s h d -> 1 h s d")
        k_sdpa = rearrange(k_full, "s h d -> 1 h s d")
        v_sdpa = rearrange(v_full, "s h d -> 1 h s d")

        attn_out = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False, # Mask 自己处理了因果逻辑
            scale=scale,
            enable_gqa=True, # PyTorch 2.2+ 原生支持 GQA
        )
        output[q_start : q_start + valid_q_len] = rearrange(attn_out, "1 h s d -> s h d").to(output.dtype)

    return output


# ---------------------------------------------------------------------------
# Test data builder
# ---------------------------------------------------------------------------
def build_test_data(
    q_lens,
    valid_q_lens,
    ctx_lens,
    num_heads,
    num_kv_heads,
    head_dim,
    page_size,
    statuses=None,
    prefix_lens=None,
    dllm_block_size=32,
    device="cuda",
    dtype=torch.bfloat16,
):
    num_seqs = len(q_lens)
    assert len(valid_q_lens) == num_seqs and len(ctx_lens) == num_seqs

    cu = [0]
    for ql in q_lens:
        cu.append(cu[-1] + ql)
    cu_seqlens_q = torch.tensor(cu, dtype=torch.int32, device=device)
    total_len = cu[-1]

    valid_slices = torch.tensor(
        [cu[i] + valid_q_lens[i] for i in range(num_seqs)],
        dtype=torch.int32,
        device=device,
    )
    context_lens_t = torch.tensor(ctx_lens, dtype=torch.int32, device=device)

    if statuses is None:
        statuses = [0] * num_seqs
    status_table = torch.tensor(statuses, dtype=torch.int32, device=device)

    if prefix_lens is None:
        prefix_lens = [0] * num_seqs
    prefix_lens_t = torch.tensor(prefix_lens, dtype=torch.int32, device=device)
    padded = [((p + dllm_block_size - 1) // dllm_block_size) * dllm_block_size for p in prefix_lens]
    padded_prefix_lens_t = torch.tensor(padded, dtype=torch.int32, device=device)

    q = torch.randn(total_len, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_len, num_kv_heads, head_dim, dtype=dtype, device=device)

    pages_per_seq = [(c + page_size - 1) // page_size for c in ctx_lens]
    max_pages = max(pages_per_seq) if any(c > 0 for c in ctx_lens) else 1
    total_pages = max(sum(pages_per_seq), 1)

    k_cache = torch.randn(
        total_pages,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    v_cache = torch.randn_like(k_cache)

    page_tables = torch.full(
        (num_seqs, max_pages),
        -1,
        dtype=torch.int32,
        device=device,
    )
    offset = 0
    for i in range(num_seqs):
        for p in range(pages_per_seq[i]):
            page_tables[i, p] = offset + p
        offset += pages_per_seq[i]

    return (
        q,
        k,
        v,
        k_cache,
        v_cache,
        page_tables,
        status_table,
        context_lens_t,
        cu_seqlens_q,
        valid_slices,
        prefix_lens_t,
        padded_prefix_lens_t,
    )


# ---------------------------------------------------------------------------
# Shared runner
# ---------------------------------------------------------------------------


def _run_test(
    q_lens,
    valid_q_lens,
    ctx_lens,
    num_heads,
    num_kv_heads,
    head_dim,
    page_size,
    dllm_block_size,
    is_block_causal,
    is_prefix_full=False,
    statuses=None,
    prefix_lens=None,
    seed=42,
    atol=5e-3,
    rtol=5e-3,
    visualize_mask=False,
):
    torch.manual_seed(seed)
    num_seqs = len(q_lens)
    if statuses is None:
        statuses = [0] * num_seqs
    if prefix_lens is None:
        prefix_lens = [0] * num_seqs
    padded_prefix_lens = [((p + dllm_block_size - 1) // dllm_block_size) * dllm_block_size for p in prefix_lens]

    data = build_test_data(
        q_lens,
        valid_q_lens,
        ctx_lens,
        num_heads,
        num_kv_heads,
        head_dim,
        page_size,
        statuses=statuses,
        prefix_lens=prefix_lens,
        dllm_block_size=dllm_block_size,
    )
    q, k, v, k_cache, v_cache, pt, st, cl, cu, vs, pl, ppl = data
    scale = 1.0 / head_dim**0.5

    out = call_chunked_prefill_kernel(
        q,
        k,
        v,
        k_cache,
        v_cache,
        pt,
        st,
        cl,
        cu,
        vs,
        pl,
        ppl,
        scale,
        dllm_block_size,
        is_block_causal=is_block_causal,
        is_prefix_full=is_prefix_full,
    )
    ref = naive_chunked_prefill_ref(
        q,
        k,
        v,
        k_cache,
        v_cache,
        pt,
        statuses,
        cl,
        cu,
        vs,
        prefix_lens,
        padded_prefix_lens,
        scale,
        page_size,
        dllm_block_size,
        is_block_causal=is_block_causal,
        is_prefix_full=is_prefix_full,
        visualize_mask=visualize_mask,
    )

    for i in range(num_seqs):
        q_start = int(cu[i].item())
        q_end = int(cu[i + 1].item())
        torch.testing.assert_close(
            out[q_start:q_end],
            ref[q_start:q_end],
            atol=atol,
            rtol=rtol,
        )


# ========================= Case 1: Pure Prefill ==========================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pure_prefill_varlen():
    """No context, varlen, full attention (non-DLLM prefill)."""
    _run_test(
        q_lens=[128, 96, 64, 160],
        valid_q_lens=[128, 96, 64, 160],
        ctx_lens=[0, 0, 0, 0],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=32,
        dllm_block_size=32,
        is_block_causal=False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pure_prefill_varlen_block_causal():
    """No context, varlen, block-causal (DLLM prefill)."""
    _run_test(
        q_lens=[128, 96, 64, 160],
        valid_q_lens=[128, 96, 64, 160],
        ctx_lens=[0, 0, 0, 0],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=32,
        dllm_block_size=32,
        is_block_causal=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pure_prefill_single_seq():
    """Single sequence pure prefill, no context."""
    _run_test(
        q_lens=[256],
        valid_q_lens=[256],
        ctx_lens=[0],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=32,
        dllm_block_size=32,
        is_block_causal=False,
    )


# ===================== Case 2: Prefix Prefill =============================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefix_prefill_varlen():
    """With prefix cache context, varlen, full attention."""
    _run_test(
        q_lens=[128, 96, 64, 160],
        valid_q_lens=[128, 96, 64, 160],
        ctx_lens=[64, 128, 32, 96],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=32,
        dllm_block_size=32,
        is_block_causal=False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefix_prefill_varlen_block_causal():
    """With prefix cache context, varlen, block-causal."""
    _run_test(
        q_lens=[128, 96, 64, 160],
        valid_q_lens=[128, 96, 64, 160],
        ctx_lens=[64, 128, 32, 96],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=32,
        dllm_block_size=32,
        is_block_causal=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefix_prefill_unaligned_context():
    """Context lengths not aligned to page_size boundary."""
    _run_test(
        q_lens=[128, 96, 64],
        valid_q_lens=[128, 96, 64],
        ctx_lens=[17, 55, 100],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=32,
        dllm_block_size=32,
        is_block_causal=False,
    )


# ====================== Case 3: Pure Decode ===============================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pure_decode_block_causal():
    """
    Decode with context, fixed q_len (CUDA graph), block-causal.
    valid_q_lens are random multiples of dllm_block_size.
    """
    dllm_block_size = 32
    padded = dllm_block_size * 4  # 128
    _run_test(
        q_lens=[padded, padded, padded, padded],
        valid_q_lens=[96, 64, 128, 32],
        ctx_lens=[256, 128, 512, 64],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=32,
        dllm_block_size=dllm_block_size,
        is_block_causal=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pure_decode_block_causal_dbs64():
    """Decode with dllm_block_size=64."""
    dllm_block_size = 64
    padded = dllm_block_size * 2  # 128
    _run_test(
        q_lens=[padded, padded, padded, padded],
        valid_q_lens=[128, 64, 128, 64],
        ctx_lens=[128, 256, 64, 192],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=32,
        dllm_block_size=dllm_block_size,
        is_block_causal=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pure_decode_single_seq():
    """Single sequence decode with context."""
    _run_test(
        q_lens=[128],
        valid_q_lens=[64],
        ctx_lens=[256],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=32,
        dllm_block_size=32,
        is_block_causal=True,
    )


# =================== Case 4: Chunked Prefill (Mixed) =====================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_chunked_prefill_mixed():
    """
    Mixed batch: first 2 are prefill (valid==full, ctx may be 0),
    last 2 are decode (valid < full, has context), block-causal.
    """
    _run_test(
        q_lens=[192, 128, 128, 128],
        valid_q_lens=[192, 128, 96, 64],
        ctx_lens=[0, 64, 256, 128],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=32,
        dllm_block_size=32,
        is_block_causal=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_chunked_prefill_mixed_all_have_cache():
    """Mixed batch where every request has prefix cache."""
    _run_test(
        q_lens=[160, 128, 128, 128],
        valid_q_lens=[160, 128, 96, 32],
        ctx_lens=[32, 96, 256, 192],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=32,
        dllm_block_size=32,
        is_block_causal=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "q_lens,valid_q_lens,ctx_lens",
    [
        # Longer sequences mixed with decode requests.
        ([512, 384, 256, 128], [512, 256, 160, 96], [0, 96, 512, 256]),
        # Prefill + decode + minimal decode buffer_size=1 (32/32).
        ([320, 160, 128, 32], [320, 160, 96, 32], [64, 0, 256, 128]),
        # Multi-strategy mix with shorter prefill and deeper decode cache.
        ([256, 32, 192, 128], [256, 32, 64, 32], [0, 64, 384, 512]),
    ],
)
def test_chunked_prefill_mixed_extended_strategies(q_lens, valid_q_lens, ctx_lens):
    """
    Extended mixed-batch coverage:
      - longer sequence lengths,
      - more mixed prefill/decode patterns,
      - explicit buffer_size=1 case (q_len=32 with block_size=32).
    """
    _run_test(
        q_lens=q_lens,
        valid_q_lens=valid_q_lens,
        ctx_lens=ctx_lens,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=32,
        dllm_block_size=32,
        is_block_causal=True,
    )


# =================== Additional Coverage =================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gqa_ratio_4():
    """GQA ratio = 4 (num_heads=32, num_kv_heads=8), mixed scenario."""
    _run_test(
        q_lens=[128, 96],
        valid_q_lens=[128, 64],
        ctx_lens=[0, 128],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=32,
        dllm_block_size=32,
        is_block_causal=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gqa_ratio_8():
    """GQA ratio = 8 (num_heads=32, num_kv_heads=4)."""
    _run_test(
        q_lens=[128, 128],
        valid_q_lens=[128, 96],
        ctx_lens=[64, 192],
        num_heads=32,
        num_kv_heads=4,
        head_dim=128,
        page_size=32,
        dllm_block_size=32,
        is_block_causal=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_head_dim_64():
    """head_dim = 64."""
    _run_test(
        q_lens=[128, 96],
        valid_q_lens=[128, 64],
        ctx_lens=[0, 96],
        num_heads=32,
        num_kv_heads=8,
        head_dim=64,
        page_size=32,
        dllm_block_size=32,
        is_block_causal=True,
    )


# =================== Case 5: Prefix Full =================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefix_full_pure_prefill_no_cache():
    """Prefix-full: all prefilling, no cache, various unaligned prefix lengths."""
    _run_test(
        q_lens=[128, 96, 64, 160],
        valid_q_lens=[128, 96, 64, 160],
        ctx_lens=[0, 0, 0, 0],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=32,
        dllm_block_size=32,
        is_block_causal=True,
        is_prefix_full=True,
        statuses=[0, 0, 0, 0],
        prefix_lens=[32, 20, 16, 48],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefix_full_pure_prefill_aligned():
    """Prefix-full: prefix_len already aligned to dllm_block_size (empty padding zone)."""
    _run_test(
        q_lens=[128, 96, 64],
        valid_q_lens=[128, 96, 64],
        ctx_lens=[0, 0, 0],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=32,
        dllm_block_size=32,
        is_block_causal=True,
        is_prefix_full=True,
        statuses=[0, 0, 0],
        prefix_lens=[32, 64, 32],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefix_full_with_cache():
    """Prefix-full: prefilling with KV cache context."""
    _run_test(
        q_lens=[128, 96, 64, 160],
        valid_q_lens=[128, 96, 64, 160],
        ctx_lens=[64, 128, 32, 96],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=32,
        dllm_block_size=32,
        is_block_causal=True,
        is_prefix_full=True,
        statuses=[0, 0, 0, 0],
        prefix_lens=[32, 20, 16, 48],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefix_full_mixed_prefill_decode():
    """Prefix-full: mixed batch with prefill (status=0) and decode (status!=0)."""
    _run_test(
        q_lens=[192, 128, 128, 128],
        valid_q_lens=[192, 128, 96, 64],
        ctx_lens=[0, 64, 256, 128],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=32,
        dllm_block_size=32,
        is_block_causal=True,
        is_prefix_full=True,
        statuses=[0, 0, 1, 1],
        prefix_lens=[40, 24, 0, 0],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefix_full_zero_prefix():
    """Prefix-full with prefix_len=0 degenerates to standard block-causal."""
    _run_test(
        q_lens=[128, 96],
        valid_q_lens=[128, 96],
        ctx_lens=[0, 64],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=32,
        dllm_block_size=32,
        is_block_causal=True,
        is_prefix_full=True,
        statuses=[0, 0],
        prefix_lens=[0, 0],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefix_full_covers_entire_seq():
    """Prefix covers the whole sequence: degenerates to full bidirectional attention."""
    _run_test(
        q_lens=[128, 96],
        valid_q_lens=[128, 96],
        ctx_lens=[0, 64],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=32,
        dllm_block_size=32,
        is_block_causal=True,
        is_prefix_full=True,
        statuses=[0, 0],
        prefix_lens=[128, 96],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefix_full_dbs64():
    """Prefix-full with dllm_block_size=64."""
    _run_test(
        q_lens=[128, 192],
        valid_q_lens=[128, 192],
        ctx_lens=[0, 64],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=64,
        dllm_block_size=64,
        is_block_causal=True,
        is_prefix_full=True,
        statuses=[0, 0],
        prefix_lens=[30, 80],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "q_lens,valid_q_lens,ctx_lens,statuses,prefix_lens",
    [
        # Mixed prefill/decode, unaligned prefix.
        ([256, 128, 128, 128], [256, 128, 96, 64], [0, 96, 256, 128], [0, 0, 1, 1], [48, 30, 0, 0]),
        # All prefill, diverse prefix lengths.
        ([128, 128, 64, 192], [128, 128, 64, 192], [32, 0, 128, 64], [0, 0, 0, 0], [64, 17, 32, 100]),
        # Mixed with deeper decode cache.
        ([320, 160, 128, 128], [320, 160, 96, 32], [64, 0, 256, 128], [0, 0, 1, 1], [55, 40, 0, 0]),
    ],
)
def test_prefix_full_extended_strategies(
    q_lens,
    valid_q_lens,
    ctx_lens,
    statuses,
    prefix_lens,
):
    """Extended prefix-full coverage with varied configurations."""
    _run_test(
        q_lens=q_lens,
        valid_q_lens=valid_q_lens,
        ctx_lens=ctx_lens,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=32,
        dllm_block_size=32,
        is_block_causal=True,
        is_prefix_full=True,
        statuses=statuses,
        prefix_lens=prefix_lens,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_chunked_prefill_multiround_cache_growth():
    """Multi-round decode coverage with cache growth across rounds."""
    torch.manual_seed(123)

    device = "cuda"
    dtype = torch.bfloat16
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 32
    dllm_block_size = 32
    q_len = 128
    scale = 1.0 / head_dim**0.5

    # Start with 2 pages of cache and grow by storing leftmost tokens from each round.
    total_pages = 16
    ctx_len = 64
    next_free_page = ctx_len // page_size

    k_cache = torch.randn(total_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_cache = torch.randn_like(k_cache)
    page_tables = torch.full((1, total_pages), -1, dtype=torch.int32, device=device)
    for i in range(next_free_page):
        page_tables[0, i] = i

    status_table = torch.tensor([1], dtype=torch.int32, device=device)
    prefix_lens = torch.tensor([0], dtype=torch.int32, device=device)
    padded_prefix_lens = torch.tensor([0], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)

    rounds = [
        (128, 1),  # full valid chunk, store 1 block
        (96, 1),   # partial valid chunk, store 1 block
        (128, 2),  # full valid chunk, store 2 blocks
        (64, 1),   # shorter valid chunk after deeper cache
    ]

    for round_id, (valid_q_len, blocks_to_store) in enumerate(rounds):
        q = torch.randn(q_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(q_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(q_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        valid_slices = torch.tensor([valid_q_len], dtype=torch.int32, device=device)
        context_lens = torch.tensor([ctx_len], dtype=torch.int32, device=device)

        active_pages = max(1, (ctx_len + page_size - 1) // page_size)
        out = call_chunked_prefill_kernel(
            q,
            k,
            v,
            k_cache,
            v_cache,
            page_tables[:, :active_pages],
            status_table,
            context_lens,
            cu_seqlens_q,
            valid_slices,
            prefix_lens,
            padded_prefix_lens,
            scale,
            dllm_block_size,
            is_block_causal=True,
        )
        ref = naive_chunked_prefill_ref(
            q,
            k,
            v,
            k_cache,
            v_cache,
            page_tables[:, :active_pages],
            [1],
            context_lens,
            cu_seqlens_q,
            valid_slices,
            [0],
            [0],
            scale,
            page_size,
            dllm_block_size,
            is_block_causal=True,
        )
        torch.testing.assert_close(
            out,
            ref,
            atol=5e-3,
            rtol=5e-3,
            msg=f"round {round_id} failed before cache update",
        )

        store_tokens = blocks_to_store * dllm_block_size
        assert store_tokens <= valid_q_len
        slot_mapping = torch.full((q_len,), -1, dtype=torch.int32, device=device)
        slot_start = next_free_page * page_size
        slot_mapping[:store_tokens] = torch.arange(
            slot_start,
            slot_start + store_tokens,
            dtype=torch.int32,
            device=device,
        )
        store_kv_cache_unified(k, v, k_cache, v_cache, slot_mapping)

        pages_added = store_tokens // page_size
        current_pages = active_pages
        for i in range(pages_added):
            page_tables[0, current_pages + i] = next_free_page + i
        next_free_page += pages_added
        ctx_len += store_tokens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_visualize_mask_example():
    """Example: visualize mask for debugging."""
    _run_test(
        q_lens=[128, 96],
        valid_q_lens=[64, 32],
        ctx_lens=[64, 128],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=32,
        dllm_block_size=32,
        is_block_causal=True,
        visualize_mask=True,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
