import pytest
import torch
import torch.nn.functional as F

from einops import rearrange
from types import SimpleNamespace

from diffulex_kernel.python.paged_attn_decode_triton import paged_attn_decode_unified_triton


def _has_fp8() -> bool:
    return hasattr(torch, "float8_e4m3fn") or hasattr(torch, "float8_e4m3fnuz") or hasattr(torch, "float8_e5m2")


def _build_cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
    # lengths: [num_seqs] int32 on cuda
    return torch.tensor(
        [0] + list(torch.cumsum(lengths, dim=0).cpu().numpy()),
        dtype=torch.int32,
        device=lengths.device,
    )


def naive_sdpa_with_kvcache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scale: float,
    num_groups: int,
    page_block_size: int,
) -> torch.Tensor:
    num_seqs = len(cu_seqlens_q) - 1
    output = torch.zeros_like(q)
    for seq_idx in range(num_seqs):
        q_start = int(cu_seqlens_q[seq_idx].item())
        q_end = int(cu_seqlens_q[seq_idx + 1].item())
        kv_start = int(cu_seqlens_k[seq_idx].item())
        kv_end = int(cu_seqlens_k[seq_idx + 1].item())

        q_seq = q[q_start:q_end]  # [q_len, Hq, D]
        k_seq = k[kv_start:kv_end]  # [new_len, Hkv, D]
        v_seq = v[kv_start:kv_end]

        ctx = int(context_lens[seq_idx].item())
        k_cache_seq_list = []
        v_cache_seq_list = []
        for blk in range(block_tables.shape[1]):
            page = int(block_tables[seq_idx, blk].item())
            if page < 0:
                continue
            blk_start = blk * page_block_size
            if blk_start >= ctx:
                continue
            blk_end = min(blk_start + page_block_size, ctx)
            n = blk_end - blk_start
            k_cache_seq_list.append(k_cache[page, :n])
            v_cache_seq_list.append(v_cache[page, :n])

        if k_cache_seq_list:
            k_ctx = torch.cat(k_cache_seq_list, dim=0)
            v_ctx = torch.cat(v_cache_seq_list, dim=0)
            k_comb = torch.cat([k_ctx, k_seq], dim=0)
            v_comb = torch.cat([v_ctx, v_seq], dim=0)
        else:
            k_comb = k_seq
            v_comb = v_seq

        q_sdpa = rearrange(q_seq, "s h d -> 1 h s d")
        k_sdpa = rearrange(k_comb, "s h d -> 1 h s d")
        v_sdpa = rearrange(v_comb, "s h d -> 1 h s d")
        attn_out = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            enable_gqa=True,
        )
        output[q_start:q_end] = rearrange(attn_out, "1 h s d -> s h d").to(output.dtype)

    return output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton paged-attention kernel")
def test_paged_decode_triton_bf16_cache_matches_reference():
    torch.manual_seed(0)
    device = torch.device("cuda")

    num_seqs = 4
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 32
    diffusion_block_size = 32

    num_groups = num_heads // num_kv_heads

    # Per-seq query/new KV length (decode step)
    q_lens = torch.full((num_seqs,), diffusion_block_size, dtype=torch.int32, device=device)
    cu_q = _build_cu_seqlens(q_lens)
    cu_k = cu_q.clone()
    total_q = int(cu_q[-1].item())

    # Context lengths (vary per seq)
    context_lens = torch.tensor([0, 17, 63, 128], dtype=torch.int32, device=device)
    max_ctx = int(context_lens.max().item())
    max_seq_blocks = (max_ctx + page_size - 1) // page_size
    num_page_blocks = num_seqs * max_seq_blocks

    # Assign each seq its own contiguous pages
    block_tables = torch.full((num_seqs, max_seq_blocks), -1, dtype=torch.int32, device=device)
    for s in range(num_seqs):
        for b in range(max_seq_blocks):
            block_tables[s, b] = s * max_seq_blocks + b

    q = torch.randn((total_q, num_heads, head_dim), device=device, dtype=torch.bfloat16)
    k = torch.randn((total_q, num_kv_heads, head_dim), device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)

    k_cache = torch.randn((num_page_blocks, page_size, num_kv_heads, head_dim), device=device, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)

    md = SimpleNamespace(
        kv_cache_layout="unified",
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_q,
        max_seqlen_q=int(q_lens.max().item()),
        page_block_size=page_size,
    )
    scale = 1.0 / (head_dim**0.5)

    out = paged_attn_decode_unified_triton(q, k, v, k_cache, v_cache, md, softmax_scale=scale, fp8_cache=False)
    ref = naive_sdpa_with_kvcache(
        q,
        k,
        v,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        cu_q,
        cu_k,
        scale,
        num_groups,
        page_size,
    )

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton paged-attention kernel")
@pytest.mark.skipif(not _has_fp8(), reason="This torch build does not expose FP8 dtypes")
def test_paged_decode_triton_fp8_cache_matches_reference():
    torch.manual_seed(0)
    device = torch.device("cuda")

    fp8_dtype = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.float8_e5m2

    num_seqs = 2
    num_heads = 16
    num_kv_heads = 4
    head_dim = 128
    page_size = 32
    diffusion_block_size = 32
    num_groups = num_heads // num_kv_heads

    q_lens = torch.full((num_seqs,), diffusion_block_size, dtype=torch.int32, device=device)
    cu_q = _build_cu_seqlens(q_lens)
    cu_k = cu_q.clone()
    total_q = int(cu_q[-1].item())

    context_lens = torch.tensor([37, 55], dtype=torch.int32, device=device)
    max_ctx = int(context_lens.max().item())
    max_seq_blocks = (max_ctx + page_size - 1) // page_size
    num_page_blocks = num_seqs * max_seq_blocks
    block_tables = torch.full((num_seqs, max_seq_blocks), -1, dtype=torch.int32, device=device)
    for s in range(num_seqs):
        for b in range(max_seq_blocks):
            block_tables[s, b] = s * max_seq_blocks + b

    q = torch.randn((total_q, num_heads, head_dim), device=device, dtype=torch.bfloat16)
    k = torch.randn((total_q, num_kv_heads, head_dim), device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)

    # Build BF16 "true" cache values, then quantize to FP8 as (x / scale) -> fp8, with per-head scales.
    k_cache_true = torch.randn((num_page_blocks, page_size, num_kv_heads, head_dim), device=device, dtype=torch.bfloat16) * 0.5
    v_cache_true = torch.randn_like(k_cache_true) * 0.5

    eps = 1e-6
    k_absmax = k_cache_true.to(torch.float32).abs().amax(dim=(0, 1, 3))
    v_absmax = v_cache_true.to(torch.float32).abs().amax(dim=(0, 1, 3))
    fp8_max = 448.0 if fp8_dtype == torch.float8_e4m3fn else 57344.0
    k_scale = (k_absmax / fp8_max).clamp_min(eps).to(torch.float32)
    v_scale = (v_absmax / fp8_max).clamp_min(eps).to(torch.float32)

    k_cache_fp8 = (k_cache_true.to(torch.float32) / k_scale.view(1, 1, -1, 1)).to(fp8_dtype)
    v_cache_fp8 = (v_cache_true.to(torch.float32) / v_scale.view(1, 1, -1, 1)).to(fp8_dtype)

    md = SimpleNamespace(
        kv_cache_layout="unified",
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_q,
        max_seqlen_q=int(q_lens.max().item()),
        page_block_size=page_size,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    scale = 1.0 / (head_dim**0.5)

    out = paged_attn_decode_unified_triton(q, k, v, k_cache_fp8, v_cache_fp8, md, softmax_scale=scale, fp8_cache=True)

    # Reference uses dequantized cache.
    k_cache_deq = (k_cache_fp8.float() * k_scale.view(1, 1, -1, 1)).to(torch.bfloat16)
    v_cache_deq = (v_cache_fp8.float() * v_scale.view(1, 1, -1, 1)).to(torch.bfloat16)
    ref = naive_sdpa_with_kvcache(
        q,
        k,
        v,
        k_cache_deq,
        v_cache_deq,
        block_tables,
        context_lens,
        cu_q,
        cu_k,
        scale,
        num_groups,
        page_size,
    )

    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

