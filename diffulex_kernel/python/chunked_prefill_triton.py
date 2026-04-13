import os
import torch
import triton
import triton.language as tl

from diffulex.attention.metadata import AttnMetaDataBase
from diffulex_kernel.python.auto_tuner import build_chunked_prefill_configs

DISABLE_CHUNKED_PREFILL_AUTOTUNE = os.environ.get("DIFFULEX_DISABLE_CHUNKED_PREFILL_AUTOTUNE", "0") == "1"


def _prune_chunked_prefill_configs(configs, _nargs, **kwargs):
    """Keep autotune numerically stable across page sizes for small decode blocks.

    For small DLLM blocks, varying the cache-stage tile shape changes the softmax
    reduction order enough to flip greedy decoding across page sizes. Constrain
    those cases to the stable launch shape; keep the full search space for larger
    blocks where we haven't observed correctness regressions.
    """
    dllm_block_size = kwargs.get("DLLM_BLOCK_SIZE")
    if dllm_block_size is None or dllm_block_size > 16:
        return configs

    stable_configs = [
        config
        for config in configs
        if config.kwargs.get("BLOCK_M") == 64
        and config.kwargs.get("BLOCK_N") == 64
        and config.num_warps == 4
        and config.num_stages == 2
    ]
    return stable_configs or configs


def maybe_autotune(fn):
    fn = triton.jit(fn)
    # NOTE: Tests pass explicit BLOCK_M / BLOCK_N meta-parameters, which conflicts with
    # Triton autotune. Disable autotune under test/debug via env instead of editing code.
    if DISABLE_CHUNKED_PREFILL_AUTOTUNE:
        return fn
    return triton.autotune(
        configs=[
            triton.Config(c, num_warps=c.pop("num_warps"), num_stages=c.pop("num_stages"))
            for c in build_chunked_prefill_configs()
        ],
        key=[
            "NUM_GROUPS",
            "HEAD_DIM",
            "PAGE_SIZE",
            "DLLM_BLOCK_SIZE",
            "IS_BLOCK_CAUSAL",
            "IS_PREFIX_FULL",
        ],
        prune_configs_by={"early_config_prune": _prune_chunked_prefill_configs},
    )(fn)

@maybe_autotune
def _chunked_prefill_attn_unified_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_tables_ptr,
    status_table_ptr,
    context_lens_ptr,
    cu_seqlens_q_ptr,
    valid_slices_ptr,
    prefix_lens_ptr,
    padded_prefix_lens_ptr,
    softmax_scale,  # fp32 scalar
    # q/k/v/o strides
    q_stride_s,
    q_stride_h,
    q_stride_d,
    kv_stride_s,
    kv_stride_h,
    kv_stride_d,
    o_stride_s,
    o_stride_h,
    o_stride_d,
    # cache strides: [npages, psz, kvh, d]
    k_cache_stride_npages,
    k_cache_stride_psz,
    k_cache_stride_h,
    k_cache_stride_d,
    v_cache_stride_npages,
    v_cache_stride_psz,
    v_cache_stride_h,
    v_cache_stride_d,
    # page_tables strides
    page_tables_stride_nreqs,
    page_tables_stride_pages,
    # misc
    NUM_GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DLLM_BLOCK_SIZE: tl.constexpr,
    IS_BLOCK_CAUSAL: tl.constexpr,
    IS_PREFIX_FULL: tl.constexpr,
):
    req_id = tl.program_id(0)
    head_id = tl.program_id(1)
    q_block_id = tl.program_id(2)

    kv_head_id = head_id // NUM_GROUPS

    status = tl.load(status_table_ptr + req_id).to(tl.int32)
    context_len = tl.load(context_lens_ptr + req_id).to(tl.int32)
    q_start = tl.load(cu_seqlens_q_ptr + req_id).to(tl.int32)
    q_end = tl.load(cu_seqlens_q_ptr + req_id + 1).to(tl.int32)
    valid_slice = tl.load(valid_slices_ptr + req_id).to(tl.int32)
    prefix_len = tl.load(prefix_lens_ptr + req_id).to(tl.int32)
    padded_prefix_len = tl.load(padded_prefix_lens_ptr + req_id).to(tl.int32)

    q_len = q_end - q_start
    valid_q_len = valid_slice - q_start
    valid_kv_len = valid_q_len
    new_len = q_len

    offs_q_block = q_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM_PADDED)
    mask_q_block = offs_q_block < valid_q_len
    mask_d = offs_d < HEAD_DIM

    offs_q = (q_start + offs_q_block[:, None]) * q_stride_s + head_id * q_stride_h + offs_d[None, :] * q_stride_d
    q = tl.load(q_ptr + offs_q, mask=mask_q_block[:, None] & mask_d[None, :], other=0.0).to(tl.bfloat16)

    m = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM_PADDED], dtype=tl.float32)

    # Stage 1: Attention against KV Cache.
    full_cache_range = tl.cdiv(context_len, BLOCK_N)
    for cache_block_id in range(0, full_cache_range):
        offs_kv_cache_block = cache_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_cache_valid_map = offs_kv_cache_block < context_len
        page_rel_ids = offs_kv_cache_block // PAGE_SIZE
        page_abs_ids = tl.load(
            page_tables_ptr + req_id * page_tables_stride_nreqs + page_rel_ids * page_tables_stride_pages,
            mask=kv_cache_valid_map,
            other=-1,
        ).to(tl.int32)
        page_offs = offs_kv_cache_block % PAGE_SIZE
        page_token_valid_map = kv_cache_valid_map & (page_abs_ids >= 0)

        k_offs = (
            page_abs_ids[:, None] * k_cache_stride_npages
            + page_offs[:, None] * k_cache_stride_psz
            + kv_head_id * k_cache_stride_h
            + offs_d[None, :] * k_cache_stride_d
        )
        k = tl.load(
            k_cache_ptr + k_offs,
            mask=page_token_valid_map[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * softmax_scale
        scores = tl.where(mask_q_block[:, None] & page_token_valid_map[None, :], scores, float("-inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        l_new = l * tl.exp(m - m_new) + tl.sum(p, axis=1)
        alpha = tl.exp(m - m_new)
        acc *= alpha[:, None]

        v_offs = (
            page_abs_ids[:, None] * v_cache_stride_npages
            + page_offs[:, None] * v_cache_stride_psz
            + kv_head_id * v_cache_stride_h
            + offs_d[None, :] * v_cache_stride_d
        )
        v = tl.load(
            v_cache_ptr + v_offs,
            mask=page_token_valid_map[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        acc += tl.dot(p.to(tl.bfloat16), v).to(tl.float32)
        m = m_new
        l = l_new

    # Stage 2: Attention against new KV
    kv_start = q_start
    full_range = tl.cdiv(valid_kv_len, BLOCK_N)

    abs_q_block = context_len + offs_q_block

    max_q_idx_in_chunk = tl.minimum(valid_q_len - 1, (q_block_id + 1) * BLOCK_M - 1)
    max_abs_q_idx = context_len + max_q_idx_in_chunk
    max_abs_kv_idx = ((max_abs_q_idx // DLLM_BLOCK_SIZE) + 1) * DLLM_BLOCK_SIZE
    max_rel_kv_len = max_abs_kv_idx - context_len

    block_causal_range = tl.minimum(
        tl.maximum(0, tl.cdiv(max_rel_kv_len, BLOCK_N)),
        full_range,
    )

    if IS_BLOCK_CAUSAL and not IS_PREFIX_FULL:
        loop_range = block_causal_range
    elif IS_BLOCK_CAUSAL and IS_PREFIX_FULL:
        is_prefilling = status == 0
        if is_prefilling:
            loop_range = full_range
        else:
            loop_range = block_causal_range
    else:
        loop_range = full_range

    for kv_block_id in range(0, loop_range):
        kv_block_start = kv_block_id * BLOCK_N
        offs_kv_block = kv_block_start + tl.arange(0, BLOCK_N)
        abs_kv_block = context_len + offs_kv_block
        kv_token_valid_map = (offs_kv_block < new_len) & (offs_kv_block < valid_q_len)

        k_offs = (
            (kv_start + offs_kv_block[None, :]) * kv_stride_s + kv_head_id * kv_stride_h + offs_d[:, None] * kv_stride_d
        )
        k = tl.load(
            k_ptr + k_offs,
            mask=kv_token_valid_map[None, :] & mask_d[:, None],
            other=0.0,
        ).to(tl.bfloat16)

        scores = tl.dot(q, k).to(tl.float32) * softmax_scale
        score_valid_mask = mask_q_block[:, None] & kv_token_valid_map[None, :]
        if IS_BLOCK_CAUSAL and not IS_PREFIX_FULL:
            score_block_mask = ((abs_q_block // DLLM_BLOCK_SIZE + 1) * DLLM_BLOCK_SIZE)[:, None] > abs_kv_block[
                None, :
            ]
            score_mask = score_valid_mask & score_block_mask
        elif IS_BLOCK_CAUSAL and IS_PREFIX_FULL:
            if is_prefilling:
                score_pure_prefix_mask = (offs_q_block < prefix_len)[:, None] & (offs_kv_block < prefix_len)[None, :]
                score_padded_causal_mask = ((offs_q_block >= prefix_len) & (offs_q_block < padded_prefix_len))[
                    :, None
                ] & (offs_kv_block < padded_prefix_len)[None, :]
                score_block_mask = ((abs_q_block // DLLM_BLOCK_SIZE + 1) * DLLM_BLOCK_SIZE)[:, None] > abs_kv_block[
                    None, :
                ]
                score_block_mask_extend_only = score_block_mask & (offs_q_block >= padded_prefix_len)[:, None]
                score_mask = score_valid_mask & (score_pure_prefix_mask | score_padded_causal_mask | score_block_mask_extend_only)
            else:
                score_block_mask = ((abs_q_block // DLLM_BLOCK_SIZE + 1) * DLLM_BLOCK_SIZE)[:, None] > abs_kv_block[
                    None, :
                ]
                score_mask = score_valid_mask & score_block_mask
        else:
            score_mask = score_valid_mask
        scores = tl.where(score_mask, scores, float("-inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        l_new = l * tl.exp(m - m_new) + tl.sum(p, axis=1)
        alpha = tl.exp(m - m_new)
        acc *= alpha[:, None]

        v_offs = (
            (kv_start + offs_kv_block[:, None]) * kv_stride_s + kv_head_id * kv_stride_h + offs_d[None, :] * kv_stride_d
        )
        v = tl.load(
            v_ptr + v_offs,
            mask=kv_token_valid_map[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        acc += tl.dot(p.to(tl.bfloat16), v).to(tl.float32)
        m = m_new
        l = l_new

    out = acc / l[:, None]
    o_offs = (q_start + offs_q_block[:, None]) * o_stride_s + head_id * o_stride_h + offs_d[None, :] * o_stride_d
    tl.store(
        o_ptr + o_offs,
        out.to(tl.bfloat16),
        mask=mask_q_block[:, None] & mask_d[None, :],
    )


def _run_chunked_prefill_attn_unified_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    attn_metadata: AttnMetaDataBase,
):
    o = torch.empty_like(q).to(q.device)
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    num_groups = num_heads // num_kv_heads

    head_dim = q.shape[-1]
    head_dim_padded = 1 << (head_dim - 1).bit_length()
    softmax_scale = 1.0 / head_dim**0.5
    page_size = k_cache.shape[1]
    num_reqs = attn_metadata.cu_seqlens_q.shape[0] - 1

    if DISABLE_CHUNKED_PREFILL_AUTOTUNE:
        block_m = 64
        block_n = 64
        grid = (num_reqs, num_heads, triton.cdiv(int(attn_metadata.max_seqlen_q), block_m))
    else:
        block_m = None
        block_n = None
        grid = lambda meta: (num_reqs, num_heads, triton.cdiv(int(attn_metadata.max_seqlen_q), meta["BLOCK_M"]))

    launch_kwargs = {}
    if DISABLE_CHUNKED_PREFILL_AUTOTUNE:
        launch_kwargs["BLOCK_M"] = block_m
        launch_kwargs["BLOCK_N"] = block_n

    _chunked_prefill_attn_unified_kernel[grid](
        q,
        k,
        v,
        o,
        k_cache,
        v_cache,
        attn_metadata.page_tables,
        attn_metadata.status_table,
        attn_metadata.context_lens,
        attn_metadata.cu_seqlens_q,
        attn_metadata.valid_slices,
        attn_metadata.prefix_lens,
        attn_metadata.padded_prefix_lens,
        softmax_scale,
        *q.stride(),
        *k.stride(),
        *o.stride(),
        *k_cache.stride(),
        *v_cache.stride(),
        *attn_metadata.page_tables.stride(),
        NUM_GROUPS=num_groups,
        HEAD_DIM=head_dim,
        HEAD_DIM_PADDED=head_dim_padded,
        PAGE_SIZE=page_size,
        DLLM_BLOCK_SIZE=attn_metadata.block_size,
        IS_BLOCK_CAUSAL=attn_metadata.is_block_causal,
        IS_PREFIX_FULL=attn_metadata.is_prefix_full,
        **launch_kwargs,
    )
    return o


def chunked_prefill_attn_unified(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    attn_metadata: AttnMetaDataBase,
):
    return _run_chunked_prefill_attn_unified_kernel(q, k, v, k_cache, v_cache, attn_metadata)
