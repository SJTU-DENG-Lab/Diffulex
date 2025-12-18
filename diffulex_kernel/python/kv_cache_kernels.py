import torch
import triton 

import triton.language as tl

from typing import Tuple

from diffulex.attention.metadata import AttnMetaDataBase
    

@triton.jit
def dllm_store_kvcache_kernel_unified(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr
):
    token_idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    key_offsets = token_idx * key_stride + tl.arange(0, D)
    value_offsets = token_idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


@triton.jit
def dllm_store_kvcache_kernel_distinct(
    k_ptr, v_ptr, k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
    k_stride, v_stride,  
    k_cache_stride_nblks, k_cache_stride_h, k_cache_stride_dx, k_cache_stride_blk_sz, k_cache_stride_x,
    v_cache_stride_nblks, v_cache_stride_h, v_cache_stride_d, v_cache_stride_blk_sz,
    nheads, hdim, blk_sz,
    x: tl.constexpr, D: tl.constexpr
):  
    # SPDX-License-Identifier: Apache-2.0
    # SPDX-FileCopyrightText: D2F

    # Translated from vLLM's CUDA kernel 
    # Referencing https://github.com/vllm-project/vllm/blob/main/csrc/cache_kernels.cu#L212
    # and https://github.com/vllm-project/vllm/blob/main/csrc/cache_kernels.cu#L415
    
    # Organization: SJTU DENG Lab
    # Author: Drew Jin (JIN. Yijie, @drewjin)
    # Date: 2025-08-03
    # Email: drewjin0827@gmail.com
    # All rights reserved.
    
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx < 0:
        return
    
    blk_idx = slot_idx // blk_sz
    off_blk = slot_idx % blk_sz
    
    offs_d = tl.arange(0, D)
    offs_k = token_idx * k_stride + offs_d
    offs_v = token_idx * v_stride + offs_d
    k = tl.load(k_ptr + offs_k)
    v = tl.load(v_ptr + offs_v)

    h_ids = offs_d // hdim
    h_offs = offs_d % hdim
    x_ids = h_offs // x
    x_offs = h_offs % x
    
    k_cache_offs = (blk_idx * k_cache_stride_nblks + h_ids * k_cache_stride_h +
                    x_ids * k_cache_stride_dx + off_blk * k_cache_stride_blk_sz + 
                    x_offs * k_cache_stride_x)
    v_cache_offs = (blk_idx * v_cache_stride_nblks + h_ids * v_cache_stride_h +
                    h_offs * v_cache_stride_d + off_blk * v_cache_stride_blk_sz)
    
    tl.store(k_cache_ptr + k_cache_offs, k)
    tl.store(v_cache_ptr + v_cache_offs, v)
    

def store_kvcache_distinct_layout(key: torch.Tensor, value: torch.Tensor, 
                                  k_cache: torch.Tensor, v_cache: torch.Tensor, 
                                  slot_mapping: torch.Tensor, attn_metadata: AttnMetaDataBase) -> None:
    # TODO: implement diffusion lm kv cache store
    # k_cache: [num_blks, h, hdim // x, blk_sz, x]
    # v_cache: [num_blks, h, hdim, blk_sz]
    NBlks, NHeads, HDim_x, Blk_sz, x = k_cache.shape
    HDim = HDim_x * x
    N = key.shape[0]
    assert HDim == key.shape[-1] and NHeads == key.shape[1]
    assert N == slot_mapping.numel()
    
    GRID = (N, )
    dllm_store_kvcache_kernel_distinct[GRID](
        key, value,
        k_cache, v_cache,
        slot_mapping,
        key.stride(0), value.stride(0), 
        *k_cache.stride(), *v_cache.stride(),
        NHeads, HDim, Blk_sz,
        x, HDim * NHeads
    )


def store_kvcache_unified_layout(key: torch.Tensor, value: torch.Tensor, 
                                 k_cache: torch.Tensor, v_cache: torch.Tensor, 
                                 slot_mapping: torch.Tensor, attn_metadata: AttnMetaDataBase) -> None:
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert N == slot_mapping.numel(), f"`N`: {N}, `slot_mapping.numel()`: {slot_mapping.numel()}"
    
    dllm_store_kvcache_kernel_unified[(N,)](
        key, key.stride(0),
        value, value.stride(0),
        k_cache, v_cache, slot_mapping, D
    )
        

@triton.jit
def load_kvcache_kernel(k_cache_ptr, v_cache_ptr,
                        k_new_ptr, v_new_ptr,
                        block_table_ptr,
                        k_out_ptr, v_out_ptr, 
                        seqlens_ptr, ctxlens_ptr,
                        cu_seqlens_q_ptr, cu_seqlens_k_ptr,
                        kv_cache_stride_nblks, kv_cache_stride_blk, kv_cache_stride_h, kv_cache_stride_d,
                        kv_new_stride_s, kv_new_stride_h, kv_new_stride_d,
                        block_table_stride_nseqs, block_table_stride_maxblks,
                        kv_out_stride_s, kv_out_stride_h, kv_out_stride_d,
                        ctxlens_stride, seqlens_stride,
                        cu_seqlens_q_stride, cu_seqlens_k_stride,
                        LAST_BLK_ID: tl.constexpr,
                        HEAD_DIM: tl.constexpr,
                        PAGE_SIZE: tl.constexpr,
                        DIFFUSION_BLOCK_SIZE: tl.constexpr,
                        KV_LOAD_UNROLL_FACTOR: tl.constexpr):
    # BUG FIX
    # SPDX-License-Identifier: Apache-2.0
    # SPDX-FileCopyrightText: D2F
    
    # Organization: SJTU DENG Lab
    # Author: Drew Jin (JIN. Yijie, @drewjin)
    # Date: 2025-08-01
    # Email: drewjin0827@gmail.com
    # All rights reserved.
    
    seq_idx = tl.program_id(0)
    local_blk_idx = tl.program_id(1)
    kv_head_idx = tl.program_id(2)

    off_local_blk = seq_idx * block_table_stride_nseqs + local_blk_idx * block_table_stride_maxblks
    global_blk_idx = tl.load(block_table_ptr + off_local_blk)
    
    if global_blk_idx != -1:
        off_ctxlen = seq_idx * ctxlens_stride
        global_ctxlen = tl.load(ctxlens_ptr + off_ctxlen)
        cur_window_sz = (local_blk_idx + 1) * PAGE_SIZE
        prev_window_sz = local_blk_idx * PAGE_SIZE
        local_ctxlen = tl.where(global_ctxlen > cur_window_sz, PAGE_SIZE, global_ctxlen % PAGE_SIZE)
        if global_ctxlen > prev_window_sz:
            # Load KV cache
            offs_kv_cache_seq = tl.arange(0, PAGE_SIZE)
            offs_kv_cache_hdim = tl.arange(0, HEAD_DIM)
            offs_kv_cache = ( # [NBlks, BlkSz, Hkv, Hdim]
                global_blk_idx[None, :] * kv_cache_stride_nblks + # NBlks: BlkId
                offs_kv_cache_seq[None, :] * kv_cache_stride_blk + # BlkSz: TokenIds
                kv_head_idx * kv_cache_stride_h + # Hkv: HeadId
                offs_kv_cache_hdim[:, None] * kv_cache_stride_d # Hdim: HeadDim Elems
            )
            kv_cache_mask = offs_kv_cache_seq[None, :] < local_ctxlen
            k_cache = tl.load(k_cache_ptr + offs_kv_cache, mask=kv_cache_mask, other=0.0)
            v_cache = tl.load(v_cache_ptr + offs_kv_cache, mask=kv_cache_mask, other=0.0)
            
            # Store KV cache into output KV tensors
            off_cu_seqlens_k = seq_idx * cu_seqlens_k_stride
            kv_out_start_idx = tl.load(cu_seqlens_k_ptr + off_cu_seqlens_k)
            cur_kv_cache_to_out_start_idx = kv_out_start_idx + prev_window_sz
            offs_kv_cache_to_out = ( # [Seq, Hkv, Hdim]
                (cur_kv_cache_to_out_start_idx + offs_kv_cache_seq[None, :]) * kv_out_stride_s + # Seq: TokenIds over Offset
                kv_head_idx * kv_out_stride_h + # Hkv: HeadId
                offs_kv_cache_hdim[:, None] * kv_out_stride_d # Hdim: HeadDim Elems
            )
            tl.store(k_out_ptr + offs_kv_cache_to_out, k_cache, mask=kv_cache_mask)
            tl.store(v_out_ptr + offs_kv_cache_to_out, v_cache, mask=kv_cache_mask)

    # Load and store active KV only once when first meet
    if local_blk_idx == LAST_BLK_ID: 
        # Load KV new
        off_cu_seqlens_q = seq_idx * cu_seqlens_q_stride
        off_seqlens = seq_idx * seqlens_stride
        kv_new_start_idx = tl.load(cu_seqlens_q_ptr + off_cu_seqlens_q)
        active_seqlen = tl.load(seqlens_ptr + off_seqlens)
        offs_kv_new_seq = tl.arange(0, DIFFUSION_BLOCK_SIZE)
        offs_kv_new_hdim = tl.arange(0, HEAD_DIM)
        
        for diff_blk_idx in tl.range(active_seqlen // DIFFUSION_BLOCK_SIZE, loop_unroll_factor=KV_LOAD_UNROLL_FACTOR):
            off_diff_blk = diff_blk_idx * DIFFUSION_BLOCK_SIZE
            cur_kv_new_start_idx = kv_new_start_idx + off_diff_blk
            offs_cur_kv_new_seq = ( # [Seq, Hkv, Hdim]
                (cur_kv_new_start_idx + offs_kv_new_seq[None, :]) * kv_new_stride_s + # Seq: TokenIds over Offset
                kv_head_idx * kv_new_stride_h + # Hkv: HeadId
                offs_kv_new_hdim[:, None] * kv_new_stride_d # Hdim: HeadDim Elems
            )
            k_new = tl.load(k_new_ptr + offs_cur_kv_new_seq)
            v_new = tl.load(v_new_ptr + offs_cur_kv_new_seq)

            # Store KV new into output KV tensors
            off_ctxlen = seq_idx * ctxlens_stride
            off_cu_seqlens_k = seq_idx * cu_seqlens_k_stride
            global_ctxlen = tl.load(ctxlens_ptr + off_ctxlen)
            kv_out_start_idx = tl.load(cu_seqlens_k_ptr + off_cu_seqlens_k)
            cur_kv_new_to_out_start_idx = global_ctxlen + kv_out_start_idx + off_diff_blk
            offs_cur_kv_new_to_out = ( # [Seq, Hkv, Hdim]
                (cur_kv_new_to_out_start_idx + offs_kv_new_seq[None, :]) * kv_out_stride_s + # Seq: TokenIds over Offset
                kv_head_idx * kv_out_stride_h + # Hkv: HeadId
                offs_kv_new_hdim[:, None] * kv_out_stride_d # Hdim: HeadDim Elems
            )
            tl.store(k_out_ptr + offs_cur_kv_new_to_out, k_new)
            tl.store(v_out_ptr + offs_cur_kv_new_to_out, v_new)


def load_kvcache(k_cache: torch.Tensor, v_cache: torch.Tensor,
                 attn_metadata: AttnMetaDataBase,
                 k_new: torch.Tensor, v_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert k_cache.shape == v_cache.shape
    assert k_new.shape == v_new.shape
    N_BLOCKS, PAGE_SIZE, H_KV, HEAD_DIM = k_cache.shape
    NUM_SEQS, MAX_SEQ_BLOCKS = attn_metadata.block_tables.shape
    
    ctxlens = attn_metadata.context_lens
    seqlens = attn_metadata.seq_lens_ts
    assert sum(seqlens) == k_new.shape[0]
    DIFFUSION_BLOCK_SIZE = attn_metadata.seqs[0].diffusion_block_size
    MAX_DIFFUSION_BLOCK_SIZE = max(seqlens)
    assert MAX_DIFFUSION_BLOCK_SIZE % DIFFUSION_BLOCK_SIZE == 0
    
    total_lens = ctxlens + seqlens
    cu_seqlens_q = attn_metadata.cu_seqlens_q
    cu_seqlens_k = attn_metadata.cu_seqlens_k
    assert sum(total_lens) == cu_seqlens_k[-1]
    assert cu_seqlens_q.shape == cu_seqlens_k.shape
    assert cu_seqlens_q.shape[0] == NUM_SEQS + 1
    
    kv_output_shape = (sum(total_lens).item(), H_KV, HEAD_DIM)
    k_output = torch.empty(kv_output_shape, device=k_cache.device, dtype=k_cache.dtype)
    v_output = torch.empty_like(k_output)
    
    GRID = (NUM_SEQS, MAX_SEQ_BLOCKS, H_KV)
    load_kvcache_kernel[GRID](
        k_cache, v_cache,
        k_new, v_new,
        attn_metadata.block_tables,
        k_output, v_output,
        seqlens, ctxlens,
        cu_seqlens_q, cu_seqlens_k,
        *k_cache.stride(),
        *k_new.stride(),
        *attn_metadata.block_tables.stride(),
        *k_output.stride(),
        ctxlens.stride(0),
        seqlens.stride(0),
        cu_seqlens_q.stride(0),
        cu_seqlens_k.stride(0),
        LAST_BLK_ID=attn_metadata.block_tables.shape[-1] - 1,
        HEAD_DIM=HEAD_DIM,
        PAGE_SIZE=PAGE_SIZE,
        DIFFUSION_BLOCK_SIZE=DIFFUSION_BLOCK_SIZE,
        KV_LOAD_UNROLL_FACTOR=2
    )
    
    return k_output, v_output