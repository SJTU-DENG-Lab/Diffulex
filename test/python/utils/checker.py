def CHECK_D2F_SLOT_MAPPING(seqs, slot_mapping):
    # check slot mapping layout
    start_idx = 0
    for seq in seqs:
        cur_ref_slot_mapping = []
        for idx in range(seq.num_diffusion_blocks):
            if seq.active_blocks[idx]:
                padding_num_tokens = (seq.num_diffusion_blocks - idx) * seq.diffusion_block_size
                cur_ref_slot_mapping.extend([-1] * padding_num_tokens)
                break
            elif seq.to_cache_blocks[idx]:
                cur_ref_slot_mapping.extend([0] * seq.diffusion_block_size)
        cur_slot_mapping = slot_mapping[start_idx:start_idx + len(cur_ref_slot_mapping)]
        for slot, ref_slot in zip(cur_slot_mapping, cur_ref_slot_mapping):
            try:
                if ref_slot == -1:
                    assert slot == -1
                elif ref_slot == 0:
                    assert slot != -1
                elif ref_slot is not None:
                    assert slot is not None
            except AssertionError:
                raise ValueError(f"Slot mapping mismatch: {slot} != {ref_slot}. "
                                    f"Check the implementation of prepare_decode.\n"
                                    f"slot_mapping: {cur_slot_mapping}\n"
                                    f"ref_slot_mapping: {cur_ref_slot_mapping}\n"
                                    f"diff: {[s - r for s, r in zip(cur_slot_mapping, cur_ref_slot_mapping)]}")
        start_idx += len(cur_ref_slot_mapping)


def CHECK_FLASH_ATTN_PREFILL(
    q, k, v, 
    cu_seqlens_q, 
    cu_seqlens_k, 
    max_seqlen_q, 
    prefill_kernel,
    diffusion_block_size: int = 32,
    is_block_attn: bool = False,
):
    """
    Verify prefill kernel correctness by comparing with PyTorch's scaled_dot_product_attention.
    
    Args:
        q: Query tensor [total_q_len, num_heads, head_dim]
        k: Key tensor [total_kv_len, num_kv_heads, head_dim]
        v: Value tensor [total_kv_len, num_kv_heads, head_dim]
        cu_seqlens_q: Cumulative sequence lengths for queries
        cu_seqlens_k: Cumulative sequence lengths for keys/values
        max_seqlen_q: Maximum sequence length for queries
        prefill_kernel: The kernel function to test
        diffusion_block_size: Size of diffusion blocks for block attention
        is_block_attn: Whether this is block attention mode
    """
    import torch
    import torch.nn.functional as F
    from einops import rearrange
    
    # Run kernel
    kernel_output = prefill_kernel(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q)
    
    # Compute reference output using PyTorch's SDPA
    head_dim = q.shape[2]
    scale = 1.0 / (head_dim ** 0.5)
    num_seqs = len(cu_seqlens_q) - 1
    
    gt_output = torch.zeros_like(q)
    for seq_idx in range(num_seqs):
        q_start = cu_seqlens_q[seq_idx].item()
        q_end = cu_seqlens_q[seq_idx + 1].item()
        kv_start = cu_seqlens_k[seq_idx].item()
        kv_end = cu_seqlens_k[seq_idx + 1].item()
        
        q_seq = q[q_start:q_end]
        k_seq = k[kv_start:kv_end]
        v_seq = v[kv_start:kv_end]
        
        q_len = q_seq.shape[0]
        kv_len = k_seq.shape[0]
        
        # Reshape for SDPA: [1, num_heads, seq_len, head_dim]
        q_sdpa = rearrange(q_seq, 's h d -> 1 h s d')
        k_sdpa = rearrange(k_seq, 's h d -> 1 h s d')
        v_sdpa = rearrange(v_seq, 's h d -> 1 h s d')
        
        if not is_block_attn:
            # Standard attention
            attn_out = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
                enable_gqa=True,
            )
        else:
            # Block attention with mask
            block_mask = torch.zeros((1, 1, q_len, kv_len), dtype=q.dtype, device=q.device).bool()
            num_diffusion_blocks = (kv_len + diffusion_block_size - 1) // diffusion_block_size
            for block_idx in range(num_diffusion_blocks):
                block_start = block_idx * diffusion_block_size
                block_end = min(block_start + diffusion_block_size, kv_len)
                block_mask[..., block_start:block_end, :block_end] = True
            
            attn_out = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=block_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
                enable_gqa=True,
            )
        
        gt_output[q_start:q_end] = rearrange(attn_out, '1 h s d -> s h d').to(gt_output.dtype)
    
    # Compare results
    atol = 1e-2
    rtol = 1e-2
    try:
        torch.testing.assert_close(
            kernel_output, 
            gt_output, 
            atol=atol, 
            rtol=rtol,
            msg="Kernel output does not match reference implementation"
        )
    except AssertionError as e:
        # Compute error statistics for debugging
        abs_diff = torch.abs(kernel_output - gt_output)
        max_diff = torch.max(abs_diff).item()
        mean_diff = torch.mean(abs_diff).item()
        rel_diff = torch.abs((kernel_output - gt_output) / (gt_output + 1e-8))
        max_rel_diff = torch.max(rel_diff).item()
        mean_rel_diff = torch.mean(rel_diff).item()
        
        # Count elements that exceed tolerance
        total_elements = kernel_output.numel()
        # Elements that exceed absolute tolerance
        exceeds_atol = (abs_diff > atol)
        num_exceeds_atol = exceeds_atol.sum().item()
        # Elements that exceed relative tolerance
        exceeds_rtol = (rel_diff > rtol)
        num_exceeds_rtol = exceeds_rtol.sum().item()
        # Elements that exceed either tolerance
        exceeds_tolerance = exceeds_atol | exceeds_rtol
        num_exceeds_tolerance = exceeds_tolerance.sum().item()
        pct_exceeds_tolerance = (num_exceeds_tolerance / total_elements * 100) if total_elements > 0 else 0
        
        raise AssertionError(
            f"Prefill kernel verification failed!\n"
            f"Max absolute difference: {max_diff:.6f}\n"
            f"Mean absolute difference: {mean_diff:.6f}\n"
            f"Max relative difference: {max_rel_diff:.6f}\n"
            f"Mean relative difference: {mean_rel_diff:.6f}\n"
            f"Total elements: {total_elements}\n"
            f"Elements exceeding absolute tolerance (atol={atol}): {num_exceeds_atol} ({num_exceeds_atol/total_elements*100:.2f}%)\n"
            f"Elements exceeding relative tolerance (rtol={rtol}): {num_exceeds_rtol} ({num_exceeds_rtol/total_elements*100:.2f}%)\n"
            f"Elements exceeding either tolerance: {num_exceeds_tolerance} ({pct_exceeds_tolerance:.2f}%)\n"
            f"Kernel output shape: {kernel_output.shape}\n"
            f"Reference output shape: {gt_output.shape}\n"
            f"Original error: {str(e)}"
        )


def CHECK_FLASH_ATTN_DECODE(
    q, k, v,
    k_cache, v_cache,
    block_tables,
    context_lens,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    decode_kernel,
    scale: float,
    num_groups: int,
    page_block_size: int,
    diffusion_block_size: int = 32,
    is_block_attn: bool = False,
):
    """
    Verify decode kernel correctness by comparing with PyTorch's scaled_dot_product_attention.
    
    Args:
        q: Query tensor [total_q_len, num_heads, head_dim]
        k: Key tensor [total_kv_len, num_kv_heads, head_dim]
        v: Value tensor [total_kv_len, num_kv_heads, head_dim]
        k_cache: KV cache for keys [num_page_blocks, page_block_size, num_kv_heads, head_dim]
        v_cache: KV cache for values [num_page_blocks, page_block_size, num_kv_heads, head_dim]
        block_tables: Block tables [num_seqs, max_seq_num_blocks]
        context_lens: Context lengths for each sequence [num_seqs]
        cu_seqlens_q: Cumulative sequence lengths for queries
        cu_seqlens_k: Cumulative sequence lengths for keys/values
        max_seqlen_q: Maximum sequence length for queries
        decode_kernel: The kernel function to test
        scale: Attention scale factor
        num_groups: Number of GQA groups (num_heads // num_kv_heads)
        page_block_size: Size of page blocks in KV cache
        diffusion_block_size: Size of diffusion blocks for block attention
        is_block_attn: Whether this is block attention mode
    """
    import torch
    import torch.nn.functional as F
    from einops import rearrange
    
    # Run kernel
    kernel_output = decode_kernel(
        q, k, v, k_cache, v_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
    )
    
    # Compute reference output using PyTorch's SDPA with KV cache
    num_seqs = len(cu_seqlens_q) - 1
    gt_output = torch.zeros_like(q)
    
    for seq_idx in range(num_seqs):
        q_start = cu_seqlens_q[seq_idx].item()
        q_end = cu_seqlens_q[seq_idx + 1].item()
        kv_start = cu_seqlens_k[seq_idx].item()
        kv_end = cu_seqlens_k[seq_idx + 1].item()
        
        q_seq = q[q_start:q_end]  # [seq_q_len, num_heads, head_dim]
        k_seq = k[kv_start:kv_end]  # [seq_kv_len, num_kv_heads, head_dim]
        v_seq = v[kv_start:kv_end]  # [seq_kv_len, num_kv_heads, head_dim]
        
        context_len = context_lens[seq_idx].item()
        
        # Load KV cache for this sequence
        k_cache_seq_list = []
        v_cache_seq_list = []
        
        for block_idx in range(block_tables.shape[1]):
            page_block_idx = block_tables[seq_idx, block_idx].item()
            if page_block_idx >= 0:
                # Calculate how many tokens to take from this block
                block_start = block_idx * page_block_size
                if block_start < context_len:
                    block_end = min(block_start + page_block_size, context_len)
                    num_tokens = block_end - block_start
                    k_cache_seq_list.append(k_cache[page_block_idx, :num_tokens])
                    v_cache_seq_list.append(v_cache[page_block_idx, :num_tokens])
        
        if k_cache_seq_list:
            k_cache_seq = torch.cat(k_cache_seq_list, dim=0)  # [context_len, num_kv_heads, head_dim]
            v_cache_seq = torch.cat(v_cache_seq_list, dim=0)  # [context_len, num_kv_heads, head_dim]
            
            # Combine KV cache and current KV
            k_combined = torch.cat([k_cache_seq, k_seq], dim=0)
            v_combined = torch.cat([v_cache_seq, v_seq], dim=0)
        else:
            k_combined = k_seq
            v_combined = v_seq
        
        q_sdpa = rearrange(q_seq, 's h d -> 1 h s d')  # [1, num_heads, seq_q_len, head_dim]
        k_sdpa = rearrange(k_combined, 's h d -> 1 h s d')  # [1, num_kv_heads, total_kv_len, head_dim]
        v_sdpa = rearrange(v_combined, 's h d -> 1 h s d')  # [1, num_kv_heads, total_kv_len, head_dim]
        
        if not is_block_attn:
            # Standard attention
            attn_out = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
                enable_gqa=True,
            )
        else:
            # Block attention with mask
            q_len = q_seq.shape[0]
            kv_len = k_combined.shape[0]
            block_mask = torch.zeros((1, 1, q_len, kv_len), dtype=q.dtype, device=q.device).bool()
            num_diffusion_blocks = (kv_len + diffusion_block_size - 1) // diffusion_block_size
            for block_idx in range(num_diffusion_blocks):
                block_start = block_idx * diffusion_block_size
                block_end = min(block_start + diffusion_block_size, kv_len)
                block_mask[..., block_start:block_end, :block_end] = True
            
            attn_out = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=block_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
                enable_gqa=True,
            )
        
        gt_output[q_start:q_end] = rearrange(attn_out, '1 h s d -> s h d').to(gt_output.dtype)
    
    # Compare results
    atol = 1e-2
    rtol = 1e-2
    try:
        torch.testing.assert_close(
            kernel_output,
            gt_output,
            atol=atol,
            rtol=rtol,
            msg="Decode kernel output does not match reference implementation"
        )
    except AssertionError as e:
        # Compute error statistics for debugging
        abs_diff = torch.abs(kernel_output - gt_output)
        max_diff = torch.max(abs_diff).item()
        mean_diff = torch.mean(abs_diff).item()
        rel_diff = torch.abs((kernel_output - gt_output) / (gt_output + 1e-8))
        max_rel_diff = torch.max(rel_diff).item()
        mean_rel_diff = torch.mean(rel_diff).item()
        
        # Count elements that exceed tolerance
        total_elements = kernel_output.numel()
        # Elements that exceed absolute tolerance
        exceeds_atol = (abs_diff > atol)
        num_exceeds_atol = exceeds_atol.sum().item()
        # Elements that exceed relative tolerance
        exceeds_rtol = (rel_diff > rtol)
        num_exceeds_rtol = exceeds_rtol.sum().item()
        # Elements that exceed either tolerance
        exceeds_tolerance = exceeds_atol | exceeds_rtol
        num_exceeds_tolerance = exceeds_tolerance.sum().item()
        pct_exceeds_tolerance = (num_exceeds_tolerance / total_elements * 100) if total_elements > 0 else 0
        
        raise AssertionError(
            f"Decode kernel verification failed!\n"
            f"Max absolute difference: {max_diff:.6f}\n"
            f"Mean absolute difference: {mean_diff:.6f}\n"
            f"Max relative difference: {max_rel_diff:.6f}\n"
            f"Mean relative difference: {mean_rel_diff:.6f}\n"
            f"Total elements: {total_elements}\n"
            f"Elements exceeding absolute tolerance (atol={atol}): {num_exceeds_atol} ({num_exceeds_atol/total_elements*100:.2f}%)\n"
            f"Elements exceeding relative tolerance (rtol={rtol}): {num_exceeds_rtol} ({num_exceeds_rtol/total_elements*100:.2f}%)\n"
            f"Elements exceeding either tolerance: {num_exceeds_tolerance} ({pct_exceeds_tolerance:.2f}%)\n"
            f"Kernel output shape: {kernel_output.shape}\n"
            f"Reference output shape: {gt_output.shape}\n"
            f"Original error: {str(e)}"
        ) 