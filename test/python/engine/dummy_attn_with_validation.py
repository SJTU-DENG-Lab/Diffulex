import torch
import torch.nn.functional as F
from einops import rearrange

from diffulex.attention.attn_impl import Attention as OriginalAttention


class AttentionWithValidation(OriginalAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_enabled = True
        self.atol = 1e-2
        self.rtol = 1e-2
        self.error_log = []

    def forward(self, q, k, v, mask=None):
        if not self.validation_enabled:
            return super().forward(q, k, v, mask)

        # Get metadata
        attn_metadata = self.fetch_attn_metadata()

        # Run original kernel
        output = super().forward(q, k, v, mask)

        # Run reference implementation
        ref_output = self._compute_reference(q, k, v, attn_metadata)
        self._validate_output(output, ref_output, attn_metadata)

        return output

    def _compute_reference(self, q, k, v, metadata):
        q_reshaped = rearrange(q, "s (nh hd) -> s nh hd", **self.q_shape)
        k_reshaped = rearrange(k, "s (nkvh hd) -> s nkvh hd", **self.kv_shape)
        v_reshaped = rearrange(v, "s (nkvh hd) -> s nkvh hd", **self.kv_shape)

        scale = self.scale
        k_cache, v_cache = self.k_cache, self.v_cache
        page_tables = metadata.page_tables
        context_lens = metadata.context_lens
        cu_seqlens_q = metadata.cu_seqlens_q
        valid_slices = getattr(metadata, 'valid_slices', None)
        page_size = metadata.page_size
        block_size = metadata.block_size

        num_seqs = len(cu_seqlens_q) - 1
        output = torch.zeros_like(q_reshaped)

        for seq_id in range(num_seqs):
            q_start = int(cu_seqlens_q[seq_id].item())
            if valid_slices is not None:
                valid_end = int(valid_slices[seq_id].item())
                valid_q_len = valid_end - q_start
            else:
                q_end = int(cu_seqlens_q[seq_id + 1].item())
                valid_q_len = q_end - q_start

            ctx_len = int(context_lens[seq_id].item())

            if valid_q_len <= 0:
                continue

            q_seq = q_reshaped[q_start:q_start + valid_q_len]

            # Reconstruct cache KV
            k_parts, v_parts = [], []
            if k_cache.numel() > 0 and ctx_len > 0:
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

            k_new = k_reshaped[q_start:q_start + valid_q_len]
            v_new = v_reshaped[q_start:q_start + valid_q_len]

            if k_parts:
                k_full = torch.cat(k_parts + [k_new], dim=0)
                v_full = torch.cat(v_parts + [v_new], dim=0)
            else:
                k_full = k_new
                v_full = v_new

            # Build block-causal mask (aligned with kernel line 179-181)
            mask = None
            if block_size > 0:
                qi = torch.arange(valid_q_len, device=q.device)
                kj = torch.arange(valid_q_len, device=q.device)
                # Kernel: ((offs_q_block // DLLM_BLOCK_SIZE + 1) * DLLM_BLOCK_SIZE)[:, None] > offs_kv_block[None, :]
                block_ends = ((qi // block_size) + 1) * block_size
                new_kv_mask = block_ends[:, None] > kj[None, :]

                if ctx_len > 0:
                    cache_mask = torch.ones(valid_q_len, ctx_len, dtype=torch.bool, device=q.device)
                    mask = torch.cat([cache_mask, new_kv_mask], dim=1)
                else:
                    mask = new_kv_mask
                mask = mask.unsqueeze(0).unsqueeze(0)

            q_sdpa = rearrange(q_seq, "s h d -> 1 h s d")
            k_sdpa = rearrange(k_full, "s h d -> 1 h s d")
            v_sdpa = rearrange(v_full, "s h d -> 1 h s d")

            attn_out = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=scale, enable_gqa=True
            )
            output[q_start:q_start + valid_q_len] = rearrange(attn_out, "1 h s d -> s h d")

        return rearrange(output, "s nh hd -> s (nh hd)").contiguous()

    def _validate_output(self, output, ref_output, metadata):
        try:
            torch.testing.assert_close(output, ref_output, atol=self.atol, rtol=self.rtol)
        except AssertionError as e:
            diff = (output - ref_output).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            error_msg = f"Validation failed - max_diff: {max_diff:.6f}, mean_diff: {mean_diff:.6f}, buffer_size: {metadata.buffer_size}"
            self.error_log.append(error_msg)
            print(f"[ATTN VALIDATION ERROR] {error_msg}")
            raise


def install_validation_hook():
    """Monkey patch to replace Attention with validation version"""
    import diffulex.attention.attn_impl
    diffulex.attention.attn_impl.Attention = AttentionWithValidation
    print("[VALIDATION] Attention class replaced with validation wrapper")

