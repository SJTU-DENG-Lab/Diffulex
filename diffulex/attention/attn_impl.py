import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from diffulex_kernel import (
    store_kv_cache_distinct_layout,
    store_kv_cache_unified_layout,
    chunked_prefill_attn_unified,
)
from diffulex.attention.metadata import AttnMetaDataBase
from diffulex.logger import get_logger


USE_REFERENCE_MULTI_BLOCK_ATTN = os.environ.get("DIFFULEX_USE_REFERENCE_MULTI_BLOCK_ATTN", "0") == "1"
logger = get_logger(__name__)


def reference_multi_block_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    attn_metadata: AttnMetaDataBase,
    scale: float,
) -> torch.Tensor:
    """Reference multi-block attention.

    Debug-only reference path for multi-block attention. Normal execution should keep
    requests batched and use the Triton kernel; set
    ``DIFFULEX_USE_REFERENCE_MULTI_BLOCK_ATTN=1`` only when validating/debugging the
    kernel path.
    """
    page_tables = attn_metadata.page_tables
    context_lens = attn_metadata.context_lens
    cu_seqlens_q = attn_metadata.cu_seqlens_q
    valid_slices = getattr(attn_metadata, "valid_slices", None)
    page_size = attn_metadata.page_size
    block_size = attn_metadata.block_size

    num_seqs = len(cu_seqlens_q) - 1
    output = torch.zeros_like(q)

    for seq_id in range(num_seqs):
        q_start = int(cu_seqlens_q[seq_id].item())
        q_end = int(cu_seqlens_q[seq_id + 1].item())
        if valid_slices is not None:
            valid_end = int(valid_slices[seq_id].item())
            valid_q_len = valid_end - q_start
        else:
            valid_q_len = q_end - q_start

        ctx_len = int(context_lens[seq_id].item())
        if valid_q_len <= 0:
            continue

        q_seq = q[q_start : q_start + valid_q_len]

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

        cached_ctx_len = sum(part.shape[0] for part in k_parts)
        if cached_ctx_len != ctx_len:
            logger.warning(
                "reference_multi_block_attn: context/page-table mismatch for seq %s: "
                "context_len=%s, reconstructed_ctx_len=%s",
                seq_id,
                ctx_len,
                cached_ctx_len,
            )

        k_new = k[q_start : q_start + valid_q_len]
        v_new = v[q_start : q_start + valid_q_len]
        if k_parts:
            k_full = torch.cat(k_parts + [k_new], dim=0)
            v_full = torch.cat(v_parts + [v_new], dim=0)
        else:
            k_full = k_new
            v_full = v_new

        mask = None
        if block_size > 0:
            qi = torch.arange(valid_q_len, device=q.device)
            kj = torch.arange(valid_q_len, device=q.device)
            block_ends = ((qi // block_size) + 1) * block_size
            new_kv_mask = block_ends[:, None] > kj[None, :]
            if cached_ctx_len > 0:
                cache_mask = torch.ones(valid_q_len, cached_ctx_len, dtype=torch.bool, device=q.device)
                mask = torch.cat([cache_mask, new_kv_mask], dim=1)
            else:
                mask = new_kv_mask
            mask = mask.unsqueeze(0).unsqueeze(0)

        q_sdpa = rearrange(q_seq, "s h d -> 1 h s d")
        k_sdpa = rearrange(k_full, "s h d -> 1 h s d")
        v_sdpa = rearrange(v_full, "s h d -> 1 h s d")
        attn_out = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            enable_gqa=True,
        )
        output[q_start : q_start + valid_q_len] = rearrange(attn_out, "1 h s d -> s h d")

    return output


class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

        self.q_shape = {
            "nh": self.num_heads,
            "hd": self.head_dim,
        }
        self.kv_shape = {
            "nkvh": self.num_kv_heads,
            "hd": self.head_dim,
        }
        # Import the specified fetch function
        from diffulex.attention import fetch_attn_metadata

        self.fetch_attn_metadata = fetch_attn_metadata

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # Reshape
        q = rearrange(q, "s (nh hd) -> s nh hd", **self.q_shape)
        k = rearrange(k, "s (nkvh hd) -> s nkvh hd", **self.kv_shape)
        v = rearrange(v, "s (nkvh hd) -> s nkvh hd", **self.kv_shape)

        attn_metadata: AttnMetaDataBase = self.fetch_attn_metadata()
        k_cache, v_cache = self.k_cache, self.v_cache
        is_unified_layout = attn_metadata.kv_cache_layout == "unified"

        # Fast Store KV cache
        if k_cache.numel() and v_cache.numel():
            if attn_metadata.need_kv_cache_store:
                store_kv_cache = store_kv_cache_unified_layout if is_unified_layout else store_kv_cache_distinct_layout
                store_kv_cache(k, v, k_cache, v_cache, attn_metadata.slot_mapping, attn_metadata)

        if getattr(attn_metadata, "use_multi_block", False) and USE_REFERENCE_MULTI_BLOCK_ATTN:
            o = reference_multi_block_attn(q, k, v, k_cache, v_cache, attn_metadata, self.scale)
        else:
            o = chunked_prefill_attn_unified(q, k, v, k_cache, v_cache, attn_metadata)

        # Final reshape
        return rearrange(o, "s nh hd -> s (nh hd)").contiguous()
