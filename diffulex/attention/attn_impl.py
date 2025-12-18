import os
import torch

import torch.nn as nn

from diffulex_kernel import (
    store_kvcache_distinct_layout, 
    store_kvcache_unified_layout, 
    dllm_flash_attn_decode, 
    dllm_flash_attn_prefill
)
from diffulex.attention.metadata import AttnMetaDataBase


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
        is_rtx_xx90 = lambda x: "4090" in x or "3090" in x
        self.kernel_options = {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_M1": 32,
            "BLOCK_N1": 64,
            "BLOCK_M2": 64,
            "BLOCK_N2": 32,
        } if is_rtx_xx90(torch.cuda.get_device_name(0)) else None
        
        # Import the specified fetch function
        from diffulex.attention import fetch_attn_metadata
        self.fetch_attn_metadata = fetch_attn_metadata
        
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: list[torch.Tensor] | None = None) -> torch.Tensor:
        # Reshape
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        attn_metadata: AttnMetaDataBase = self.fetch_attn_metadata()
        k_cache, v_cache = self.k_cache, self.v_cache
        is_unified_layout = attn_metadata.kv_cache_layout == "unified"

        # Fast Store KV cache
        if k_cache.numel() and v_cache.numel():
            if attn_metadata.need_kv_cache_store:
                store_kvcache = store_kvcache_unified_layout if is_unified_layout else store_kvcache_distinct_layout
                store_kvcache(k, v, k_cache, v_cache, attn_metadata.slot_mapping, attn_metadata)

        # Prefill / Decode logic
        if attn_metadata.is_prefill:
            if attn_metadata.block_tables is not None:
                # TODO: Implement Prefix Caching
                pass
            o = dllm_flash_attn_prefill(q, k, v, self.scale, attn_metadata)
        else:
            if is_unified_layout:
                o = dllm_flash_attn_decode(q, k, v, k_cache, v_cache, self.scale, attn_metadata)
            else:
                raise NotImplementedError("Distinct layout is not supported for decode mode")
            
        # Final reshape
        return o.view(-1, self.num_heads * self.head_dim).contiguous()