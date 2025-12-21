import torch
import torch.nn as nn
from einops import rearrange

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
        
        self.q_shape = {
            'nh': self.num_heads,
            'hd': self.head_dim,
        }
        self.kv_shape = {
            'nkvh': self.num_kv_heads,
            'hd': self.head_dim,
        }
        # Import the specified fetch function
        from diffulex.attention import fetch_attn_metadata
        self.fetch_attn_metadata = fetch_attn_metadata
        
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: list[torch.Tensor] | None = None) -> torch.Tensor:
        # Reshape
        q = rearrange(q, 's (nh hd) -> s nh hd', **self.q_shape)
        k = rearrange(k, 's (nkvh hd) -> s nkvh hd', **self.kv_shape)
        v = rearrange(v, 's (nkvh hd) -> s nkvh hd', **self.kv_shape)

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
                raise NotImplementedError("Distinct layout is not supported yet...")
            
        # Final reshape
        return rearrange(o, 's nh hd -> s (nh hd)').contiguous()