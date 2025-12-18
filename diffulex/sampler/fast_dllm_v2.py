import torch

from dataclasses import dataclass

from diffulex.sampler.auto_sampler import AutoSampler
from diffulex.sampler.base import SamplerBase, SampleOutputBase
from diffulex.engine.sequence import SequenceBase


@dataclass
class FastdLLMV2SampleOutputForDiffusionLM(SampleOutputBase):
    pass


@AutoSampler.register("fast_dllm_v2")
class FastdLLMV2SamplerForDiffusionLM(SamplerBase):
    def _shift_logits(self, logits, last_logit=None):
        """
        Shift logits to align with Fast-dLLM's prediction pattern.
        参考 generation_functions.py 中的 logits shift 逻辑（105, 112行）
        """
        if logits.shape[1] == 0:
            print("Warning: logits sequence length is 0, returning empty logits")
            raise Exception("logits sequence length is 0")
            
        # 对应 generation_functions.py: logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
        shifted_logits = torch.zeros_like(logits)
        shifted_logits[1:, ...] = logits[:-1, ...]
        if last_logit is not None:
            shifted_logits[0, ...] = last_logit
            return shifted_logits
        shifted_logits[0, ...] = 1.0
        return shifted_logits
    
    def forward(self, seqs: list[SequenceBase], logits: torch.Tensor, temperatures: torch.Tensor,
                top_p=None, top_k=None, margin_confidence=False, neg_entropy=False, threshold=0.95):
        attn_metadata = self.fetch_attn_metadata()
        split_logits = torch.split(
            logits, [len(seq) for seq in seqs] if attn_metadata.is_prefill 
            else [attn_metadata.diffusion_block_size] * len(seqs), dim=0
        )
        
        accepted_ids_map = {}
        sampled_tokens_map = {}
        true_local_ids_map = {}
        
        for temperature, seq, seq_logits in zip(temperatures, seqs, split_logits):
            true_local_ids_sub_map = {}
            accepted_ids_sub_map = {}
            sampled_tokens_sub_map = {}
            
            shifted_logits = self._shift_logits(seq_logits, seq.cached_or_caching_num_tokens - 1)
            
            for block_id, block in enumerate(seq.diffusion_blocks):
                if not block.is_active or sum(block.local_mask_tokens) == 0:
                    continue
                
                if len(block.global_mask_token_ids) == 0:
                    continue
                
                mask_token_logits = shifted_logits[block.global_mask_token_ids, ...]
                
                confidence, sampled_tokens, initial_confidence = self.sample_tokens(
                    mask_token_logits, 
                    temperature, 
                    top_p=top_p, 
                    top_k=top_k, 
                    neg_entropy=(neg_entropy == "neg_entropy"),
                    margin_confidence=(margin_confidence == "margin_confidence")
                )
                
                high_conf_indices = torch.where(initial_confidence > threshold)[0]
                
                if len(high_conf_indices) == 0:
                    max_prob_idx = initial_confidence.argmax()
                    accepted_ids = torch.tensor([max_prob_idx], device=sampled_tokens.device, dtype=torch.long)
                else:
                    max_prob_idx = initial_confidence.argmax()
                    accepted_ids = torch.unique(torch.cat([
                        high_conf_indices,
                        torch.tensor([max_prob_idx], device=sampled_tokens.device, dtype=torch.long)
                    ]))
                
                true_local_ids_sub_map[str(block_id)] = [
                    block.local_mask_token_ids[accepted_id] for accepted_id in accepted_ids.tolist()
                ]
                accepted_ids_sub_map[str(block_id)] = accepted_ids.tolist()
                sampled_tokens_sub_map[str(block_id)] = sampled_tokens
            
            seq_idx = str(seq.seq_id)
            true_local_ids_map[seq_idx] = true_local_ids_sub_map
            accepted_ids_map[seq_idx] = accepted_ids_sub_map
            sampled_tokens_map[seq_idx] = sampled_tokens_sub_map

        return FastdLLMV2SampleOutputForDiffusionLM(
            true_local_ids_map=true_local_ids_map,
            accepted_ids_map=accepted_ids_map,
            sampled_tokens_map=sampled_tokens_map
        )