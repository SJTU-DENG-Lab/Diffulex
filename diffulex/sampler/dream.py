import torch

from dataclasses import dataclass

from diffulex.sampler.auto_sampler import AutoSampler
from diffulex.sampler.base import SamplerShiftLogits, SampleOutputBase


@dataclass
class DreamSampleOutputForDiffusionLM(SampleOutputBase):
    pass


@AutoSampler.register("dream")
class DreamSamplerForDiffusionLM(SamplerShiftLogits):
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor,
                top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):
        context = self.fetch_attn_metadata()
        seqs = context.seqs
        split_logits = torch.split(logits, [len(seq) for seq in seqs] if context.is_prefill else context.seq_lens, dim=0)
        accepted_ids_map = {}
        sampled_tokens_map = {}
        true_local_ids_map = {}
        for temperature, seq, seq_logits in zip(temperatures, seqs, split_logits):
            true_local_ids_sub_map = {}
            accepted_ids_sub_map = {}
            sampled_tokens_sub_map = {}
            
            last_logits = self._fetch_last_logits(seq_logits, seq)
            
            shifted_logits = self._shift_logits(seq_logits, last_logits)
            for block_id, block in enumerate(seq.diffusion_blocks):
                if not block.is_active or sum(block.local_mask_tokens) == 0:
                    continue
                
                if len(block.global_mask_token_ids) > 0:
                    mask_token_logits = shifted_logits[block.global_mask_token_ids, ...]
                    confidence, sampled_tokens, initial_confidence = self.sample_tokens(
                        mask_token_logits, 
                        temperature, 
                        top_p=top_p, 
                        top_k=top_k, 
                        neg_entropy=(neg_entropy == "neg_entropy"),
                        margin_confidence=(margin_confidence == "margin_confidence")
                    )
                    
                if block.pre_block_complete:
                    high_conf_indices = torch.where(initial_confidence > block.accept_threshold)[0]        
                    if len(high_conf_indices) == 0:
                        number_transfer_tokens = 1
                        _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                    else:
                        transfer_index = torch.tensor([], device=sampled_tokens.device, dtype=torch.long)
                    accepted_ids = torch.unique(torch.cat([transfer_index, high_conf_indices]))
                else:
                    high_conf_indices = torch.where(initial_confidence > block.accept_threshold)[0]
                    accepted_ids = high_conf_indices

                true_local_ids_sub_map[str(block_id)] = [block.local_mask_token_ids[accepted_id] for accepted_id in accepted_ids.tolist()]
                accepted_ids_sub_map[str(block_id)] = accepted_ids.tolist()
                sampled_tokens_sub_map[str(block_id)] = sampled_tokens
            
            seq_idx = str(seq.seq_id)
            true_local_ids_map[seq_idx] = true_local_ids_sub_map
            accepted_ids_map[seq_idx] = accepted_ids_sub_map
            sampled_tokens_map[seq_idx] = sampled_tokens_sub_map

        return DreamSampleOutputForDiffusionLM(
            true_local_ids_map=true_local_ids_map,
            accepted_ids_map=accepted_ids_map,
            sampled_tokens_map=sampled_tokens_map
        )

