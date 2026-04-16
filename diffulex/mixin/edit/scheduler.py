from __future__ import annotations


class EditSchedulerMixin:
    def apply_edit_writes_map(self, req, sample_output) -> None:
        req_id_str = str(req.req_id)
        edit_writes_map = sample_output.edit_writes_map[req_id_str]
        for block_id, block_writes in edit_writes_map.items():
            if not block_writes:
                continue

            dllm_block = req.dllm_blocks[int(block_id)]
            for rel_idx, token in block_writes.items():
                rel_idx = int(rel_idx)
                token = int(token)
                prev_token = dllm_block.token_ids[rel_idx]
                dllm_block.write_token(token, rel_idx)
                if prev_token == dllm_block.mask_token_id and token != dllm_block.mask_token_id:
                    req.new_tokens += 1
