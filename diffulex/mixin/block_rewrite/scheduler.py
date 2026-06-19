from __future__ import annotations


class BlockRewriteSchedulerMixin:
    def apply_block_writes_map(self, req, sample_output) -> None:
        req_id_str = str(req.req_id)
        block_writes_map = sample_output.block_writes_map[req_id_str]
        for block_id, block_writes in block_writes_map.items():
            if not block_writes:
                continue

            dllm_block = req.dllm_blocks[int(block_id)]
            for rel_idx, token in block_writes.items():
                rel_idx = int(rel_idx)
                token = int(token)
                prev_token = dllm_block.token_ids[rel_idx]
                dllm_block.write_token(token, rel_idx)
                req.on_block_token_rewrite(
                    block=dllm_block,
                    rel_idx=rel_idx,
                    old_token=prev_token,
                    new_token=token,
                )
