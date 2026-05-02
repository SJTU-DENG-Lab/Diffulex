from __future__ import annotations

import torch

from diffulex.attention.metadata import AttnMetaDataBase
from diffulex.engine.request import AutoReq, DllmReq
from diffulex.engine.model_runner import ModelRunnerBase
from diffulex.logger import get_logger
from diffulex.strategy_template.multi_block.engine.cudagraph import MultiBlockCudaGraphMixin

logger = get_logger(__name__)


class MultiBlockModelRunnerTemplate(MultiBlockCudaGraphMixin, ModelRunnerBase):
    def _prefill_warmup(self):
        logger.info("Warming up prefill...")

        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        num_reqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_reqs)
        test_input_ids = [0] * max_model_len
        reqs = [AutoReq.create(config=self.config, token_ids=test_input_ids) for _ in range(num_reqs)]

        for req in reqs:
            req.init_multi_block(self.config)
            req.make_pending()
            req.dp_rank = self.dp_rank
            req.step()
            req.mark_execution_prepared()

        self.run(reqs)

        for req in reqs:
            req.clear_execution_prepared()
            req.postprocess()

        torch.cuda.empty_cache()

    @staticmethod
    def _cached_prefix_len(req: DllmReq) -> int:
        return int(req.contiguous_in_cache_prefix_len)

    def _prepare_prefill_req(self, req: DllmReq):
        input_ids = list(req.running_sequence)
        q_len = len(input_ids)
        context_len = self._cached_prefix_len(req)
        positions = list(range(context_len, context_len + q_len))

        # Prefix-cache prefill runs the model only on the uncached suffix.
        seqlen_q = q_len
        seqlen_k = q_len

        slot_mapping = []
        for block in req.dllm_blocks:
            if block.end <= context_len:
                continue
            if block.start >= req.running_len:
                break
            if block.rel_page_id >= len(req.page_table):
                break

            abs_page_id = req.page_table[block.rel_page_id]
            start = abs_page_id * self.page_size + block.start % self.page_size
            end = start + self.block_size

            if block.is_to_cache:
                slot_mapping.extend(range(start, end))
            else:
                slot_mapping.extend([-1] * self.block_size)

        remain_num_tokens = q_len - len(slot_mapping)
        slot_mapping.extend([-1] * remain_num_tokens)

        # Prefill logits are defined over the whole running suffix, even when some
        # tokens are not written back to KV this step (e.g. an active uncached tail
        # block after a cached prefix hit). `slot_mapping` controls KV store only.
        valid_slice = q_len

        return dict(
            input_ids=input_ids,
            positions=positions,
            context_len=context_len,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            valid_slice=valid_slice,
            slot_mapping=slot_mapping,
            status=0,
            prefix_len=req.prefix_len,
            padded_prefix_len=req.padded_prefix_len,
        )

    def _prepare_decode_req(self, req: DllmReq):
        input_ids = list(req.running_sequence)
        positions = list(req.running_position_ids)
        context_len = self._cached_prefix_len(req)

        seqlen_q = req.chunk_size
        seqlen_k = req.chunk_size
        valid_slice = req.valid_len
    
        slot_mapping = []
        for block in req.dllm_block_buffer.dllm_blocks:
            if block.rel_page_id >= len(req.page_table):
                break
            
            abs_page_id = req.page_table[block.rel_page_id]
            start = abs_page_id * self.page_size + block.start % self.page_size
            end = start + self.block_size
            
            if block.is_to_cache:
                slot_mapping.extend(range(start, end))
            else:
                slot_mapping.extend([-1] * self.block_size)
                
        remain_num_tokens = len(input_ids) - len(slot_mapping)
        slot_mapping.extend([-1] * remain_num_tokens)

        return dict(
            input_ids=input_ids,
            positions=positions,
            context_len=context_len,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            valid_slice=valid_slice,
            slot_mapping=slot_mapping,
            status=1,
            prefix_len=0,
            padded_prefix_len=0,
        )

    def prepare_chunked_prefill_multi_block(self, reqs: list[DllmReq]):
        input_ids: list[int] = []
        positions: list[int] = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        valid_slices: list[int] = []
        slot_mapping: list[int] = []
        context_lens: list[int] = []
        is_prefill: list[bool] = []
        status_table: list[int] = []
        prefix_lens_list: list[int] = []
        padded_prefix_lens_list: list[int] = []

        for req in reqs:
            if not req.is_execution_prepared:
                req.step()
            prepared = self._prepare_prefill_req(req) if req.is_prefilling else self._prepare_decode_req(req)
            status_table.append(prepared["status"])
            prefix_lens_list.append(prepared["prefix_len"])
            padded_prefix_lens_list.append(prepared["padded_prefix_len"])
            is_prefill.append(req.is_prefilling)
            input_ids.extend(prepared["input_ids"])
            positions.extend(prepared["positions"])
            context_lens.append(prepared["context_len"])

            seqlen_q = prepared["seqlen_q"]
            seqlen_k = prepared["seqlen_k"]
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            valid_slices.append(cu_seqlens_q[-2] + prepared["valid_slice"])
            slot_mapping.extend(prepared["slot_mapping"])

        max_page_len = max((len(req.page_table) for req in reqs), default=1)
        buffers = self._ensure_runtime_static_buffers(
            token_capacity=len(input_ids),
            req_capacity=len(reqs),
            page_capacity=max_page_len,
        )
        page_tables = self._prepare_static_page_tables(buffers, reqs)
        input_ids_tensor = self._copy_1d_to_runtime_static(buffers, "input_ids", input_ids, torch.int64)
        positions_tensor = self._copy_1d_to_runtime_static(buffers, "positions", positions, torch.int64)
        context_lens_tensor = self._copy_1d_to_runtime_static(buffers, "context_lens", context_lens, torch.int32)
        cu_seqlens_q_tensor = self._copy_1d_to_runtime_static(buffers, "cu_seqlens_q", cu_seqlens_q, torch.int32)
        cu_seqlens_k_tensor = self._copy_1d_to_runtime_static(buffers, "cu_seqlens_k", cu_seqlens_k, torch.int32)
        slot_mapping_tensor = self._copy_1d_to_runtime_static(buffers, "slot_mapping", slot_mapping, torch.int32)
        valid_slices_tensor = self._copy_1d_to_runtime_static(buffers, "valid_slices", valid_slices, torch.int32)
        status_table_tensor = self._copy_1d_to_runtime_static(buffers, "status_table", status_table, torch.int32)
        prefix_lens_tensor = self._copy_1d_to_runtime_static(buffers, "prefix_lens", prefix_lens_list, torch.int32)
        padded_prefix_lens_tensor = self._copy_1d_to_runtime_static(
            buffers,
            "padded_prefix_lens",
            padded_prefix_lens_list,
            torch.int32,
        )

        self.set_attn_metadata(
            is_prefill=is_prefill,
            cu_seqlens_q=cu_seqlens_q_tensor,
            cu_seqlens_k=cu_seqlens_k_tensor,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping_tensor,
            context_lens=context_lens_tensor,
            page_tables=page_tables,
            page_size=self.page_size,
            block_size=self.block_size,
            kv_cache_layout=self.config.kv_cache_layout,
        )
        attn_metadata: AttnMetaDataBase = self.fetch_attn_metadata()
        attn_metadata.init_multi_block(
            valid_slices=valid_slices_tensor,
            buffer_size=self.config.buffer_size,
            is_prefix_full=self.is_prefix_full,
            status_table=status_table_tensor,
            prefix_lens=prefix_lens_tensor,
            padded_prefix_lens=padded_prefix_lens_tensor,
        )
        return input_ids_tensor, positions_tensor

    def prepare_idle_multi_block(self):
        buffers = self._ensure_runtime_static_buffers(token_capacity=1, req_capacity=1, page_capacity=1)
        input_ids_tensor = buffers["input_ids"][:0]
        positions_tensor = buffers["positions"][:0]
        cu_seqlens_tensor = buffers["cu_seqlens_q"][:1]
        cu_seqlens_tensor.zero_()
        empty_i32 = buffers["slot_mapping"][:0]
        page_tables = buffers["page_tables"][:0, :1]

        self.set_attn_metadata(
            is_prefill=[],
            cu_seqlens_q=cu_seqlens_tensor,
            cu_seqlens_k=cu_seqlens_tensor,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=empty_i32,
            context_lens=empty_i32,
            page_tables=page_tables,
            page_size=self.page_size,
            block_size=self.block_size,
            kv_cache_layout=self.config.kv_cache_layout,
        )
        attn_metadata: AttnMetaDataBase = self.fetch_attn_metadata()
        attn_metadata.init_multi_block(
            valid_slices=empty_i32,
            buffer_size=self.config.buffer_size,
            is_prefix_full=self.is_prefix_full,
            status_table=empty_i32,
            prefix_lens=empty_i32,
            padded_prefix_lens=empty_i32,
        )
        return input_ids_tensor, positions_tensor

    @torch.inference_mode()
    def run_idle_multi_block(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        # Idle ranks must enter model forward so EP dispatch/combine collectives
        # see all participants, but they have no local logits to sample.
        _ = self.model(input_ids, positions)

    @torch.inference_mode()
    def run_model_multi_block(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ):
        attn_metadata: AttnMetaDataBase = self.fetch_attn_metadata()

        if (attn_metadata.status_table == 0).any():
            return self.model.compute_logits(self.model(input_ids, positions))

        if not self._can_use_decode_graph(attn_metadata, input_ids):
            return self.model.compute_logits(self.model(input_ids, positions))
        return self._run_decode_cudagraph(input_ids, positions, attn_metadata)

    def run_multi_block(self, reqs: list[DllmReq]) -> list[int]:
        return self._run_multi_block_subgroup(reqs)

    def _run_multi_block_subgroup(self, reqs: list[DllmReq]):
        local_reqs = self.filter_local_reqs(reqs)
        if not local_reqs:
            if self.cross_dp_ep:
                input_ids, positions = self.prepare_idle_multi_block()
                self.run_idle_multi_block(input_ids, positions)
                self.reset_attn_metadata()
            return self.gather_dp_sample_output(None)

        input_ids, positions = self.prepare_chunked_prefill_multi_block(local_reqs)
        temperatures = self.prepare_sample(local_reqs) if self.is_model_parallel_root else None
        logits = self.run_model_multi_block(input_ids, positions)
        sample_output = self.sampler(local_reqs, logits, temperatures) if self.is_model_parallel_root else None
        self.reset_attn_metadata()
        return self.gather_dp_sample_output(sample_output)
