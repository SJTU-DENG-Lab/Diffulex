from __future__ import annotations

import torch

from tqdm import tqdm

from typing import TYPE_CHECKING

from diffulex.attention.metadata import (
    AttnMetaDataBase,
    set_warming_up,
    reset_warming_up,
)
from diffulex.engine.request import DllmReq

if TYPE_CHECKING:
    from diffulex.engine.model_runner import ModelRunnerBase


class ModelRunnerMultiBlockMixin:
    def _prepare_prefill_req(self: ModelRunnerBase, req: DllmReq):
        running_sequence = req.running_sequence
        total_seqlen = len(running_sequence)
        in_cache_len = req.in_cache_len

        input_ids = list(running_sequence)
        positions = list(range(in_cache_len, total_seqlen))
        context_len = 0

        seqlen_q = total_seqlen - in_cache_len
        seqlen_k = total_seqlen

        slot_mapping: list[int] = []
        if req.page_table:
            has_padding_mask = req.padded_prefix_len > 0
            for rel_page_id in range(0, req.num_pages_with_seq_len(len(running_sequence))):
                if req.page_cache_missed[rel_page_id]:
                    if has_padding_mask and rel_page_id >= req.num_prefix_pages - 1:
                        slot_mapping.extend([-1] * self.page_size)
                    else:
                        # NOTE: only support block_size % page_size == 0
                        start = req.page_table[rel_page_id] * self.page_size
                        end = start + self.page_size
                        slot_mapping.extend(range(start, end))
                else:
                    slot_mapping.extend([-1] * self.page_size)

        return dict(
            input_ids=input_ids,
            positions=positions,
            context_len=context_len,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            valid_slice=seqlen_q,
            slot_mapping=slot_mapping,
            status=0,
            prefix_len=req.prefix_len,
            padded_prefix_len=req.padded_prefix_len,
        )

    def _prepare_decode_req(self: ModelRunnerBase, req: DllmReq):
        input_ids = list(req.running_sequence)
        positions = list(req.running_position_ids)
        context_len = req.in_cache_len

        seqlen_q = req.chunk_size
        seqlen_k = req.chunk_size
        valid_slice = req.valid_len

        slot_mapping: list[int] = []
        num_pages_per_block = req.block_size // self.page_size
        for block in req.dllm_block_buffer.dllm_blocks:
            if block.is_to_cache:
                cur_prefix_num_pages = (block.start + self.page_size - 1) // self.page_size
                for in_blk_page_id in range(num_pages_per_block):
                    rel_page_id = cur_prefix_num_pages + in_blk_page_id
                    start = req.page_table[rel_page_id] * self.page_size
                    end = start + self.page_size
                    slot_mapping.extend(range(start, end))
            else:
                slot_mapping.extend([-1] * block.block_size)

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

    def prepare_chunked_prefill_multi_block(self: ModelRunnerBase, reqs: list[DllmReq]):
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

        page_tables = self.prepare_page_tables(reqs)
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_tensor = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        context_lens_tensor = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q_tensor = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k_tensor = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        valid_slices_tensor = torch.tensor(valid_slices, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        status_table_tensor = torch.tensor(status_table, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        prefix_lens_tensor = torch.tensor(prefix_lens_list, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        padded_prefix_lens_tensor = torch.tensor(padded_prefix_lens_list, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
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

    @torch.inference_mode()
    def run_model_multi_block(
        self: ModelRunnerBase,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ):
        attn_metadata: AttnMetaDataBase = self.fetch_attn_metadata()

        if (
            (attn_metadata.status_table == 0).any()
            or self.enforce_eager
            or input_ids.size(0) > 512 * (self.config.buffer_size * self.config.block_size)
        ):
            return self.model.compute_logits(self.model(input_ids, positions))

        num_tokens = input_ids.size(0)
        
        graph = self.graphs[next(x for x in self.graph_bs if x >= num_tokens)]
        graph_vars = self.graph_vars
        for key, value in graph_vars.items():
            if key != "outputs":
                value.zero_()

        num_seqs = len(attn_metadata.context_lens)
        graph_vars["input_ids"][:num_tokens] = input_ids
        graph_vars["positions"][:num_tokens] = positions
        graph_vars["slot_mapping"][:num_tokens] = attn_metadata.slot_mapping
        graph_vars["context_lens"][:num_seqs] = attn_metadata.context_lens
        graph_vars["cu_seqlens_q"][: num_seqs + 1] = attn_metadata.cu_seqlens_q
        graph_vars["cu_seqlens_k"][: num_seqs + 1] = attn_metadata.cu_seqlens_k
        graph_vars["valid_slices"][:num_seqs] = attn_metadata.valid_slices
        graph_vars["status_table"][:num_seqs] = attn_metadata.status_table
        graph_vars["prefix_lens"][:num_seqs] = attn_metadata.prefix_lens
        graph_vars["padded_prefix_lens"][:num_seqs] = attn_metadata.padded_prefix_lens
        graph_vars["page_tables"][:num_seqs, : attn_metadata.page_tables.size(1)] = attn_metadata.page_tables

        # Update attn_metadata to use graph_vars tensors
        attn_metadata.slot_mapping = graph_vars["slot_mapping"]
        attn_metadata.context_lens = graph_vars["context_lens"]
        attn_metadata.cu_seqlens_q = graph_vars["cu_seqlens_q"]
        attn_metadata.cu_seqlens_k = graph_vars["cu_seqlens_k"]
        attn_metadata.valid_slices = graph_vars["valid_slices"]
        attn_metadata.status_table = graph_vars["status_table"]
        attn_metadata.prefix_lens = graph_vars["prefix_lens"]
        attn_metadata.padded_prefix_lens = graph_vars["padded_prefix_lens"]
        attn_metadata.page_tables = graph_vars["page_tables"]

        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:num_tokens])

    def run_multi_block(self: ModelRunnerBase, reqs: list[DllmReq]) -> list[int]:
        input_ids, positions = self.prepare_chunked_prefill_multi_block(reqs)
        temperatures = self.prepare_sample(reqs) if self.rank == 0 else None
        logits = self.run_model_multi_block(input_ids, positions)
        sample_output = self.sampler(reqs, logits, temperatures) if self.rank == 0 else None
        self.reset_attn_metadata()
        return sample_output

    @torch.inference_mode()
    def capture_cudagraph_multi_block(self: ModelRunnerBase):
        set_warming_up(True)
        config = self.config
        hf_config = config.hf_config
        max_num_seqs = min(self.config.max_num_reqs, 512)
        max_num_pages = (config.max_model_len + self.page_size - 1) // self.page_size
        block_size = config.block_size
        buffer_size = config.buffer_size
        chunk_size = block_size * buffer_size

        max_num_tokens = max_num_seqs * chunk_size

        input_ids = torch.zeros(max_num_tokens, dtype=torch.int64)
        positions = torch.zeros(max_num_tokens, dtype=torch.int64)
        slot_mapping = torch.full((max_num_tokens,), -1, dtype=torch.int32)
        context_lens = torch.zeros(max_num_seqs, dtype=torch.int32)
        page_tables = torch.zeros(max_num_seqs, max_num_pages, dtype=torch.int32)
        valid_slices = torch.zeros(max_num_seqs, dtype=torch.int32)
        status_table = torch.zeros(max_num_seqs, dtype=torch.int32)
        prefix_lens = torch.zeros(max_num_seqs, dtype=torch.int32)
        padded_prefix_lens = torch.zeros(max_num_seqs, dtype=torch.int32)
        outputs = torch.zeros(max_num_tokens, hf_config.hidden_size)

        cu_seqlens_q = torch.zeros(max_num_seqs + 1, dtype=torch.int32)
        for i in range(max_num_seqs + 1):
            cu_seqlens_q[i] = i * chunk_size

        cu_seqlens_k = torch.zeros(max_num_seqs + 1, dtype=torch.int32)
        for i in range(max_num_seqs + 1):
            cu_seqlens_k[i] = i * config.max_model_len

        self.graph_bs = []
        seq_bs_list = [1, 2, 4, 8] + list(range(16, max_num_seqs + 16, 16))
        for num_seqs in seq_bs_list:
            self.graph_bs.append(num_seqs * chunk_size)
        self.graphs = {}
        self.graph_pool = None

        for num_tokens in tqdm(reversed(self.graph_bs), desc="Capturing CUDA graphs"):
            num_seqs = num_tokens // chunk_size
            graph = torch.cuda.CUDAGraph()

            self.set_attn_metadata(
                False,
                slot_mapping=slot_mapping[:num_tokens],
                context_lens=context_lens[:num_seqs],
                cu_seqlens_q=cu_seqlens_q[:num_seqs + 1],
                cu_seqlens_k=cu_seqlens_k[:num_seqs + 1],
                max_seqlen_q=chunk_size,
                max_seqlen_k=config.max_model_len,
                page_size=config.kv_cache_page_size,
                page_tables=page_tables[:num_seqs],
                block_size=block_size,
                kv_cache_layout=config.kv_cache_layout,
            )
            attn_metadata: AttnMetaDataBase = self.fetch_attn_metadata()
            attn_metadata.init_multi_block(
                valid_slices=valid_slices[:num_seqs],
                buffer_size=buffer_size,
                is_prefix_full=getattr(self, "is_prefix_full", False),
                status_table=status_table[:num_seqs],
                prefix_lens=prefix_lens[:num_seqs],
                padded_prefix_lens=padded_prefix_lens[:num_seqs],
            )

            outputs[:num_tokens] = self.model(input_ids[:num_tokens], positions[:num_tokens])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:num_tokens] = self.model(input_ids[:num_tokens], positions[:num_tokens])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[num_tokens] = graph
            torch.cuda.synchronize()
            self.reset_attn_metadata()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_tables=page_tables,
            valid_slices=valid_slices,
            status_table=status_table,
            prefix_lens=prefix_lens,
            padded_prefix_lens=padded_prefix_lens,
            outputs=outputs,
        )
        reset_warming_up()
