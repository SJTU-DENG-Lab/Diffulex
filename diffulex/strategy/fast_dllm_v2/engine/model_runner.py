from __future__ import annotations

import os
from multiprocessing.synchronize import Event

import torch
from tqdm import tqdm

from diffulex.attention.metadata import reset_warming_up, set_fetch_fn_for_attn_metadata, set_warming_up
from diffulex.config import Config
from diffulex.engine.model_runner import AutoModelRunner, ModelRunnerBase
from diffulex.profiling import record_function
from diffulex.strategy.fast_dllm_v2.attention.metadata import (
    fetch_fast_dllm_v2_attn_metadata,
    reset_fast_dllm_v2_attn_metadata,
    set_fast_dllm_v2_attn_metadata,
)
from diffulex.strategy.fast_dllm_v2.engine.request import FastDLLMV2Mode

_FDV2_GRAPH_FULL_BUFFER = "full_buffer_init"
_FDV2_GRAPH_SUB_BLOCK = "sub_block_cache_only"
_FDV2_GRAPH_FINAL_COMMIT = "final_commit"


@AutoModelRunner.register("fast_dllm_v2")
class FastDLLMV2ModelRunner(ModelRunnerBase):
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        set_fetch_fn_for_attn_metadata(fetch_fast_dllm_v2_attn_metadata)
        self.init_attn_metadata_fn(
            set_fast_dllm_v2_attn_metadata,
            reset_fast_dllm_v2_attn_metadata,
            fetch_fast_dllm_v2_attn_metadata,
        )
        self.mask_token_id = config.mask_token_id
        self.fdv2_attention_block_size = int(config.block_size) * int(config.buffer_size)
        self.is_prefix_full = False
        self.mask_prefix_hole = False
        self.prefix_causal = False
        super().__init__(config, rank, event)

    def _prepare_decode_req(self, req):
        prepared = super()._prepare_decode_req(req)

        if getattr(req, "fdv2_mode", None) == FastDLLMV2Mode.SUB_BLOCK_REFINE:
            if not bool(getattr(req, "fdv2_use_block_cache", True)):
                prepared["slot_mapping"] = self._fdv2_buffer_slot_mapping(req)
                prepared["fdv2_cache_only"] = False
                return prepared

            block = req.fdv2_current_sub_block
            input_ids = list(block.token_ids)
            positions = list(range(block.start, block.end))
            slot_mapping = []
            for abs_pos in range(block.start, block.end):
                rel_page_id = abs_pos // req.page_size
                if rel_page_id >= len(req.page_table):
                    slot_mapping.append(-1)
                    continue
                page_id = req.page_table[rel_page_id]
                slot_mapping.append(page_id * self.page_size + abs_pos % self.page_size)

            return dict(
                input_ids=input_ids,
                positions=positions,
                context_len=req.fdv2_read_cache_len,
                seqlen_q=req.block_size,
                seqlen_k=req.fdv2_read_cache_len,
                valid_slice=req.block_size,
                slot_mapping=slot_mapping,
                status=2,
                prefix_len=0,
                padded_prefix_len=0,
                fdv2_cache_only=True,
            )

        if getattr(req, "fdv2_mode", None) == FastDLLMV2Mode.FULL_BUFFER_INIT:
            prepared["slot_mapping"] = self._fdv2_buffer_slot_mapping(req)
            prepared["fdv2_cache_only"] = False
            return prepared

        if getattr(req, "fdv2_mode", None) == FastDLLMV2Mode.FINAL_COMMIT:
            prepared["slot_mapping"] = self._fdv2_buffer_slot_mapping(req)
            prepared["fdv2_cache_only"] = False
            return prepared

        prepared["fdv2_cache_only"] = False
        return prepared

    def _fdv2_buffer_slot_mapping(self, req) -> list[int]:
        slots: list[int] = []
        for abs_pos in range(req.fdv2_buffer_start, req.fdv2_buffer_end):
            rel_page_id = abs_pos // req.page_size
            if rel_page_id >= len(req.page_table):
                slots.append(-1)
                continue
            page_id = req.page_table[rel_page_id]
            slots.append(page_id * self.page_size + abs_pos % self.page_size)
        return slots

    def prepare_chunked_prefill_multi_block(self, reqs):
        input_ids, positions = super().prepare_chunked_prefill_multi_block(reqs)
        if not reqs:
            return input_ids, positions
        attn_metadata = self.fetch_attn_metadata()
        # Diffulex uses `block_size` as Fast-dLLM v2's sub-block scheduling
        # unit. The adapted model's block-causal attention mask is still the
        # full Fast-dLLM block (`small_block_size * num_small_blocks`).
        attn_metadata.block_size = self.fdv2_attention_block_size
        attn_metadata.fdv2_mode = int(getattr(reqs[0], "fdv2_mode", FastDLLMV2Mode.FULL_BUFFER_INIT))
        fdv2_cache_only = any(
            getattr(req, "is_decoding", False)
            and getattr(req, "fdv2_mode", None) == FastDLLMV2Mode.SUB_BLOCK_REFINE
            and bool(getattr(req, "fdv2_use_block_cache", True))
            for req in reqs
        )
        attn_metadata.fdv2_cache_only = bool(fdv2_cache_only)
        return input_ids, positions

    @staticmethod
    def _graph_seq_batch_sizes(max_num_seqs: int) -> list[int]:
        # Fast-dLLM v2's greedy refinement is very sensitive to tiny bf16
        # differences. Replaying a larger padded CUDA graph changes GEMM shapes
        # versus eager and can flip accepted tokens, so capture exact request
        # counts for lossless graph replay.
        return list(range(1, max_num_seqs + 1))

    def _fdv2_graph_mode(self, attn_metadata) -> str:
        if bool(getattr(attn_metadata, "fdv2_cache_only", False)):
            return _FDV2_GRAPH_SUB_BLOCK
        if int(getattr(attn_metadata, "fdv2_mode", FastDLLMV2Mode.FULL_BUFFER_INIT)) == int(
            FastDLLMV2Mode.FINAL_COMMIT
        ):
            return _FDV2_GRAPH_FINAL_COMMIT
        return _FDV2_GRAPH_FULL_BUFFER

    def _fdv2_graph_q_len(self, mode: str) -> int:
        if mode == _FDV2_GRAPH_SUB_BLOCK:
            return int(self.config.block_size)
        if mode in (_FDV2_GRAPH_FULL_BUFFER, _FDV2_GRAPH_FINAL_COMMIT):
            return int(self.fdv2_attention_block_size)
        raise ValueError(f"Unknown Fast-dLLM v2 CUDA graph mode: {mode}")

    @staticmethod
    def _fdv2_graph_mode_enabled(mode: str) -> bool:
        raw = os.environ.get("DIFFULEX_FDV2_GRAPH_MODES")
        if not raw:
            return True
        aliases = {
            "full": _FDV2_GRAPH_FULL_BUFFER,
            "full_buffer": _FDV2_GRAPH_FULL_BUFFER,
            "full_buffer_init": _FDV2_GRAPH_FULL_BUFFER,
            "sub": _FDV2_GRAPH_SUB_BLOCK,
            "sub_block": _FDV2_GRAPH_SUB_BLOCK,
            "sub_block_cache_only": _FDV2_GRAPH_SUB_BLOCK,
            "commit": _FDV2_GRAPH_FINAL_COMMIT,
            "final": _FDV2_GRAPH_FINAL_COMMIT,
            "final_commit": _FDV2_GRAPH_FINAL_COMMIT,
        }
        enabled = {aliases.get(part.strip(), part.strip()) for part in raw.split(",") if part.strip()}
        return mode in enabled

    def _fdv2_can_run_decode_graph(self, input_ids: torch.Tensor, attn_metadata) -> bool:
        if self.enforce_eager or not bool(getattr(self.config, "enable_full_static_runner", True)):
            return False
        graphs = getattr(self, "graphs", None)
        graph_bs = getattr(self, "graph_bs", None)
        if not graphs or not graph_bs:
            return False
        mode = self._fdv2_graph_mode(attn_metadata)
        if not self._fdv2_graph_mode_enabled(mode):
            return False
        q_len = self._fdv2_graph_q_len(mode)
        num_tokens = int(input_ids.size(0))
        if num_tokens <= 0 or num_tokens % q_len != 0:
            return False
        mode_bs = graph_bs.get(mode)
        return bool(mode_bs) and num_tokens <= max(mode_bs)

    @torch.inference_mode()
    def _fdv2_run_decode_graph(self, input_ids: torch.Tensor, positions: torch.Tensor, attn_metadata):
        mode = self._fdv2_graph_mode(attn_metadata)
        q_len = self._fdv2_graph_q_len(mode)
        num_tokens = int(input_ids.size(0))
        captured_num_tokens = next(x for x in self.graph_bs[mode] if x >= num_tokens)
        captured_num_seqs = captured_num_tokens // q_len
        graph = self.graphs[(mode, captured_num_tokens)]
        graph_vars = self.graph_vars

        num_reqs = attn_metadata.num_reqs
        graph_capacity = int(graph_vars["context_lens"].size(0))
        if captured_num_seqs > graph_capacity:
            raise RuntimeError(
                "Captured Fast-dLLM v2 CUDA graph batch size exceeds graph buffer capacity: "
                f"mode={mode}, captured_num_seqs={captured_num_seqs}, "
                f"graph_capacity={graph_capacity}, captured_num_tokens={captured_num_tokens}, "
                f"num_tokens={num_tokens}"
            )
        if num_reqs > captured_num_seqs:
            raise RuntimeError(
                "Fast-dLLM v2 CUDA graph bucket cannot cover current request count: "
                f"mode={mode}, num_reqs={num_reqs}, captured_num_seqs={captured_num_seqs}, "
                f"num_tokens={num_tokens}, q_len={q_len}"
            )

        self._copy_common_graph_inputs(
            graph_vars,
            attn_metadata,
            input_ids,
            positions,
            num_tokens,
            num_reqs,
        )

        for i in range(num_reqs, captured_num_seqs):
            graph_vars["cu_seqlens_q"][i + 1] = graph_vars["cu_seqlens_q"][i]
            graph_vars["cu_seqlens_k"][i + 1] = graph_vars["cu_seqlens_k"][i]

        restore_fields = (
            "slot_mapping",
            "context_lens",
            "cu_seqlens_q",
            "cu_seqlens_k",
            "valid_slices",
            "status_table",
            "prefix_lens",
            "padded_prefix_lens",
            "page_tables",
            "block_size",
            "fdv2_cache_only",
            "fdv2_mode",
        )
        original_metadata = {field: getattr(attn_metadata, field, None) for field in restore_fields}
        try:
            attn_metadata.slot_mapping = graph_vars["slot_mapping"]
            attn_metadata.context_lens = graph_vars["context_lens"]
            attn_metadata.cu_seqlens_q = graph_vars["cu_seqlens_q"]
            attn_metadata.cu_seqlens_k = graph_vars["cu_seqlens_k"]
            attn_metadata.valid_slices = graph_vars["valid_slices"]
            attn_metadata.status_table = graph_vars["status_table"]
            attn_metadata.prefix_lens = graph_vars["prefix_lens"]
            attn_metadata.padded_prefix_lens = graph_vars["padded_prefix_lens"]
            attn_metadata.page_tables = graph_vars["page_tables"]
            attn_metadata.block_size = self.fdv2_attention_block_size
            attn_metadata.fdv2_cache_only = mode == _FDV2_GRAPH_SUB_BLOCK
            attn_metadata.fdv2_mode = int(
                FastDLLMV2Mode.SUB_BLOCK_REFINE
                if mode == _FDV2_GRAPH_SUB_BLOCK
                else FastDLLMV2Mode.FINAL_COMMIT
                if mode == _FDV2_GRAPH_FINAL_COMMIT
                else FastDLLMV2Mode.FULL_BUFFER_INIT
            )
            self._bind_decode_graph_extra_metadata(attn_metadata, graph_vars, num_tokens)
            graph.replay()
        finally:
            for field, value in original_metadata.items():
                setattr(attn_metadata, field, value)

        if bool(getattr(self, "graph_outputs_are_logits", False)):
            return graph_vars["outputs"][:num_tokens]
        return self.model.compute_logits(graph_vars["outputs"][:num_tokens])

    @torch.inference_mode()
    def run_model_multi_block(self, input_ids: torch.Tensor, positions: torch.Tensor):
        with record_function("diffulex.fast_dllm_v2.model_forward"):
            attn_metadata = self.fetch_attn_metadata()
            if attn_metadata.has_prefill:
                with record_function("diffulex.fast_dllm_v2.eager_prefill"):
                    return self.model.compute_logits(self.model(input_ids, positions))

            if not self._fdv2_can_run_decode_graph(input_ids, attn_metadata):
                with record_function("diffulex.fast_dllm_v2.eager_decode"):
                    return self.model.compute_logits(self.model(input_ids, positions))

            with record_function("diffulex.fast_dllm_v2.cuda_graph_decode"):
                return self._fdv2_run_decode_graph(input_ids, positions, attn_metadata)

    @torch.inference_mode()
    def capture_cudagraph(self):
        set_warming_up(True)
        config = self.config
        hf_config = config.hf_config
        max_num_seqs = min(self.config.max_num_reqs, 512)
        max_num_pages = (config.max_model_len + self.page_size - 1) // self.page_size
        max_q_len = int(self.fdv2_attention_block_size)
        max_num_tokens = max_num_seqs * max_q_len
        device = self._cuda_graph_device()
        capture_logits = self._can_capture_logits_in_graph()
        self.graph_outputs_are_logits = capture_logits
        output_size = self._model_logits_size() if capture_logits else hf_config.hidden_size

        input_ids = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        slot_mapping = torch.full((max_num_tokens,), -1, dtype=torch.int32, device=device)
        context_lens = torch.zeros(max_num_seqs, dtype=torch.int32, device=device)
        page_tables = torch.zeros(max_num_seqs, max_num_pages, dtype=torch.int32, device=device)
        valid_slices = torch.zeros(max_num_seqs, dtype=torch.int32, device=device)
        status_table = torch.zeros(max_num_seqs, dtype=torch.int32, device=device)
        prefix_lens = torch.zeros(max_num_seqs, dtype=torch.int32, device=device)
        padded_prefix_lens = torch.zeros(max_num_seqs, dtype=torch.int32, device=device)
        outputs = torch.zeros(max_num_tokens, output_size, dtype=self._model_logits_dtype(), device=device)

        cu_seqlens_q = torch.zeros(max_num_seqs + 1, dtype=torch.int32, device=device)
        cu_seqlens_k = torch.zeros(max_num_seqs + 1, dtype=torch.int32, device=device)

        graph_vars = dict(
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

        seq_bs_list = self._graph_seq_batch_sizes(max_num_seqs)
        self.graph_bs = {
            mode: [num_seqs * self._fdv2_graph_q_len(mode) for num_seqs in seq_bs_list]
            for mode in (_FDV2_GRAPH_FULL_BUFFER, _FDV2_GRAPH_SUB_BLOCK, _FDV2_GRAPH_FINAL_COMMIT)
        }
        self.graphs = {}
        self.graph_pool = None

        try:
            for mode in (_FDV2_GRAPH_FULL_BUFFER, _FDV2_GRAPH_SUB_BLOCK, _FDV2_GRAPH_FINAL_COMMIT):
                if not self._fdv2_graph_mode_enabled(mode):
                    continue
                q_len = self._fdv2_graph_q_len(mode)
                is_sub_block = mode == _FDV2_GRAPH_SUB_BLOCK
                is_final_commit = mode == _FDV2_GRAPH_FINAL_COMMIT
                for num_tokens in tqdm(
                    reversed(self.graph_bs[mode]),
                    desc=f"Capturing Fast-dLLM v2 CUDA graphs ({mode})",
                ):
                    num_seqs = num_tokens // q_len
                    input_ids.zero_()
                    positions.zero_()
                    slot_mapping.fill_(-1)
                    page_tables.fill_(0)
                    context_lens.zero_()
                    valid_slices.zero_()
                    status_table.zero_()
                    prefix_lens.zero_()
                    padded_prefix_lens.zero_()

                    for i in range(max_num_seqs + 1):
                        cu_seqlens_q[i] = i * q_len
                        cu_seqlens_k[i] = i * config.max_model_len
                    for i in range(num_seqs):
                        valid_slices[i] = (i + 1) * q_len
                    # Capture decode graphs with a non-empty cache prefix. The
                    # Triton attention kernel must record the cache-attention
                    # path because real Fast-dLLM v2 decode replays against
                    # prefix/current-buffer KV cache even for full-buffer init
                    # and final commit.
                    context_lens[:num_seqs] = config.max_model_len
                    if is_sub_block:
                        status_table[:num_seqs] = 2
                    else:
                        status_table[:num_seqs] = 1

                    self.set_attn_metadata(
                        False,
                        slot_mapping=slot_mapping[:num_tokens],
                        need_kv_cache_store=True,
                        context_lens=context_lens[:num_seqs],
                        cu_seqlens_q=cu_seqlens_q[: num_seqs + 1],
                        cu_seqlens_k=cu_seqlens_k[: num_seqs + 1],
                        max_seqlen_q=q_len,
                        max_seqlen_k=config.max_model_len,
                        page_size=config.kv_cache_page_size,
                        page_tables=page_tables[:num_seqs],
                        block_size=self.fdv2_attention_block_size,
                        kv_cache_layout=config.kv_cache_layout,
                        fdv2_cache_only=is_sub_block,
                        fdv2_mode=int(
                            FastDLLMV2Mode.SUB_BLOCK_REFINE
                            if is_sub_block
                            else FastDLLMV2Mode.FINAL_COMMIT
                            if is_final_commit
                            else FastDLLMV2Mode.FULL_BUFFER_INIT
                        ),
                    )
                    attn_metadata = self.fetch_attn_metadata()
                    attn_metadata.init_multi_block(
                        valid_slices=valid_slices[:num_seqs],
                        buffer_size=config.buffer_size,
                        is_prefix_full=self.is_prefix_full,
                        status_table=status_table[:num_seqs],
                        prefix_lens=prefix_lens[:num_seqs],
                        padded_prefix_lens=padded_prefix_lens[:num_seqs],
                        mask_prefix_hole=bool(getattr(self, "mask_prefix_hole", False)),
                        prefix_causal=bool(getattr(self, "prefix_causal", False)),
                    )
                    attn_metadata.block_size = self.fdv2_attention_block_size
                    attn_metadata.fdv2_cache_only = is_sub_block
                    attn_metadata.fdv2_mode = int(
                        FastDLLMV2Mode.SUB_BLOCK_REFINE
                        if is_sub_block
                        else FastDLLMV2Mode.FINAL_COMMIT
                        if is_final_commit
                        else FastDLLMV2Mode.FULL_BUFFER_INIT
                    )
                    self._init_graph_capture_extra_metadata(attn_metadata, graph_vars, num_tokens)

                    graph = self._capture_model_forward_graph(
                        input_ids,
                        positions,
                        outputs,
                        num_tokens,
                        allow_compile=True,
                        capture_logits=capture_logits,
                    )
                    if self.graph_pool is None:
                        self.graph_pool = graph.pool()
                    self.graphs[(mode, num_tokens)] = graph
                    torch.cuda.synchronize()
                    self.reset_attn_metadata()

            self.graph_vars = graph_vars
        finally:
            reset_warming_up()
