from __future__ import annotations

import gc
from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist
import torch._inductor.config as inductor_config

from tqdm import tqdm

from diffulex.attention.metadata import (
    AttnMetaDataBase,
    set_warming_up,
    reset_warming_up,
)
from diffulex.engine.request import AutoReq, DllmReq
from diffulex.engine.status import DllmReqStatus
from diffulex.engine.model_runner import ModelRunnerBase
from diffulex.logger import get_logger
from diffulex.strategy_template.multi_block.engine.full_static_runner import FullStaticRunner
from diffulex.vllm_compat import vllm_graph_capture

logger = get_logger(__name__)


@contextmanager
def _freeze_gc_for_cudagraph():
    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


class MultiBlockModelRunnerTemplate(ModelRunnerBase):
    def _full_static_runner(self) -> FullStaticRunner:
        runner = getattr(self, "full_static_runner", None)
        if runner is None:
            runner = FullStaticRunner(self)
            self.full_static_runner = runner
        return runner

    @staticmethod
    def _round_up_to_multiple(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple

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

    def _torch_compile_enabled_for_capture(self) -> bool:
        # dInfer only compiles selected decode graph shapes after model-side
        # capture/compile boundaries are prepared. Keep this disabled here until
        # those boundaries are fully ported; eager CUDA graph capture is stable
        # and is also what dInfer uses for prefill.
        return bool(
            getattr(self.config, "enable_torch_compile", True)
            and getattr(self.config, "enable_cudagraph_torch_compile", False)
        )

    @contextmanager
    def _patch_model_forward_for_cuda_graph_capture(self, num_tokens: int):
        if not self._torch_compile_enabled_for_capture():
            yield False
            return

        original_forward = self.model.forward
        mode = str(getattr(self.config, "torch_compile_mode", "reduce-overhead") or "reduce-overhead")
        compile_config_patch = {
            # We already wrap the compiled forward in our own CUDA graph.
            # Inductor's internal cudagraph trees try to replay during outer
            # capture and fail with "Cannot prepare for replay during capturing".
            "triton.cudagraphs": False,
            "triton.cudagraph_trees": False,
        }
        try:
            with inductor_config.patch(compile_config_patch):
                self.model.forward = torch.compile(
                    torch.no_grad()(original_forward),
                    mode=mode,
                    fullgraph=False,
                    dynamic=False,
                )
                yield True
        finally:
            self.model.forward = original_forward

    def _get_graph_capture_stream(self) -> torch.cuda.Stream:
        stream = getattr(self, "graph_capture_stream", None)
        if stream is None:
            stream = torch.cuda.Stream()
            self.graph_capture_stream = stream
        return stream

    def _get_graph_pool(self):
        pool = getattr(self, "graph_pool", None)
        if pool is None:
            pool = torch.cuda.graph_pool_handle()
            self.graph_pool = pool
        return pool

    def _graph_capture_barrier(self) -> None:
        if self.world_size <= 1:
            return
        try:
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        except Exception:
            logger.debug("Distributed barrier failed before CUDA graph capture on rank %s.", self.rank, exc_info=True)
            raise

    def _capture_model_forward_graph(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        outputs: torch.Tensor,
        num_tokens: int,
        *,
        allow_compile: bool = False,
        capture_logits: bool = False,
    ) -> torch.cuda.CUDAGraph:
        def run_once() -> None:
            hidden_states = self.model(input_ids[:num_tokens], positions[:num_tokens])
            if capture_logits:
                outputs[:num_tokens] = self.model.compute_logits(hidden_states)
            else:
                outputs[:num_tokens] = hidden_states

        stream = self._get_graph_capture_stream()
        pool = self._get_graph_pool()
        graph = torch.cuda.CUDAGraph()
        compile_capture = allow_compile and self._torch_compile_enabled_for_capture()

        patch_context = (
            self._patch_model_forward_for_cuda_graph_capture(num_tokens) if compile_capture else nullcontext(False)
        )
        with _freeze_gc_for_cudagraph(), patch_context:
            for _ in range(2):
                torch.cuda.synchronize()
                self._graph_capture_barrier()
                with torch.cuda.stream(stream):
                    run_once()
                stream.synchronize()

            torch.cuda.synchronize()
            self._graph_capture_barrier()
            with vllm_graph_capture(stream, pool) as capture_stream:
                with torch.cuda.graph(graph, pool=pool, stream=capture_stream):
                    run_once()
            stream.synchronize()
        return graph

    @staticmethod
    def _graph_seq_batch_sizes(max_num_seqs: int) -> list[int]:
        """CUDA graph capture buckets, always bounded by max_num_seqs."""
        if max_num_seqs <= 0:
            return []

        seq_bs = [1, 2, 4, 8]
        seq_bs.extend(range(16, max_num_seqs + 1, 16))
        seq_bs.append(max_num_seqs)
        return sorted({bs for bs in seq_bs if 1 <= bs <= max_num_seqs})

    @staticmethod
    def _cached_prefix_len(req: DllmReq) -> int:
        return int(req.contiguous_in_cache_prefix_len)

    def _prepare_prefill_req(self: ModelRunnerBase, req: DllmReq):
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

    def _prepare_decode_req(self: ModelRunnerBase, req: DllmReq):
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

    def _prefill_graph_max_len(self) -> int:
        configured = int(getattr(self.config, "prefill_cudagraph_max_len", 0) or 0)
        max_len = configured if configured > 0 else int(self.config.max_model_len)
        return min(
            self._round_up_to_multiple(max_len, self.block_size),
            self._round_up_to_multiple(int(self.config.max_num_batched_tokens), self.block_size),
        )

    def _prefill_graph_bucket_len(self, num_tokens: int) -> int:
        return self._round_up_to_multiple(int(num_tokens), self.block_size)

    def _prefill_graph_req_capacity(self) -> int:
        # One extra row is reserved for a padding-only phantom request when a
        # runtime prefill length is smaller than the captured bucket.
        return int(self.config.max_num_reqs) + 1

    @staticmethod
    def _cuda_graph_device() -> torch.device:
        return torch.device(f"cuda:{torch.cuda.current_device()}")

    def _model_hidden_dtype(self) -> torch.dtype:
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.get_default_dtype()

    def _model_logits_dtype(self) -> torch.dtype:
        return self._model_hidden_dtype()

    def _model_logits_size(self) -> int:
        lm_head = getattr(self.model, "lm_head", None)
        partition_size = getattr(lm_head, "num_embeddings_per_partition", None)
        if partition_size is not None:
            return int(partition_size)
        vocab_size = getattr(self.config, "tokenizer_vocab_size", None) or getattr(self.config.hf_config, "vocab_size")
        if self.world_size <= 1:
            return int(vocab_size)
        return int(vocab_size) // int(self.world_size)

    def _can_capture_logits_in_graph(self) -> bool:
        # TP lm_head captures all-gather/NCCL and rank-local None outputs.
        # Logits buffers are vocab-sized, so keep the first version to the
        # single-request SDAR eval path before enabling broader serving shapes.
        return self.world_size == 1 and int(self.config.max_num_reqs) == 1

    def _ensure_runtime_static_buffers(
        self,
        *,
        token_capacity: int,
        req_capacity: int,
        page_capacity: int,
    ) -> dict[str, torch.Tensor]:
        device = self._cuda_graph_device()
        token_capacity = max(1, int(token_capacity), int(self.config.max_num_batched_tokens))
        req_capacity = max(1, int(req_capacity), int(self.config.max_num_reqs) + 1)
        page_capacity = max(
            1,
            int(page_capacity),
            (int(self.config.max_model_len) + self.page_size - 1) // self.page_size,
        )

        buffers = getattr(self, "_runtime_static_buffers", None)
        if buffers is not None:
            if (
                int(buffers["input_ids"].numel()) >= token_capacity
                and int(buffers["context_lens"].numel()) >= req_capacity
                and int(buffers["page_tables"].size(1)) >= page_capacity
            ):
                return buffers

        buffers = {
            "input_ids": torch.empty(token_capacity, dtype=torch.int64, device=device),
            "positions": torch.empty(token_capacity, dtype=torch.int64, device=device),
            "slot_mapping": torch.empty(token_capacity, dtype=torch.int32, device=device),
            "context_lens": torch.empty(req_capacity, dtype=torch.int32, device=device),
            "cu_seqlens_q": torch.empty(req_capacity + 1, dtype=torch.int32, device=device),
            "cu_seqlens_k": torch.empty(req_capacity + 1, dtype=torch.int32, device=device),
            "valid_slices": torch.empty(req_capacity, dtype=torch.int32, device=device),
            "status_table": torch.empty(req_capacity, dtype=torch.int32, device=device),
            "prefix_lens": torch.empty(req_capacity, dtype=torch.int32, device=device),
            "padded_prefix_lens": torch.empty(req_capacity, dtype=torch.int32, device=device),
            "page_tables": torch.empty(req_capacity, page_capacity, dtype=torch.int32, device=device),
        }
        self._runtime_static_buffers = buffers
        return buffers

    @staticmethod
    def _cpu_tensor(values, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.tensor(values, dtype=dtype, device="cpu")
        if tensor.numel() > 0 and torch.cuda.is_available():
            tensor = tensor.pin_memory()
        return tensor

    def _copy_1d_to_runtime_static(
        self,
        buffers: dict[str, torch.Tensor],
        name: str,
        values,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        length = len(values)
        dst = buffers[name][:length]
        if length:
            dst.copy_(self._cpu_tensor(values, dtype), non_blocking=True)
        return dst

    def _prepare_static_page_tables(
        self,
        buffers: dict[str, torch.Tensor],
        reqs: list[DllmReq],
    ) -> torch.Tensor:
        if not reqs:
            return buffers["page_tables"][:0, :1]

        max_len = max(1, max(len(req.page_table) for req in reqs))
        page_tables = buffers["page_tables"][: len(reqs), :max_len]
        page_tables.fill_(-1)
        rows = [req.page_table + [-1] * (max_len - len(req.page_table)) for req in reqs]
        page_tables.copy_(self._cpu_tensor(rows, torch.int32), non_blocking=True)
        return page_tables

    def _can_use_prefill_graph(self, attn_metadata: AttnMetaDataBase, num_tokens: int) -> bool:
        if self.enforce_eager or not bool(getattr(self.config, "enable_prefill_cudagraph", True)):
            return False
        if num_tokens <= 0 or attn_metadata.status_table is None:
            return False
        if not bool((attn_metadata.status_table == 0).all().item()):
            return False
        bucket_len = self._prefill_graph_bucket_len(num_tokens)
        if bucket_len > self._prefill_graph_max_len():
            return False
        if bucket_len > int(self.config.max_num_batched_tokens):
            return False
        needs_padding_row = bucket_len > num_tokens
        required_rows = int(attn_metadata.status_table.numel()) + (1 if needs_padding_row else 0)
        return required_rows <= self._prefill_graph_req_capacity()

    def _init_prefill_graph_extra_metadata(
        self,
        attn_metadata: AttnMetaDataBase,
        graph_vars: dict[str, torch.Tensor],
        bucket_len: int,
    ) -> None:
        return None

    def _prefill_graph_extra_vars(self, bucket_len: int, device: torch.device) -> dict[str, torch.Tensor]:
        return {}

    def _bind_prefill_graph_extra_metadata(
        self,
        attn_metadata: AttnMetaDataBase,
        graph_vars: dict[str, torch.Tensor],
        bucket_len: int,
        runtime_num_tokens: int,
        source_attn_metadata: AttnMetaDataBase | None = None,
    ) -> None:
        return None

    def _bind_decode_graph_extra_metadata(
        self,
        attn_metadata: AttnMetaDataBase,
        graph_vars: dict[str, torch.Tensor],
        num_tokens: int,
    ) -> None:
        return None

    def _set_replay_prefill_graph_metadata(
        self,
        graph_vars: dict[str, torch.Tensor],
        bucket_len: int,
        graph_num_reqs: int,
    ) -> AttnMetaDataBase:
        self.set_attn_metadata(
            is_prefill=[True] * graph_num_reqs,
            slot_mapping=graph_vars["slot_mapping"],
            context_lens=graph_vars["context_lens"][:graph_num_reqs],
            cu_seqlens_q=graph_vars["cu_seqlens_q"][: graph_num_reqs + 1],
            cu_seqlens_k=graph_vars["cu_seqlens_k"][: graph_num_reqs + 1],
            max_seqlen_q=bucket_len,
            max_seqlen_k=bucket_len,
            page_size=self.config.kv_cache_page_size,
            page_tables=graph_vars["page_tables"][:graph_num_reqs],
            block_size=self.block_size,
            kv_cache_layout=self.config.kv_cache_layout,
        )
        attn_metadata: AttnMetaDataBase = self.fetch_attn_metadata()
        attn_metadata.init_multi_block(
            valid_slices=graph_vars["valid_slices"][:graph_num_reqs],
            buffer_size=self.config.buffer_size,
            is_prefix_full=self.is_prefix_full,
            status_table=graph_vars["status_table"][:graph_num_reqs],
            prefix_lens=graph_vars["prefix_lens"][:graph_num_reqs],
            padded_prefix_lens=graph_vars["padded_prefix_lens"][:graph_num_reqs],
        )
        return attn_metadata

    @torch.inference_mode()
    def _capture_prefill_cudagraph(self, bucket_len: int):
        if not hasattr(self, "prefill_graphs"):
            self.prefill_graphs = {}
        if bucket_len in self.prefill_graphs:
            return self.prefill_graphs[bucket_len]

        config = self.config
        hf_config = config.hf_config
        req_capacity = self._prefill_graph_req_capacity()
        max_num_pages = (max(config.max_model_len, bucket_len) + self.page_size - 1) // self.page_size
        device = self._cuda_graph_device()

        input_ids = torch.zeros(bucket_len, dtype=torch.int64, device=device)
        positions = torch.zeros(bucket_len, dtype=torch.int64, device=device)
        slot_mapping = torch.full((bucket_len,), -1, dtype=torch.int32, device=device)
        context_lens = torch.zeros(req_capacity, dtype=torch.int32, device=device)
        page_tables = torch.full((req_capacity, max_num_pages), -1, dtype=torch.int32, device=device)
        valid_slices = torch.zeros(req_capacity, dtype=torch.int32, device=device)
        status_table = torch.zeros(req_capacity, dtype=torch.int32, device=device)
        prefix_lens = torch.zeros(req_capacity, dtype=torch.int32, device=device)
        padded_prefix_lens = torch.zeros(req_capacity, dtype=torch.int32, device=device)
        outputs = torch.zeros(bucket_len, hf_config.hidden_size, dtype=self._model_hidden_dtype(), device=device)

        cu_seqlens = torch.full((req_capacity + 1,), bucket_len, dtype=torch.int32, device=device)
        cu_seqlens[0] = 0
        valid_slices[0] = bucket_len

        self.set_attn_metadata(
            is_prefill=[True] * req_capacity,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=bucket_len,
            max_seqlen_k=bucket_len,
            page_size=config.kv_cache_page_size,
            page_tables=page_tables,
            block_size=self.block_size,
            kv_cache_layout=config.kv_cache_layout,
        )
        attn_metadata: AttnMetaDataBase = self.fetch_attn_metadata()
        attn_metadata.init_multi_block(
            valid_slices=valid_slices,
            buffer_size=config.buffer_size,
            is_prefix_full=self.is_prefix_full,
            status_table=status_table,
            prefix_lens=prefix_lens,
            padded_prefix_lens=padded_prefix_lens,
        )
        graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            page_tables=page_tables,
            valid_slices=valid_slices,
            status_table=status_table,
            prefix_lens=prefix_lens,
            padded_prefix_lens=padded_prefix_lens,
            outputs=outputs,
        )
        graph_vars.update(self._prefill_graph_extra_vars(bucket_len, device))
        self._init_prefill_graph_extra_metadata(attn_metadata, graph_vars, bucket_len)

        graph = self._capture_model_forward_graph(input_ids, positions, outputs, bucket_len)
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        torch.cuda.synchronize()
        self.reset_attn_metadata()

        self.prefill_graphs[bucket_len] = (graph, graph_vars, req_capacity)
        return self.prefill_graphs[bucket_len]

    def _copy_common_graph_inputs(
        self,
        graph_vars: dict[str, torch.Tensor],
        attn_metadata: AttnMetaDataBase,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        num_tokens: int,
        num_reqs: int,
    ) -> None:
        for key, value in graph_vars.items():
            if key == "outputs":
                continue
            if key in ("slot_mapping", "page_tables"):
                value.fill_(-1)
            else:
                value.zero_()

        graph_vars["input_ids"][:num_tokens] = input_ids
        graph_vars["positions"][:num_tokens] = positions
        graph_vars["slot_mapping"][:num_tokens] = attn_metadata.slot_mapping
        graph_vars["context_lens"][:num_reqs] = attn_metadata.context_lens
        graph_vars["cu_seqlens_q"][: num_reqs + 1] = attn_metadata.cu_seqlens_q
        graph_vars["cu_seqlens_k"][: num_reqs + 1] = attn_metadata.cu_seqlens_k
        graph_vars["valid_slices"][:num_reqs] = attn_metadata.valid_slices
        graph_vars["status_table"][:num_reqs] = attn_metadata.status_table
        graph_vars["prefix_lens"][:num_reqs] = attn_metadata.prefix_lens
        graph_vars["padded_prefix_lens"][:num_reqs] = attn_metadata.padded_prefix_lens
        pt_w = attn_metadata.page_tables.size(1)
        graph_vars["page_tables"][:num_reqs, :pt_w] = attn_metadata.page_tables
        if pt_w < graph_vars["page_tables"].size(1):
            graph_vars["page_tables"][:, pt_w:].fill_(-1)

    @torch.inference_mode()
    def _run_prefill_cudagraph(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttnMetaDataBase,
    ):
        num_tokens = int(input_ids.size(0))
        bucket_len = self._prefill_graph_bucket_len(num_tokens)
        graph, graph_vars, req_capacity = self._capture_prefill_cudagraph(bucket_len)
        num_reqs = int(attn_metadata.status_table.numel())
        has_padding = bucket_len > num_tokens
        graph_num_reqs = num_reqs + (1 if has_padding else 0)
        if graph_num_reqs > req_capacity:
            return self.model.compute_logits(self.model(input_ids, positions))

        self._copy_common_graph_inputs(graph_vars, attn_metadata, input_ids, positions, num_tokens, num_reqs)
        if has_padding:
            graph_vars["cu_seqlens_q"][num_reqs + 1] = bucket_len
            graph_vars["cu_seqlens_k"][num_reqs + 1] = bucket_len
            graph_vars["valid_slices"][num_reqs] = bucket_len
        for i in range(graph_num_reqs, req_capacity):
            graph_vars["cu_seqlens_q"][i + 1] = graph_vars["cu_seqlens_q"][i]
            graph_vars["cu_seqlens_k"][i + 1] = graph_vars["cu_seqlens_k"][i]

        source_attn_metadata = attn_metadata
        attn_metadata = self._set_replay_prefill_graph_metadata(graph_vars, bucket_len, graph_num_reqs)
        self._bind_prefill_graph_extra_metadata(
            attn_metadata,
            graph_vars,
            bucket_len,
            num_tokens,
            source_attn_metadata=source_attn_metadata,
        )

        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:num_tokens])

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

    def prepare_idle_multi_block(self: ModelRunnerBase):
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
        self: ModelRunnerBase,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        # Idle ranks must enter model forward so EP dispatch/combine collectives
        # see all participants, but they have no local logits to sample.
        _ = self.model(input_ids, positions)

    @torch.inference_mode()
    def run_model_multi_block(
        self: ModelRunnerBase,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ):
        attn_metadata: AttnMetaDataBase = self.fetch_attn_metadata()
        full_runner = self._full_static_runner()

        if (attn_metadata.status_table == 0).any():
            if full_runner.can_run_prefill(attn_metadata, int(input_ids.size(0))):
                return full_runner.run_prefill(input_ids, positions, attn_metadata)
            return self.model.compute_logits(self.model(input_ids, positions))

        if not full_runner.can_run_decode(input_ids):
            return self.model.compute_logits(self.model(input_ids, positions))
        return full_runner.run_decode(input_ids, positions, attn_metadata)

    def run_multi_block(self: ModelRunnerBase, reqs: list[DllmReq]) -> list[int]:
        return self._run_multi_block_subgroup(reqs)

    def _run_multi_block_subgroup(self: ModelRunnerBase, reqs: list[DllmReq]):
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
        for i in range(max_num_seqs + 1):
            cu_seqlens_q[i] = i * chunk_size

        cu_seqlens_k = torch.zeros(max_num_seqs + 1, dtype=torch.int32, device=device)
        for i in range(max_num_seqs + 1):
            cu_seqlens_k[i] = i * config.max_model_len

        self.graph_bs = []
        seq_bs_list = self._graph_seq_batch_sizes(max_num_seqs)
        for num_seqs in seq_bs_list:
            self.graph_bs.append(num_seqs * chunk_size)
        self.graphs = {}
        self.prefill_graphs = {}
        self.graph_pool = None

        for num_tokens in tqdm(reversed(self.graph_bs), desc="Capturing CUDA graphs"):
            num_seqs = num_tokens // chunk_size

            self.set_attn_metadata(
                False,
                slot_mapping=slot_mapping[:num_tokens],
                context_lens=context_lens[:num_seqs],
                cu_seqlens_q=cu_seqlens_q[: num_seqs + 1],
                cu_seqlens_k=cu_seqlens_k[: num_seqs + 1],
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
                is_prefix_full=self.is_prefix_full,
                status_table=status_table[:num_seqs],
                prefix_lens=prefix_lens[:num_seqs],
                padded_prefix_lens=padded_prefix_lens[:num_seqs],
            )

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
