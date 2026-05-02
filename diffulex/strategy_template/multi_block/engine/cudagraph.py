from __future__ import annotations

import gc
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from tqdm import tqdm

from diffulex.attention.metadata import AttnMetaDataBase, warming_up_context
from diffulex.distributed.parallel_state import fetch_parallel_state
from diffulex.engine.request import DllmReq
from diffulex.logger import get_logger

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


@dataclass
class MultiBlockGraphBuffers:
    common: dict[str, torch.Tensor]
    extra: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class MultiBlockCudaGraphState:
    decode_bucket_tokens: list[int] = field(default_factory=list)
    decode_graphs: dict[int, torch.cuda.CUDAGraph] = field(default_factory=dict)
    decode_buffers: MultiBlockGraphBuffers | None = None


class MultiBlockCudaGraphMixin:
    @staticmethod
    def _round_up_to_multiple(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple

    def _ensure_cuda_graph_state(self) -> MultiBlockCudaGraphState:
        state = getattr(self, "cuda_graph_state", None)
        if state is None:
            state = MultiBlockCudaGraphState()
            self.cuda_graph_state = state
        return state

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
    ) -> torch.cuda.CUDAGraph:
        def run_once() -> None:
            outputs[:num_tokens] = self.model(input_ids[:num_tokens], positions[:num_tokens])

        stream = self._get_graph_capture_stream()
        pool = self._get_graph_pool()
        graph = torch.cuda.CUDAGraph()
        with _freeze_gc_for_cudagraph():
            for _ in range(2):
                torch.cuda.synchronize()
                self._graph_capture_barrier()
                with torch.cuda.stream(stream):
                    run_once()
                stream.synchronize()

            torch.cuda.synchronize()
            self._graph_capture_barrier()
            with torch.cuda.stream(stream):
                parallel_state = fetch_parallel_state()
                with parallel_state.sglang_graph_capture(stream):
                    with torch.cuda.graph(graph, pool=pool, stream=stream):
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

    @property
    def cuda_graph_device(self) -> torch.device:
        return torch.device(f"cuda:{torch.cuda.current_device()}")

    @property
    def model_hidden_dtype(self) -> torch.dtype:
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.get_default_dtype()

    @property
    def decode_graph_chunk_size(self) -> int:
        return int(self.config.block_size) * int(self.config.buffer_size)

    @property
    def decode_graph_max_num_seqs(self) -> int:
        return min(int(self.config.max_num_reqs), 512)

    def _create_common_graph_buffers(
        self,
        *,
        token_capacity: int,
        req_capacity: int,
        page_capacity: int,
    ) -> dict[str, torch.Tensor]:
        device = self.cuda_graph_device
        hidden_size = int(self.config.hf_config.hidden_size)
        return {
            "input_ids": torch.zeros(token_capacity, dtype=torch.int64, device=device),
            "positions": torch.zeros(token_capacity, dtype=torch.int64, device=device),
            "slot_mapping": torch.full((token_capacity,), -1, dtype=torch.int32, device=device),
            "context_lens": torch.zeros(req_capacity, dtype=torch.int32, device=device),
            "cu_seqlens_q": torch.zeros(req_capacity + 1, dtype=torch.int32, device=device),
            "cu_seqlens_k": torch.zeros(req_capacity + 1, dtype=torch.int32, device=device),
            "page_tables": torch.full((req_capacity, page_capacity), -1, dtype=torch.int32, device=device),
            "valid_slices": torch.zeros(req_capacity, dtype=torch.int32, device=device),
            "status_table": torch.zeros(req_capacity, dtype=torch.int32, device=device),
            "prefix_lens": torch.zeros(req_capacity, dtype=torch.int32, device=device),
            "padded_prefix_lens": torch.zeros(req_capacity, dtype=torch.int32, device=device),
            "outputs": torch.zeros(
                token_capacity,
                hidden_size,
                dtype=self.model_hidden_dtype,
                device=device,
            ),
        }

    def _can_fit_common_graph_inputs(
        self,
        common_vars: dict[str, torch.Tensor],
        attn_metadata: AttnMetaDataBase,
        num_tokens: int,
        num_reqs: int,
    ) -> bool:
        if num_tokens > int(common_vars["input_ids"].numel()):
            return False
        if num_reqs > int(common_vars["context_lens"].numel()):
            return False
        page_tables = attn_metadata.page_tables
        if page_tables is not None and int(page_tables.size(1)) > int(common_vars["page_tables"].size(1)):
            return False
        return True

    def _ensure_runtime_static_buffers(
        self,
        *,
        token_capacity: int,
        req_capacity: int,
        page_capacity: int,
    ) -> dict[str, torch.Tensor]:
        device = self.cuda_graph_device
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

    def _decode_graph_extra_vars(self, max_num_tokens: int, device: torch.device) -> dict[str, torch.Tensor]:
        return {}

    def _init_decode_graph_extra_metadata(
        self,
        attn_metadata: AttnMetaDataBase,
        graph_vars: dict[str, torch.Tensor],
        num_tokens: int,
    ) -> None:
        return None

    def _bind_decode_graph_extra_metadata(
        self,
        attn_metadata: AttnMetaDataBase,
        graph_vars: dict[str, torch.Tensor],
        num_tokens: int,
        source_attn_metadata: AttnMetaDataBase | None = None,
    ) -> None:
        return None

    def _can_use_decode_graph_extra(
        self,
        attn_metadata: AttnMetaDataBase,
        num_tokens: int,
        *,
        captured_num_tokens: int | None = None,
    ) -> bool:
        return True

    def _select_decode_bucket(self, num_tokens: int) -> int | None:
        state = getattr(self, "cuda_graph_state", None)
        if state is None:
            return None
        for bucket in state.decode_bucket_tokens:
            if bucket >= num_tokens:
                return bucket
        return None

    def _can_use_prefill_graph(self, attn_metadata: AttnMetaDataBase, num_tokens: int) -> bool:
        return False

    def _can_use_decode_graph(self, attn_metadata: AttnMetaDataBase, input_ids: torch.Tensor) -> bool:
        if self.enforce_eager:
            return False
        num_tokens = int(input_ids.size(0))
        if num_tokens <= 0:
            return False
        state = getattr(self, "cuda_graph_state", None)
        if state is None or state.decode_buffers is None or not state.decode_bucket_tokens:
            return False
        captured_num_tokens = self._select_decode_bucket(num_tokens)
        if captured_num_tokens is None:
            return False
        num_reqs = int(attn_metadata.num_reqs)
        if not self._can_fit_common_graph_inputs(state.decode_buffers.common, attn_metadata, num_tokens, num_reqs):
            return False
        return self._can_use_decode_graph_extra(
            attn_metadata,
            num_tokens,
            captured_num_tokens=captured_num_tokens,
        )

    def _set_replay_decode_graph_metadata(
        self,
        common_vars: dict[str, torch.Tensor],
        captured_num_seqs: int,
    ) -> AttnMetaDataBase:
        self.set_attn_metadata(
            is_prefill=[False] * captured_num_seqs,
            slot_mapping=common_vars["slot_mapping"],
            context_lens=common_vars["context_lens"][:captured_num_seqs],
            cu_seqlens_q=common_vars["cu_seqlens_q"][: captured_num_seqs + 1],
            cu_seqlens_k=common_vars["cu_seqlens_k"][: captured_num_seqs + 1],
            max_seqlen_q=self.decode_graph_chunk_size,
            max_seqlen_k=int(self.config.max_model_len),
            page_size=self.config.kv_cache_page_size,
            page_tables=common_vars["page_tables"][:captured_num_seqs],
            block_size=self.block_size,
            kv_cache_layout=self.config.kv_cache_layout,
        )
        attn_metadata: AttnMetaDataBase = self.fetch_attn_metadata()
        attn_metadata.init_multi_block(
            valid_slices=common_vars["valid_slices"][:captured_num_seqs],
            buffer_size=self.config.buffer_size,
            is_prefix_full=self.is_prefix_full,
            status_table=common_vars["status_table"][:captured_num_seqs],
            prefix_lens=common_vars["prefix_lens"][:captured_num_seqs],
            padded_prefix_lens=common_vars["padded_prefix_lens"][:captured_num_seqs],
        )
        return attn_metadata

    def _copy_common_graph_inputs(
        self,
        common_vars: dict[str, torch.Tensor],
        attn_metadata: AttnMetaDataBase,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        num_tokens: int,
        num_reqs: int,
    ) -> None:
        for key, value in common_vars.items():
            if key == "outputs":
                continue
            if key in ("slot_mapping", "page_tables"):
                value.fill_(-1)
            else:
                value.zero_()

        common_vars["input_ids"][:num_tokens] = input_ids
        common_vars["positions"][:num_tokens] = positions
        common_vars["slot_mapping"][:num_tokens] = attn_metadata.slot_mapping
        common_vars["context_lens"][:num_reqs] = attn_metadata.context_lens
        common_vars["cu_seqlens_q"][: num_reqs + 1] = attn_metadata.cu_seqlens_q
        common_vars["cu_seqlens_k"][: num_reqs + 1] = attn_metadata.cu_seqlens_k
        common_vars["valid_slices"][:num_reqs] = attn_metadata.valid_slices
        common_vars["status_table"][:num_reqs] = attn_metadata.status_table
        common_vars["prefix_lens"][:num_reqs] = attn_metadata.prefix_lens
        common_vars["padded_prefix_lens"][:num_reqs] = attn_metadata.padded_prefix_lens
        pt_w = int(attn_metadata.page_tables.size(1))
        common_vars["page_tables"][:num_reqs, :pt_w] = attn_metadata.page_tables
        if pt_w < int(common_vars["page_tables"].size(1)):
            common_vars["page_tables"][:, pt_w:].fill_(-1)

    @torch.inference_mode()
    def _run_decode_cudagraph(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttnMetaDataBase,
    ):
        num_tokens = int(input_ids.size(0))
        captured_num_tokens = self._select_decode_bucket(num_tokens)
        if captured_num_tokens is None:
            return self.model.compute_logits(self.model(input_ids, positions))

        state = self._ensure_cuda_graph_state()
        graph = state.decode_graphs.get(captured_num_tokens)
        if graph is None or state.decode_buffers is None:
            raise RuntimeError(
                "CUDA graph runtime state is inconsistent: missing decode graph or buffers for "
                f"captured_num_tokens={captured_num_tokens}."
            )

        captured_num_seqs = captured_num_tokens // self.decode_graph_chunk_size
        common_vars = state.decode_buffers.common
        extra_vars = state.decode_buffers.extra
        graph_capacity = int(common_vars["context_lens"].size(0))
        if captured_num_seqs > graph_capacity:
            raise RuntimeError(
                "Captured CUDA graph batch size exceeds allocated graph buffer capacity: "
                f"captured_num_seqs={captured_num_seqs}, graph_capacity={graph_capacity}, "
                f"captured_num_tokens={captured_num_tokens}, num_tokens={num_tokens}, "
                f"max_num_reqs={self.config.max_num_reqs}, "
                f"block_size={self.config.block_size}, "
                f"buffer_size={self.config.buffer_size}"
            )

        num_reqs = int(attn_metadata.num_reqs)
        if not self._can_fit_common_graph_inputs(common_vars, attn_metadata, num_tokens, num_reqs):
            return self.model.compute_logits(self.model(input_ids, positions))

        self._copy_common_graph_inputs(common_vars, attn_metadata, input_ids, positions, num_tokens, num_reqs)
        for i in range(num_reqs, captured_num_seqs):
            common_vars["cu_seqlens_q"][i + 1] = common_vars["cu_seqlens_q"][i]
            common_vars["cu_seqlens_k"][i + 1] = common_vars["cu_seqlens_k"][i]

        graph_attn_metadata = self._set_replay_decode_graph_metadata(common_vars, captured_num_seqs)
        self._bind_decode_graph_extra_metadata(
            graph_attn_metadata,
            extra_vars,
            num_tokens,
            source_attn_metadata=attn_metadata,
        )

        graph.replay()
        return self.model.compute_logits(common_vars["outputs"][:num_tokens])

    @torch.inference_mode()
    def capture_cudagraph_multi_block(self) -> None:
        config = self.config
        max_num_seqs = self.decode_graph_max_num_seqs
        max_num_pages = (int(config.max_model_len) + self.page_size - 1) // self.page_size
        chunk_size = self.decode_graph_chunk_size
        max_num_tokens = max_num_seqs * chunk_size
        device = self.cuda_graph_device

        common_vars = self._create_common_graph_buffers(
            token_capacity=max_num_tokens,
            req_capacity=max_num_seqs,
            page_capacity=max_num_pages,
        )
        extra_vars = self._decode_graph_extra_vars(max_num_tokens, device)
        decode_buffers = MultiBlockGraphBuffers(common=common_vars, extra=extra_vars)

        for i in range(max_num_seqs + 1):
            common_vars["cu_seqlens_q"][i] = i * chunk_size
            common_vars["cu_seqlens_k"][i] = i * int(config.max_model_len)

        decode_bucket_tokens = [num_seqs * chunk_size for num_seqs in self._graph_seq_batch_sizes(max_num_seqs)]
        next_state = MultiBlockCudaGraphState(
            decode_bucket_tokens=decode_bucket_tokens,
            decode_graphs={},
            decode_buffers=decode_buffers,
        )

        self.graph_pool = None
        with warming_up_context():
            try:
                for num_tokens in tqdm(reversed(decode_bucket_tokens), desc="Capturing CUDA graphs"):
                    num_seqs = num_tokens // chunk_size
                    self.set_attn_metadata(
                        is_prefill=False,
                        slot_mapping=common_vars["slot_mapping"][:num_tokens],
                        context_lens=common_vars["context_lens"][:num_seqs],
                        cu_seqlens_q=common_vars["cu_seqlens_q"][: num_seqs + 1],
                        cu_seqlens_k=common_vars["cu_seqlens_k"][: num_seqs + 1],
                        max_seqlen_q=chunk_size,
                        max_seqlen_k=int(config.max_model_len),
                        page_size=config.kv_cache_page_size,
                        page_tables=common_vars["page_tables"][:num_seqs],
                        block_size=self.block_size,
                        kv_cache_layout=config.kv_cache_layout,
                    )
                    attn_metadata: AttnMetaDataBase = self.fetch_attn_metadata()
                    attn_metadata.init_multi_block(
                        valid_slices=common_vars["valid_slices"][:num_seqs],
                        buffer_size=config.buffer_size,
                        is_prefix_full=self.is_prefix_full,
                        status_table=common_vars["status_table"][:num_seqs],
                        prefix_lens=common_vars["prefix_lens"][:num_seqs],
                        padded_prefix_lens=common_vars["padded_prefix_lens"][:num_seqs],
                    )
                    self._init_decode_graph_extra_metadata(attn_metadata, extra_vars, num_tokens)

                    graph = self._capture_model_forward_graph(
                        common_vars["input_ids"],
                        common_vars["positions"],
                        common_vars["outputs"],
                        num_tokens,
                    )
                    if self.graph_pool is None:
                        self.graph_pool = graph.pool()
                    next_state.decode_graphs[num_tokens] = graph
                    torch.cuda.synchronize()
                    self.reset_attn_metadata()

                self.cuda_graph_state = next_state
            finally:
                self.reset_attn_metadata()
