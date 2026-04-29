from __future__ import annotations

import torch
from tqdm import tqdm

from diffulex.attention.metadata import AttnMetaDataBase, reset_warming_up, set_warming_up
from diffulex.profiling import record_function
from diffulex.strategy_template.token_merging_multi_block.attention.metadata import (
    TokenMergingMultiBlockAttnMetaDataTemplate,
)
from diffulex.strategy_template.multi_block.engine.model_runner import MultiBlockModelRunnerTemplate
from diffulex.strategy_template.token_merging_multi_block.engine.request import TokenMergingMultiBlockReqTemplate


class TokenMergingMultiBlockModelRunnerTemplate(MultiBlockModelRunnerTemplate):
    token_merge_renormalize = True

    def _ensure_runtime_token_merge_buffers(
        self,
        *,
        token_capacity: int,
        top_k_capacity: int,
    ) -> dict[str, torch.Tensor]:
        device = self._cuda_graph_device()
        token_capacity = max(1, int(token_capacity), int(self.config.max_num_batched_tokens))
        top_k_capacity = max(1, int(top_k_capacity), int(self.config.token_merge_top_k))
        buffers = getattr(self, "_runtime_token_merge_buffers", None)
        if buffers is not None:
            if (
                int(buffers["merge_mask"].numel()) >= token_capacity
                and int(buffers["topk_ids"].size(1)) >= top_k_capacity
            ):
                return buffers

        buffers = {
            "merge_mask": torch.empty(token_capacity, dtype=torch.bool, device=device),
            "topk_ids": torch.empty(token_capacity, top_k_capacity, dtype=torch.int64, device=device),
            "topk_probs": torch.empty(token_capacity, top_k_capacity, dtype=torch.float32, device=device),
            "residual_probs": torch.empty(token_capacity, 1, dtype=torch.float32, device=device),
        }
        self._runtime_token_merge_buffers = buffers
        return buffers

    def prepare_chunked_prefill_token_merging_multi_block(
        self: TokenMergingMultiBlockModelRunnerTemplate,
        reqs: list[TokenMergingMultiBlockReqTemplate],
    ):
        input_ids, positions = self.prepare_chunked_prefill_multi_block(reqs)
        self._init_token_merge_metadata(reqs, positions)
        return input_ids, positions

    def _init_token_merge_metadata(
        self: TokenMergingMultiBlockModelRunnerTemplate,
        reqs: list[TokenMergingMultiBlockReqTemplate],
        positions: torch.Tensor,
    ) -> None:
        attn_metadata: TokenMergingMultiBlockAttnMetaDataTemplate = self.fetch_attn_metadata()

        num_tokens = int(positions.numel())
        if num_tokens == 0:
            attn_metadata.init_token_merging(mask_token_id=self.config.mask_token_id)
            return

        descriptors = []
        max_top_k = 1
        has_merge = False
        for req in reqs:
            for position in req.running_position_ids:
                descriptor = req.token_merge_descriptor_for_position(position)
                descriptors.append(descriptor)
                if descriptor is not None:
                    has_merge = True
                    max_top_k = max(max_top_k, len(descriptor.topk_ids))

        if len(descriptors) != num_tokens:
            raise RuntimeError(
                "Token-merge descriptor alignment failed: "
                f"descriptors={len(descriptors)}, num_tokens={num_tokens}"
            )

        if not has_merge:
            attn_metadata.init_token_merging(mask_token_id=self.config.mask_token_id)
            return

        buffers = self._ensure_runtime_token_merge_buffers(token_capacity=num_tokens, top_k_capacity=max_top_k)
        merge_mask = buffers["merge_mask"][:num_tokens]
        topk_ids = buffers["topk_ids"][:num_tokens, :max_top_k]
        topk_probs = buffers["topk_probs"][:num_tokens, :max_top_k]
        residual_probs = buffers["residual_probs"][:num_tokens]

        mask_token_id = int(self.config.mask_token_id)
        merge_mask_rows = [False] * num_tokens
        topk_id_rows = [[mask_token_id] * max_top_k for _ in range(num_tokens)]
        topk_prob_rows = [[0.0] * max_top_k for _ in range(num_tokens)]
        residual_prob_rows = [[0.0] for _ in range(num_tokens)]

        for idx, descriptor in enumerate(descriptors):
            if descriptor is None:
                continue
            k = len(descriptor.topk_ids)
            merge_mask_rows[idx] = True
            topk_id_rows[idx][:k] = descriptor.topk_ids
            topk_prob_rows[idx][:k] = descriptor.topk_probs
            residual_prob_rows[idx][0] = float(descriptor.residual_prob)

        merge_mask.copy_(self._cpu_tensor(merge_mask_rows, torch.bool), non_blocking=True)
        topk_ids.copy_(self._cpu_tensor(topk_id_rows, torch.int64), non_blocking=True)
        topk_probs.copy_(self._cpu_tensor(topk_prob_rows, torch.float32), non_blocking=True)
        residual_probs.copy_(self._cpu_tensor(residual_prob_rows, torch.float32), non_blocking=True)

        attn_metadata.init_token_merging(
            merge_mask=merge_mask,
            topk_ids=topk_ids,
            topk_probs=topk_probs,
            residual_probs=residual_probs,
            mask_token_id=self.config.mask_token_id,
            renormalize=bool(self.config.token_merge_renormalize),
            mode=self.config.token_merge_mode,
            weight=float(self.config.token_merge_weight),
        )

    def _graph_token_merge_top_k(self) -> int:
        return max(1, int(self.config.token_merge_top_k))

    def _init_graph_capture_token_merge_metadata(
        self: TokenMergingMultiBlockModelRunnerTemplate,
        attn_metadata: TokenMergingMultiBlockAttnMetaDataTemplate,
        graph_vars: dict[str, torch.Tensor],
        num_tokens: int,
    ) -> None:
        merge_mask = graph_vars["token_merge_mask"][:num_tokens]
        topk_ids = graph_vars["token_merge_topk_ids"][:num_tokens]
        topk_probs = graph_vars["token_merge_topk_probs"][:num_tokens]
        residual_probs = graph_vars["token_merge_residual_probs"][:num_tokens]

        merge_mask.zero_()
        topk_ids.fill_(int(self.config.mask_token_id))
        topk_probs.zero_()
        residual_probs.zero_()
        if num_tokens > 0:
            # Capture the token-merge branch once; replay-time merge_mask controls no-op behavior.
            merge_mask[0] = True
            residual_probs[0, 0] = 1.0

        attn_metadata.init_token_merging(
            merge_mask=merge_mask,
            topk_ids=topk_ids,
            topk_probs=topk_probs,
            residual_probs=residual_probs,
            mask_token_id=self.config.mask_token_id,
            renormalize=bool(self.config.token_merge_renormalize),
            mode=self.config.token_merge_mode,
            weight=float(self.config.token_merge_weight),
        )

    def _init_prefill_graph_extra_metadata(
        self: TokenMergingMultiBlockModelRunnerTemplate,
        attn_metadata: TokenMergingMultiBlockAttnMetaDataTemplate,
        graph_vars: dict[str, torch.Tensor],
        bucket_len: int,
    ) -> None:
        capture_graph_vars = {
            "token_merge_mask": graph_vars["token_merge_mask"],
            "token_merge_topk_ids": graph_vars["token_merge_topk_ids"],
            "token_merge_topk_probs": graph_vars["token_merge_topk_probs"],
            "token_merge_residual_probs": graph_vars["token_merge_residual_probs"],
        }
        self._init_graph_capture_token_merge_metadata(attn_metadata, capture_graph_vars, bucket_len)

    def _prefill_graph_extra_vars(
        self: TokenMergingMultiBlockModelRunnerTemplate,
        bucket_len: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        token_merge_top_k = self._graph_token_merge_top_k()
        return {
            "token_merge_mask": torch.zeros(bucket_len, dtype=torch.bool, device=device),
            "token_merge_topk_ids": torch.full(
                (bucket_len, token_merge_top_k),
                int(self.config.mask_token_id),
                dtype=torch.int64,
                device=device,
            ),
            "token_merge_topk_probs": torch.zeros(
                (bucket_len, token_merge_top_k),
                dtype=torch.float32,
                device=device,
            ),
            "token_merge_residual_probs": torch.zeros((bucket_len, 1), dtype=torch.float32, device=device),
        }

    def _bind_graph_token_merge_metadata(
        self: TokenMergingMultiBlockModelRunnerTemplate,
        attn_metadata: TokenMergingMultiBlockAttnMetaDataTemplate,
        graph_vars: dict[str, torch.Tensor],
        num_tokens: int,
    ) -> None:
        graph_merge_mask = graph_vars["token_merge_mask"][:num_tokens]
        graph_topk_ids = graph_vars["token_merge_topk_ids"][:num_tokens]
        graph_topk_probs = graph_vars["token_merge_topk_probs"][:num_tokens]
        graph_residual_probs = graph_vars["token_merge_residual_probs"][:num_tokens]

        graph_merge_mask.zero_()
        graph_topk_ids.fill_(int(self.config.mask_token_id))
        graph_topk_probs.zero_()
        graph_residual_probs.zero_()

        src_merge_mask = attn_metadata.token_merge_mask
        src_topk_ids = attn_metadata.token_merge_topk_ids
        src_topk_probs = attn_metadata.token_merge_topk_probs
        src_residual_probs = attn_metadata.token_merge_residual_probs
        has_merge = (
            bool(attn_metadata.token_merge_enabled)
            and src_merge_mask is not None
            and src_topk_ids is not None
            and src_topk_probs is not None
            and src_residual_probs is not None
        )

        if has_merge:
            if int(src_merge_mask.numel()) != num_tokens:
                raise RuntimeError(
                    "Token-merge metadata length does not match CUDA graph replay size: "
                    f"merge_mask={src_merge_mask.numel()}, num_tokens={num_tokens}"
                )
            src_top_k = int(src_topk_ids.shape[1])
            graph_top_k = int(graph_topk_ids.shape[1])
            if src_top_k > graph_top_k:
                raise RuntimeError(
                    "Token-merge top-k exceeds CUDA graph capture capacity: "
                    f"src_top_k={src_top_k}, graph_top_k={graph_top_k}"
                )

            graph_merge_mask.copy_(src_merge_mask.to(device=graph_merge_mask.device, dtype=torch.bool))
            graph_topk_ids[:, :src_top_k].copy_(src_topk_ids.to(device=graph_topk_ids.device, dtype=torch.int64))
            graph_topk_probs[:, :src_top_k].copy_(
                src_topk_probs.to(device=graph_topk_probs.device, dtype=torch.float32)
            )
            graph_residual_probs.copy_(
                src_residual_probs.to(device=graph_residual_probs.device, dtype=torch.float32)
            )

        attn_metadata.init_token_merging(
            merge_mask=graph_merge_mask,
            topk_ids=graph_topk_ids,
            topk_probs=graph_topk_probs,
            residual_probs=graph_residual_probs,
            mask_token_id=self.config.mask_token_id,
            renormalize=bool(self.config.token_merge_renormalize),
            mode=self.config.token_merge_mode,
            weight=float(self.config.token_merge_weight),
        )

    def _bind_prefill_graph_extra_metadata(
        self: TokenMergingMultiBlockModelRunnerTemplate,
        attn_metadata: TokenMergingMultiBlockAttnMetaDataTemplate,
        graph_vars: dict[str, torch.Tensor],
        bucket_len: int,
        runtime_num_tokens: int,
        source_attn_metadata: TokenMergingMultiBlockAttnMetaDataTemplate | None = None,
    ) -> None:
        graph_merge_mask = graph_vars["token_merge_mask"][:bucket_len]
        graph_topk_ids = graph_vars["token_merge_topk_ids"][:bucket_len]
        graph_topk_probs = graph_vars["token_merge_topk_probs"][:bucket_len]
        graph_residual_probs = graph_vars["token_merge_residual_probs"][:bucket_len]

        graph_merge_mask.zero_()
        graph_topk_ids.fill_(int(self.config.mask_token_id))
        graph_topk_probs.zero_()
        graph_residual_probs.zero_()

        src_metadata = source_attn_metadata if source_attn_metadata is not None else attn_metadata
        src_merge_mask = src_metadata.token_merge_mask
        src_topk_ids = src_metadata.token_merge_topk_ids
        src_topk_probs = src_metadata.token_merge_topk_probs
        src_residual_probs = src_metadata.token_merge_residual_probs
        has_merge = (
            bool(src_metadata.token_merge_enabled)
            and src_merge_mask is not None
            and src_topk_ids is not None
            and src_topk_probs is not None
            and src_residual_probs is not None
        )
        if has_merge:
            if int(src_merge_mask.numel()) != runtime_num_tokens:
                raise RuntimeError(
                    "Token-merge metadata length does not match prefill graph runtime size: "
                    f"merge_mask={src_merge_mask.numel()}, runtime_num_tokens={runtime_num_tokens}"
                )
            src_top_k = int(src_topk_ids.shape[1])
            graph_top_k = int(graph_topk_ids.shape[1])
            if src_top_k > graph_top_k:
                raise RuntimeError(
                    "Token-merge top-k exceeds CUDA graph capture capacity: "
                    f"src_top_k={src_top_k}, graph_top_k={graph_top_k}"
                )
            graph_merge_mask[:runtime_num_tokens].copy_(
                src_merge_mask.to(device=graph_merge_mask.device, dtype=torch.bool)
            )
            graph_topk_ids[:runtime_num_tokens, :src_top_k].copy_(
                src_topk_ids.to(device=graph_topk_ids.device, dtype=torch.int64)
            )
            graph_topk_probs[:runtime_num_tokens, :src_top_k].copy_(
                src_topk_probs.to(device=graph_topk_probs.device, dtype=torch.float32)
            )
            graph_residual_probs[:runtime_num_tokens].copy_(
                src_residual_probs.to(device=graph_residual_probs.device, dtype=torch.float32)
            )

        attn_metadata.init_token_merging(
            merge_mask=graph_merge_mask,
            topk_ids=graph_topk_ids,
            topk_probs=graph_topk_probs,
            residual_probs=graph_residual_probs,
            mask_token_id=self.config.mask_token_id,
            renormalize=bool(self.config.token_merge_renormalize),
            mode=self.config.token_merge_mode,
            weight=float(self.config.token_merge_weight),
        )

    def run_multi_block(
        self: TokenMergingMultiBlockModelRunnerTemplate,
        reqs: list[TokenMergingMultiBlockReqTemplate],
    ) -> list[int]:
        return self._run_token_merging_multi_block_subgroup(reqs)

    @torch.inference_mode()
    def run_model_multi_block(
        self: TokenMergingMultiBlockModelRunnerTemplate,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ):
        with record_function("diffulex.token_merging_multi_block.model_forward"):
            attn_metadata: AttnMetaDataBase = self.fetch_attn_metadata()
            full_runner = self._full_static_runner()

            if (attn_metadata.status_table == 0).any():
                if full_runner.can_run_prefill(attn_metadata, int(input_ids.size(0))):
                    with record_function("diffulex.token_merging_multi_block.full_static_prefill"):
                        return full_runner.run_prefill(input_ids, positions, attn_metadata)
                with record_function("diffulex.token_merging_multi_block.eager_prefill"):
                    return self.model.compute_logits(self.model(input_ids, positions))

            if not full_runner.can_run_decode(input_ids):
                with record_function("diffulex.token_merging_multi_block.eager_decode"):
                    return self.model.compute_logits(self.model(input_ids, positions))

            with record_function("diffulex.token_merging_multi_block.full_static_decode"):
                return full_runner.run_decode(input_ids, positions, attn_metadata)

    def _bind_decode_graph_extra_metadata(
        self: TokenMergingMultiBlockModelRunnerTemplate,
        attn_metadata: TokenMergingMultiBlockAttnMetaDataTemplate,
        graph_vars: dict[str, torch.Tensor],
        num_tokens: int,
    ) -> None:
        self._bind_graph_token_merge_metadata(attn_metadata, graph_vars, num_tokens)

    @torch.inference_mode()
    def capture_cudagraph_token_merging_multi_block(self: TokenMergingMultiBlockModelRunnerTemplate):
        set_warming_up(True)
        config = self.config
        hf_config = config.hf_config
        max_num_seqs = min(self.config.max_num_reqs, 512)
        max_num_pages = (config.max_model_len + self.page_size - 1) // self.page_size
        block_size = config.block_size
        buffer_size = config.buffer_size
        chunk_size = block_size * buffer_size
        max_num_tokens = max_num_seqs * chunk_size
        token_merge_top_k = self._graph_token_merge_top_k()
        device = self._cuda_graph_device()

        input_ids = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        slot_mapping = torch.full((max_num_tokens,), -1, dtype=torch.int32, device=device)
        context_lens = torch.zeros(max_num_seqs, dtype=torch.int32, device=device)
        page_tables = torch.zeros(max_num_seqs, max_num_pages, dtype=torch.int32, device=device)
        valid_slices = torch.zeros(max_num_seqs, dtype=torch.int32, device=device)
        status_table = torch.zeros(max_num_seqs, dtype=torch.int32, device=device)
        prefix_lens = torch.zeros(max_num_seqs, dtype=torch.int32, device=device)
        padded_prefix_lens = torch.zeros(max_num_seqs, dtype=torch.int32, device=device)
        token_merge_mask = torch.zeros(max_num_tokens, dtype=torch.bool, device=device)
        token_merge_topk_ids = torch.full(
            (max_num_tokens, token_merge_top_k),
            int(self.config.mask_token_id),
            dtype=torch.int64,
            device=device,
        )
        token_merge_topk_probs = torch.zeros(
            (max_num_tokens, token_merge_top_k),
            dtype=torch.float32,
            device=device,
        )
        token_merge_residual_probs = torch.zeros((max_num_tokens, 1), dtype=torch.float32, device=device)
        outputs = torch.zeros(max_num_tokens, hf_config.hidden_size, dtype=self._model_hidden_dtype(), device=device)

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
            attn_metadata: TokenMergingMultiBlockAttnMetaDataTemplate = self.fetch_attn_metadata()
            attn_metadata.init_multi_block(
                valid_slices=valid_slices[:num_seqs],
                buffer_size=buffer_size,
                is_prefix_full=self.is_prefix_full,
                status_table=status_table[:num_seqs],
                prefix_lens=prefix_lens[:num_seqs],
                padded_prefix_lens=padded_prefix_lens[:num_seqs],
            )
            capture_graph_vars = {
                "token_merge_mask": token_merge_mask,
                "token_merge_topk_ids": token_merge_topk_ids,
                "token_merge_topk_probs": token_merge_topk_probs,
                "token_merge_residual_probs": token_merge_residual_probs,
            }
            self._init_graph_capture_token_merge_metadata(attn_metadata, capture_graph_vars, num_tokens)

            graph = self._capture_model_forward_graph(input_ids, positions, outputs, num_tokens, allow_compile=True)
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
            token_merge_mask=token_merge_mask,
            token_merge_topk_ids=token_merge_topk_ids,
            token_merge_topk_probs=token_merge_topk_probs,
            token_merge_residual_probs=token_merge_residual_probs,
            outputs=outputs,
        )
        reset_warming_up()

    def _run_token_merging_multi_block_subgroup(
        self: TokenMergingMultiBlockModelRunnerTemplate,
        reqs: list[TokenMergingMultiBlockReqTemplate],
    ):
        with record_function("diffulex.token_merging_multi_block.filter_local_reqs"):
            local_reqs = self.filter_local_reqs(reqs)
        if not local_reqs:
            with record_function("diffulex.token_merging_multi_block.gather_dp_sample_output"):
                return self.gather_dp_sample_output(None)

        with record_function("diffulex.token_merging_multi_block.prepare_chunked_prefill"):
            input_ids, positions = self.prepare_chunked_prefill_token_merging_multi_block(local_reqs)
        with record_function("diffulex.token_merging_multi_block.prepare_sample"):
            temperatures = self.prepare_sample(local_reqs) if self.is_model_parallel_root else None
        logits = self.run_model_multi_block(input_ids, positions)
        with record_function("diffulex.token_merging_multi_block.sampler"):
            sample_output = self.sampler(local_reqs, logits, temperatures) if self.is_model_parallel_root else None
        with record_function("diffulex.token_merging_multi_block.reset_attn_metadata"):
            self.reset_attn_metadata()
        with record_function("diffulex.token_merging_multi_block.gather_dp_sample_output"):
            return self.gather_dp_sample_output(sample_output)
