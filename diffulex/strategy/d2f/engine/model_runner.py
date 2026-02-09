from __future__ import annotations

import time

from multiprocessing.synchronize import Event

import torch

from diffulex.config import Config
from diffulex.engine.sequence import SequenceBase
from diffulex.strategy.d2f.engine.sequence import D2FSequence
from diffulex.attention.metadata import set_fetch_fn_for_attn_metadata, set_warming_up, reset_warming_up
from diffulex.engine.model_runner import AutoModelRunner, ModelRunnerBase
from diffulex.strategy.d2f.attention.metadata import fetch_d2f_attn_metadata, set_d2f_attn_metadata, reset_d2f_attn_metadata


@AutoModelRunner.register("d2f", is_default=True)
class D2FModelRunner(ModelRunnerBase):
    """Reference implementation of D2F decoding strategy."""
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        set_fetch_fn_for_attn_metadata(fetch_d2f_attn_metadata)
        
        self.diffusion_block_size = config.diffusion_block_size
        self.mask_token_id = config.mask_token_id
        
        super().__init__(config, rank, event)

    def _get_decode_mode(self) -> str:
        """
        统一选择 decode_mode 的逻辑：
        1. 如果 config.decode_mode 已设置，优先使用 config 的值
        2. 否则，如果 kv_cache_dtype 是 FP8，自动切换到 "static"
        3. 否则，默认使用 "varlen"
        """
        if self.config.decode_mode is not None:
            return self.config.decode_mode
        
        # Auto-select based on kv_cache_dtype
        decode_mode = "varlen"
        try:
            from diffulex.utils.kv_cache_dtype import parse_kv_cache_dtype
            if parse_kv_cache_dtype(getattr(self.config, "kv_cache_dtype", "bf16")).is_fp8:
                decode_mode = "static"
        except Exception:
            decode_mode = "varlen"
        
        return decode_mode

    def prepare_prefill(self, seqs: list[D2FSequence]):
        input_ids: list[int] = []
        positions: list[int] = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping: list[int] = []
        block_tables = None
        context_lens: list[int] = []
        seq_lens: list[int] = []

        for seq in seqs:
            seq.next_diffusion_step(is_prefill=True)

            total_seqlen = len(seq)
            input_ids.extend(seq[seq.cached_num_tokens:])
            positions.extend(range(seq.cached_num_tokens, total_seqlen))
            seq_lens.append(total_seqlen)
            context_lens.append(0)
            assert len(input_ids) == len(positions), (
                "prepare_prefill(diffusion): len(input_ids) {len_ids} != len(positions) {len_pos}".format(
                    len_ids=len(input_ids),
                    len_pos=len(positions),
                )
            )

            seqlen_q = total_seqlen - seq.cached_num_tokens
            seqlen_k = total_seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:
                continue
            for i in range(0, seq.num_prompt_blocks):
                if seq.block_cache_missed[i]:
                    start = seq.block_table[i] * self.block_size
                    if i != seq.num_prompt_blocks - 1:
                        end = start + self.block_size
                    else:
                        end = start + seq.last_block_prompt_num_tokens
                    slot_mapping.extend(range(start, end))
                else:
                    slot_mapping.extend([-1] * self.block_size)
            slot_mapping.extend([-1] * seq.diffusion_block_size)

        block_tables = self.prepare_block_tables(seqs)

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_tensor = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        seq_lens_ts = torch.tensor(seq_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens_tensor = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q_tensor = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k_tensor = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        assert cu_seqlens_q_tensor[-1].item() == input_ids_tensor.numel(), (
            "prepare_prefill(diffusion): cu_seqlens_q[-1]={cq} != num_tokens={nt}".format(
                cq=cu_seqlens_q_tensor[-1].item(),
                nt=input_ids_tensor.numel(),
            )
        )
        assert cu_seqlens_k_tensor[-1].item() == sum(seq_lens), (
            "prepare_prefill(diffusion): cu_seqlens_k[-1]={ck} != sum(seq_lens)={sl}".format(
                ck=cu_seqlens_k_tensor[-1].item(),
                sl=sum(seq_lens),
            )
        )

        decode_mode = self._get_decode_mode()
        set_d2f_attn_metadata(
            True,
            cu_seqlens_q=cu_seqlens_q_tensor,
            cu_seqlens_k=cu_seqlens_k_tensor,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping_tensor,
            context_lens=context_lens_tensor,
            block_tables=block_tables,
            seqs=seqs,
            kv_cache_layout=self.config.kv_cache_layout,
            seq_lens=seq_lens,
            seq_lens_ts=seq_lens_ts,
            diffusion_block_size=self.diffusion_block_size,
            decode_mode=decode_mode,
            attn_type="full_attention",
        )
        return input_ids_tensor, positions_tensor

    def prepare_decode(self, seqs: list[D2FSequence]):
        input_ids: list[int] = []
        positions: list[int] = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        slot_mapping: list[int] = []
        context_lens: list[int] = []
        seq_lens: list[int] = []
        seq_id_to_queue_id: dict[int, int] = {}
        need_kv_cache_store = False
        max_seqlen_q = 0
        max_seqlen_k = 0

        for seq_idx_in_queue, seq in enumerate(seqs):
            seq_id = seq.seq_id
            seq_id_to_queue_id[seq_id] = seq_idx_in_queue
            seq.next_diffusion_step()
            cur_input_ids, cur_positions, cur_context_len = seq.diffusion_decoding_inputs()

            seq_lens.append(len(cur_input_ids))
            input_ids.extend(cur_input_ids)
            positions.extend(cur_positions)
            context_lens.append(cur_context_len)

            total_seqlen = len(seq)
            seqlen_q = total_seqlen - seq.cached_num_tokens
            seqlen_k = total_seqlen
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            mem_block_to_diffusion_blocks_map = seq.mem_block_to_diffusion_blocks_map
            context_len = context_lens[seq_id_to_queue_id[seq_id]]
            for mem_block_idx in range(0, seq.num_blocks):
                start_idx = mem_block_idx * seq.block_size
                end_idx = start_idx + seq.block_size
                cur_map = mem_block_to_diffusion_blocks_map[mem_block_idx]
                is_last_block = False
                meet_active_block = False
                while start_idx < end_idx and not is_last_block and not meet_active_block:
                    local_start_idx = lambda: start_idx % seq.block_size
                    diffusion_block = seq.diffusion_blocks[cur_map[local_start_idx()]]
                    if diffusion_block.block_id == 0 and diffusion_block.cursor != start_idx:
                        diffusion_block.cursor = start_idx
                    if cur_map[local_start_idx()] == seq.num_diffusion_blocks - 1:
                        is_last_block = True

                    def get_step(diff_blk, begin_idx):
                        remaining = diff_blk.remaining_length(begin_idx)
                        if remaining + local_start_idx() <= seq.block_size:
                            return remaining
                        return seq.block_size - local_start_idx()

                    if diffusion_block.is_in_cache:
                        step = get_step(diffusion_block, start_idx)
                        diffusion_block.cursor += step
                        start_idx += step
                    elif diffusion_block.is_to_cache:
                        step = get_step(diffusion_block, start_idx)
                        diffusion_block.cursor += step
                        cur_diffusion_block_start = 0
                        cur_diffusion_block_end = step
                        start_idx += step
                        # IMPORTANT:
                        # We must have a KV-cache block allocated for this mem_block_idx.
                        # If not, this is almost always due to insufficient KV cache blocks
                        # (e.g. higher model/weight memory footprint leaves too few blocks).
                        if mem_block_idx >= len(seq.block_table):
                            raise RuntimeError(
                                "KV cache block allocation is insufficient during decode: "
                                f"mem_block_idx={mem_block_idx} requires block_table length >= {mem_block_idx + 1}, "
                                f"but got len(block_table)={len(seq.block_table)} (seq.num_blocks={seq.num_blocks}). "
                                "This usually means GPU memory utilization is too low to allocate enough KV cache "
                                f"blocks for this run (num_kvcache_blocks={getattr(self.config, 'num_kvcache_blocks', None)}, "
                                f"gpu_memory_utilization={getattr(self.config, 'gpu_memory_utilization', None)}). "
                                "Try increasing gpu_memory_utilization, reducing max_model_len/max_tokens/max_num_seqs, "
                                "or using a lower-memory weight quantization (e.g. int4)."
                            )
                        mem_block_start = (
                            seq.block_table[mem_block_idx] * self.block_size
                            + context_len % seq.block_size
                        )
                        context_len += step
                        slot_mapping.extend(
                            range(
                                mem_block_start + cur_diffusion_block_start,
                                mem_block_start + cur_diffusion_block_end,
                            )
                        )
                        need_kv_cache_store = True
                    elif diffusion_block.is_active:
                        meet_active_block = True

                if meet_active_block:
                    active = seq.active_blocks
                    first_active_idx = next((i for i, v in enumerate(active) if v), None)
                    if first_active_idx is not None:
                        num_blocks_to_pad = len(active) - first_active_idx
                        slot_mapping.extend([-1] * (num_blocks_to_pad * seq.diffusion_block_size))
                    break
            assert len(input_ids) == len(positions), (
                "Input IDs length {len_ids} does not match positions length {len_pos}".format(
                    len_ids=len(input_ids),
                    len_pos=len(positions),
                )
            )
            assert len(input_ids) == len(slot_mapping), (
                "Input IDs length {len_ids} does not match slot mapping length {len_slot}".format(
                    len_ids=len(input_ids),
                    len_slot=len(slot_mapping),
                )
            )

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_tensor = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        seq_lens_ts = torch.tensor(seq_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q_tensor = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k_tensor = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens_tensor = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        # NOTE:
        # - d2f decode supports "varlen" and "static" modes (see config.decode_mode).
        # - For FP8 KV, the (varlen/distinct-layout) path uses `load_kvcache` which is expected to
        #   handle FP8 dequantization / scale application inside the fused operator (no Python-level dequant).
        # - Performance can still differ between modes/kernels; when FP8 KV is enabled, prefer the
        #   best-supported kernel path on your stack (often "static"/unified-layout) and validate with profiling.
        # - Allow manual override via config.decode_mode if specified.
        decode_mode = self._get_decode_mode()
        set_d2f_attn_metadata(
            False,
            slot_mapping=slot_mapping_tensor,
            context_lens=context_lens_tensor,
            cu_seqlens_q=cu_seqlens_q_tensor,
            cu_seqlens_k=cu_seqlens_k_tensor,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            block_tables=block_tables,
            seqs=seqs,
            seq_lens=seq_lens,
            seq_lens_ts=seq_lens_ts,
            kv_cache_layout=self.config.kv_cache_layout,
            need_kv_cache_store=need_kv_cache_store,
            diffusion_block_size=self.diffusion_block_size,
            decode_mode=decode_mode,
            attn_type="full_attention",
        )
        return input_ids_tensor, positions_tensor

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        num_tokens = input_ids.size(0)
        context = fetch_d2f_attn_metadata()
        candidates = [x for x in self.graph_bs if x >= num_tokens]
        if not candidates:
            # Safety: fall back if capture didn't include a large-enough bucket.
            return self.model.compute_logits(self.model(input_ids, positions))
        bucket_tokens = candidates[0]
        graph = self.graphs[bucket_tokens]
        graph_vars = self.graph_vars
        # Safety: fall back if runtime batch exceeds captured metadata capacity.
        num_seqs = int(context.context_lens.numel())
        max_num_seqs_for_graph = int(graph_vars["context_lens"].numel())
        if num_seqs > max_num_seqs_for_graph:
            return self.model.compute_logits(self.model(input_ids, positions))

        # Reset buffers to safe defaults (avoid "0" being interpreted as a valid index).
        graph_vars["input_ids"].zero_()
        graph_vars["positions"].zero_()
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["context_lens"].zero_()
        graph_vars["block_tables"].fill_(-1)
        graph_vars["input_ids"][:num_tokens] = input_ids
        graph_vars["positions"][:num_tokens] = positions
        graph_vars["slot_mapping"][:num_tokens] = context.slot_mapping
        graph_vars["context_lens"][:num_seqs] = context.context_lens
        # cu_seqlens are required by unified paged-attn decode kernels.
        if getattr(context, "cu_seqlens_q", None) is not None:
            # Pad to captured length so "extra" sequences become 0-length.
            graph_vars["cu_seqlens_q"].fill_(int(num_tokens))
            graph_vars["cu_seqlens_q"][: num_seqs + 1] = context.cu_seqlens_q
        if getattr(context, "cu_seqlens_k", None) is not None:
            last_k = int(context.cu_seqlens_k[num_seqs].item())
            graph_vars["cu_seqlens_k"].fill_(last_k)
            graph_vars["cu_seqlens_k"][: num_seqs + 1] = context.cu_seqlens_k

        bt_cols = min(int(graph_vars["block_tables"].size(1)), int(context.block_tables.size(1)))
        graph_vars["block_tables"][:num_seqs, :bt_cols] = context.block_tables[:, :bt_cols]
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:num_tokens])

    def run(self, seqs: list[SequenceBase], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        sample_output = self.sampler(logits, temperatures) if self.rank == 0 else None
        reset_d2f_attn_metadata()
        return sample_output

    @torch.inference_mode()
    def capture_cudagraph(self):
        # Static-mode CUDA graph capture for D2F decode.
        #
        # NOTE:
        # - This matches `run_model()`'s replay protocol: we only overwrite
        #   input_ids/positions/slot_mapping/context_lens/block_tables per step.
        # - Varlen mode is intentionally not supported here (assume static flow).
        from tqdm import tqdm

        # Enable per-layer forward-plan dispatch to stabilize capture and minimize
        # Python branching inside the captured region.
        try:
            from diffulex.layer.linear import LinearBase
            for m in self.model.modules():
                if isinstance(m, LinearBase):
                    m.enable_forward_plan(True)
        except Exception:
            pass

        set_warming_up(True)
        config = self.config
        hf_config = config.hf_config
        diffusion_block_size = int(self.diffusion_block_size)
        max_num_seqs = int(self.config.max_num_seqs)
        # Graph path is only used when num_tokens <= 512.
        #
        # IMPORTANT:
        # In D2F decode, `num_tokens` (sum of per-seq seqlen_q) is NOT guaranteed to equal
        # `num_seqs * diffusion_block_size`. A single seq can contribute multiple diffusion blocks,
        # so we must bucket by `num_tokens` directly and keep metadata tensors sized by
        # `max_num_seqs_for_graph` (padding unused seqs to 0-length via cu_seqlens).
        max_num_seqs_for_graph = max(1, min(max_num_seqs, 512))
        max_num_tokens = 512
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        # Allocate graph buffers on the same device/dtype as the model.
        try:
            p0 = next(self.model.parameters())
            graph_device = p0.device
            graph_dtype = p0.dtype
        except StopIteration:
            graph_device = torch.device("cuda")
            graph_dtype = torch.float16

        # Allocate max-size graph buffers.
        input_ids = torch.zeros(max_num_tokens, dtype=torch.int64, device=graph_device)
        positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=graph_device)
        slot_mapping = torch.full((max_num_tokens,), -1, dtype=torch.int32, device=graph_device)
        context_lens = torch.zeros(max_num_seqs_for_graph, dtype=torch.int32, device=graph_device)
        block_tables = torch.full((max_num_seqs_for_graph, max_num_blocks), -1, dtype=torch.int32, device=graph_device)
        outputs = torch.zeros(max_num_tokens, hf_config.hidden_size, dtype=graph_dtype, device=graph_device)
        cu_seqlens_q = torch.zeros(max_num_seqs_for_graph + 1, dtype=torch.int32, device=graph_device)
        cu_seqlens_k = torch.zeros(max_num_seqs_for_graph + 1, dtype=torch.int32, device=graph_device)

        # Capture bucketed graphs by total num_tokens.
        self.graph_bs = []
        # Keep buckets aligned to diffusion_block_size for stable kernel shapes.
        for t in range(diffusion_block_size, max_num_tokens + 1, diffusion_block_size):
            self.graph_bs.append(int(t))
        self.graphs = {}
        self.graph_pool = None

        for num_tokens in tqdm(reversed(self.graph_bs), desc="Capturing CUDA graphs"):
            num_seqs = int(max_num_seqs_for_graph)
            graph = torch.cuda.CUDAGraph()
            # Fill placeholder metadata with valid monotonic cu_seqlens to satisfy kernel assertions.
            # IMPORTANT: cu_seqlens_q must be non-decreasing and end at `num_tokens`
            # (it is used to index into Q/slot_mapping which are length `num_tokens`).
            # Use a simple placeholder: put all Q tokens into the first seq and make
            # the remaining seqs 0-length.
            cu_seqlens_q[: num_seqs + 1].fill_(int(num_tokens))
            cu_seqlens_q[0] = 0
            # Use a conservative max-seqlen for K to keep shapes stable; values are overwritten before replay.
            cu_seqlens_k[: num_seqs + 1] = (
                torch.arange(num_seqs + 1, dtype=torch.int32, device=graph_device) * int(config.max_model_len)
            )
            context_lens[:num_seqs].fill_(int(config.max_model_len))
            # Use a benign placeholder block table for the first seq.
            block_tables[:1].zero_()
            # For static decode, use placeholder metadata tensors; per-step values are copied
            # into `graph_vars` before replay.
            set_d2f_attn_metadata(
                False,
                slot_mapping=slot_mapping[:num_tokens],
                context_lens=context_lens[:num_seqs],
                cu_seqlens_q=cu_seqlens_q[: num_seqs + 1],
                cu_seqlens_k=cu_seqlens_k[: num_seqs + 1],
                max_seqlen_q=int(num_tokens),
                max_seqlen_k=int(config.max_model_len),
                block_tables=block_tables[:num_seqs],
                kv_cache_layout=self.config.kv_cache_layout,
                need_kv_cache_store=True,
                diffusion_block_size=self.diffusion_block_size,
                decode_mode="static",
                attn_type="full_attention",
            )
            outputs[:num_tokens] = self.model(input_ids[:num_tokens], positions[:num_tokens])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:num_tokens] = self.model(input_ids[:num_tokens], positions[:num_tokens])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[num_tokens] = graph
            torch.cuda.synchronize()
            reset_d2f_attn_metadata()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            block_tables=block_tables,
            outputs=outputs,
        )
        reset_warming_up()
