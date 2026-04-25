from __future__ import annotations

import torch

from diffulex.attention.metadata import AttnMetaDataBase


class FullStaticRunner:
    """CUDA graph runner with stable replay buffers owned by the model runner.

    The model runner still owns capture and metadata construction because those
    are strategy-specific. This class is the execution boundary: can-run checks,
    prefill/decode graph replay, and logits projection after replay.
    """

    def __init__(self, owner) -> None:
        self.owner = owner

    def enabled(self) -> bool:
        return bool(getattr(self.owner.config, "enable_full_static_runner", True))

    def can_run_prefill(self, attn_metadata: AttnMetaDataBase, num_tokens: int) -> bool:
        return self.enabled() and self.owner._can_use_prefill_graph(attn_metadata, num_tokens)

    def can_run_decode(self, input_ids: torch.Tensor) -> bool:
        if not self.enabled() or self.owner.enforce_eager:
            return False
        max_graph_tokens = 512 * (self.owner.config.buffer_size * self.owner.config.block_size)
        return int(input_ids.size(0)) <= max_graph_tokens

    @torch.inference_mode()
    def run_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttnMetaDataBase,
    ):
        return self.owner._run_prefill_cudagraph(input_ids, positions, attn_metadata)

    @torch.inference_mode()
    def run_decode(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttnMetaDataBase,
    ):
        num_tokens = int(input_ids.size(0))
        captured_num_tokens = next(x for x in self.owner.graph_bs if x >= num_tokens)
        captured_num_seqs = captured_num_tokens // (
            self.owner.config.block_size * self.owner.config.buffer_size
        )
        graph = self.owner.graphs[captured_num_tokens]
        graph_vars = self.owner.graph_vars
        graph_capacity = int(graph_vars["context_lens"].size(0))
        if captured_num_seqs > graph_capacity:
            raise RuntimeError(
                "Captured CUDA graph batch size exceeds allocated graph buffer capacity: "
                f"captured_num_seqs={captured_num_seqs}, graph_capacity={graph_capacity}, "
                f"captured_num_tokens={captured_num_tokens}, num_tokens={num_tokens}, "
                f"max_num_reqs={self.owner.config.max_num_reqs}, "
                f"block_size={self.owner.config.block_size}, "
                f"buffer_size={self.owner.config.buffer_size}"
            )

        num_reqs = attn_metadata.num_reqs
        self.owner._copy_common_graph_inputs(
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

        attn_metadata.slot_mapping = graph_vars["slot_mapping"]
        attn_metadata.context_lens = graph_vars["context_lens"]
        attn_metadata.cu_seqlens_q = graph_vars["cu_seqlens_q"]
        attn_metadata.cu_seqlens_k = graph_vars["cu_seqlens_k"]
        attn_metadata.valid_slices = graph_vars["valid_slices"]
        attn_metadata.status_table = graph_vars["status_table"]
        attn_metadata.prefix_lens = graph_vars["prefix_lens"]
        attn_metadata.padded_prefix_lens = graph_vars["padded_prefix_lens"]
        attn_metadata.page_tables = graph_vars["page_tables"]
        self.owner._bind_decode_graph_extra_metadata(attn_metadata, graph_vars, num_tokens)

        graph.replay()
        if bool(getattr(self.owner, "graph_outputs_are_logits", False)):
            return graph_vars["outputs"][:num_tokens]
        return self.owner.model.compute_logits(graph_vars["outputs"][:num_tokens])
