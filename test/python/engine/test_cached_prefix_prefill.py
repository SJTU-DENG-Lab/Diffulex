from types import SimpleNamespace

import pytest
import torch
import diffulex.attention.attn_impl as attn_impl
from diffulex.attention.attn_impl import Attention, reference_torch_attention
from diffulex.attention.metadata import is_warming_up, warming_up_context
from diffulex.mixin.edit.sampler import EditSamplerMixin
from diffulex.mixin.token_merge.sampler import TokenMergeSamplerMixin
from diffulex.sampler.auto_sampler import AutoSampler
from diffulex.sampler.base.shift import SamplerShiftLogits
from diffulex.sampler.dream import DreamSampler
from diffulex.sampler.llada import LLaDASampler
from diffulex.sampler.llada2 import LLaDA2DMaxSampler, LLaDA2Sampler, LLaDA2dot1Sampler
from diffulex.sampler.sdar import SDARSampler
from diffulex.sampler.base import SampleOutputBase
from diffulex.strategy_template.multi_block.engine.model_runner import MultiBlockModelRunnerTemplate
from diffulex.strategy_template.multi_block.engine.cudagraph import (
    MultiBlockCudaGraphState,
    MultiBlockGraphBuffers,
)
from diffulex.strategy_template.token_merge.attention.metadata import (
    TokenMergeAttnMetaDataTemplate,
)
from diffulex.strategy_template.token_merge.engine.cudagraph import (
    TokenMergeCudaGraphMixin,
)
from diffulex.strategy_template.token_merge.engine.model_runner import (
    TokenMergeModelRunnerTemplate,
)
from diffulex.engine.dllm_block import DllmBlock
from diffulex.config import DecodingThresholds


class _MultiBlockRunnerTestBase(MultiBlockModelRunnerTemplate):
    def __init__(self):
        pass

    def prepare_prefill(self, reqs):
        pass

    def prepare_decode(self, reqs):
        pass

    def run_model(self, input_ids, positions):
        pass

    def run(self, reqs):
        pass

    def capture_cudagraph(self):
        pass


class _Runner(_MultiBlockRunnerTestBase):
    page_size = 4
    block_size = 4


class _BatchRunner(_MultiBlockRunnerTestBase):
    rank = 0
    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def _run_multi_block_subgroup(self, reqs):
        self.calls.append([req.req_id for req in reqs])
        return SampleOutputBase(
            true_local_ids_map={},
            accepted_ids_map={},
            sampled_tokens_map={},
        )


class _TokenMergeRunner(TokenMergeModelRunnerTemplate):
    def __init__(self):
        self.config = SimpleNamespace(
            mask_token_id=99,
            token_merge_top_k=4,
            token_merge_renormalize=True,
            token_merge_mode="dmax_topk",
            token_merge_weight=1.0,
            buffer_size=1,
            block_size=4,
            max_num_reqs=4,
            max_num_batched_tokens=16,
            max_model_len=16,
            kv_cache_layout="unified",
        )

    def prepare_prefill(self, reqs):
        pass

    def prepare_decode(self, reqs):
        pass

    def run_model(self, input_ids, positions):
        pass

    def run(self, reqs):
        pass

    def capture_cudagraph(self):
        pass


def test_token_merge_runner_inherits_graph_hooks_from_graph_mixin() -> None:
    assert TokenMergeModelRunnerTemplate.__mro__[1] is TokenMergeCudaGraphMixin
    assert TokenMergeModelRunnerTemplate._decode_graph_extra_vars is TokenMergeCudaGraphMixin._decode_graph_extra_vars
    assert (
        TokenMergeModelRunnerTemplate._bind_decode_graph_extra_metadata
        is TokenMergeCudaGraphMixin._bind_decode_graph_extra_metadata
    )


def test_prepare_prefill_req_uses_suffix_positions_and_lengths_for_cached_prefix() -> None:
    req = SimpleNamespace(
        running_sequence=list(range(8, 20)),
        contiguous_in_cache_prefix_len=8,
        in_cache_len=8,
        running_len=20,
        page_table=[0, 1, 2, 3, 4],
        prefix_len=20,
        padded_prefix_len=20,
        dllm_blocks=[
            SimpleNamespace(start=0, end=4, rel_page_id=0, is_to_cache=False),
            SimpleNamespace(start=4, end=8, rel_page_id=1, is_to_cache=False),
            SimpleNamespace(start=8, end=12, rel_page_id=2, is_to_cache=True),
            SimpleNamespace(start=12, end=16, rel_page_id=3, is_to_cache=True),
            SimpleNamespace(start=16, end=20, rel_page_id=4, is_to_cache=True),
        ],
    )

    prepared = _Runner()._prepare_prefill_req(req)

    assert prepared["input_ids"] == list(range(8, 20))
    assert prepared["positions"] == list(range(8, 20))
    assert prepared["context_len"] == 8
    assert prepared["seqlen_q"] == 12
    assert prepared["seqlen_k"] == 12
    assert prepared["valid_slice"] == 12
    assert prepared["slot_mapping"] == list(range(8, 20))


def test_token_merge_graph_binding_copies_runtime_metadata_into_fixed_buffers() -> None:
    runner = _TokenMergeRunner()
    attn_metadata = TokenMergeAttnMetaDataTemplate()
    attn_metadata.init_token_merge(
        merge_mask=torch.tensor([False, True, False], dtype=torch.bool),
        topk_ids=torch.tensor([[99, 99], [7, 8], [99, 99]], dtype=torch.int64),
        topk_probs=torch.tensor([[0.0, 0.0], [0.6, 0.4], [0.0, 0.0]], dtype=torch.float32),
        residual_probs=torch.tensor([[0.0], [0.1], [0.0]], dtype=torch.float32),
        mask_token_id=99,
        renormalize=True,
        mode="dmax_topk",
        weight=1.0,
    )
    graph_vars = {
        "token_merge_mask": torch.zeros(3, dtype=torch.bool),
        "token_merge_topk_ids": torch.full((3, 4), 99, dtype=torch.int64),
        "token_merge_topk_probs": torch.zeros((3, 4), dtype=torch.float32),
        "token_merge_residual_probs": torch.zeros((3, 1), dtype=torch.float32),
    }

    runner._bind_graph_token_merge_metadata(attn_metadata, graph_vars, num_tokens=3)

    assert torch.equal(attn_metadata.token_merge_mask, torch.tensor([False, True, False], dtype=torch.bool))
    assert torch.equal(graph_vars["token_merge_mask"], torch.tensor([False, True, False], dtype=torch.bool))
    assert torch.equal(
        graph_vars["token_merge_topk_ids"],
        torch.tensor([[99, 99, 99, 99], [7, 8, 99, 99], [99, 99, 99, 99]], dtype=torch.int64),
    )
    assert torch.allclose(
        graph_vars["token_merge_topk_probs"],
        torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.6, 0.4, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    )
    assert torch.allclose(
        graph_vars["token_merge_residual_probs"],
        torch.tensor([[0.0], [0.1], [0.0]], dtype=torch.float32),
    )


def test_prefill_cuda_graph_is_disabled() -> None:
    runner = _Runner()
    runner.enforce_eager = False
    runner.config = SimpleNamespace(
        block_size=4,
        max_model_len=20,
        max_num_batched_tokens=20,
        max_num_reqs=4,
    )
    attn_metadata = TokenMergeAttnMetaDataTemplate()
    attn_metadata.status_table = torch.tensor([0], dtype=torch.int32)

    assert not runner._can_use_prefill_graph(attn_metadata, 17)


def test_decode_graph_gate_uses_captured_bucket_upper_bound() -> None:
    runner = _Runner()
    runner.enforce_eager = False
    runner.config = SimpleNamespace(
        block_size=4,
        buffer_size=1,
        max_num_reqs=2,
        max_model_len=16,
        kv_cache_layout="unified",
    )
    runner.cuda_graph_state = MultiBlockCudaGraphState(
        decode_bucket_tokens=[4, 8],
        decode_buffers=MultiBlockGraphBuffers(
            common={
                "input_ids": torch.zeros(8, dtype=torch.int64),
                "context_lens": torch.zeros(2, dtype=torch.int32),
                "page_tables": torch.zeros((2, 4), dtype=torch.int32),
            }
        ),
    )

    attn_metadata = TokenMergeAttnMetaDataTemplate(
        cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int32),
        page_tables=torch.zeros((1, 4), dtype=torch.int32),
        status_table=torch.tensor([1], dtype=torch.int32),
    )

    assert runner._can_use_decode_graph(attn_metadata, torch.zeros(8, dtype=torch.int64))
    assert not runner._can_use_decode_graph(attn_metadata, torch.zeros(12, dtype=torch.int64))


def test_token_merge_graph_binding_disables_merge_with_zero_mask_buffers() -> None:
    runner = _TokenMergeRunner()
    attn_metadata = TokenMergeAttnMetaDataTemplate()
    attn_metadata.init_token_merge(mask_token_id=99)
    graph_vars = {
        "token_merge_mask": torch.ones(2, dtype=torch.bool),
        "token_merge_topk_ids": torch.full((2, 4), -1, dtype=torch.int64),
        "token_merge_topk_probs": torch.full((2, 4), 1.0, dtype=torch.float32),
        "token_merge_residual_probs": torch.full((2, 1), 1.0, dtype=torch.float32),
    }

    runner._bind_graph_token_merge_metadata(attn_metadata, graph_vars, num_tokens=2)

    assert not bool(attn_metadata.token_merge_enabled)
    assert torch.equal(graph_vars["token_merge_mask"], torch.tensor([False, False], dtype=torch.bool))
    assert torch.equal(graph_vars["token_merge_topk_ids"], torch.full((2, 4), 99, dtype=torch.int64))
    assert torch.allclose(graph_vars["token_merge_topk_probs"], torch.zeros((2, 4), dtype=torch.float32))
    assert torch.allclose(graph_vars["token_merge_residual_probs"], torch.zeros((2, 1), dtype=torch.float32))


def test_token_merge_graph_capacity_vetoes_topk_over_capture_limit() -> None:
    runner = _TokenMergeRunner()
    attn_metadata = TokenMergeAttnMetaDataTemplate()
    attn_metadata.init_token_merge(
        merge_mask=torch.tensor([True], dtype=torch.bool),
        topk_ids=torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64),
        topk_probs=torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]], dtype=torch.float32),
        residual_probs=torch.tensor([[0.0]], dtype=torch.float32),
        mask_token_id=99,
        renormalize=True,
        mode="dmax_topk",
        weight=1.0,
    )

    assert not runner._can_use_decode_graph_extra(attn_metadata, num_tokens=1, captured_num_tokens=4)


def test_token_merge_decode_graph_binding_uses_graph_owned_metadata_without_mutating_source() -> None:
    runner = _TokenMergeRunner()
    source_mask = torch.tensor([False, True, False], dtype=torch.bool)
    source_topk_ids = torch.tensor([[99, 99], [7, 8], [99, 99]], dtype=torch.int64)
    source_topk_probs = torch.tensor([[0.0, 0.0], [0.6, 0.4], [0.0, 0.0]], dtype=torch.float32)
    source_residual_probs = torch.tensor([[0.0], [0.1], [0.0]], dtype=torch.float32)
    source_attn_metadata = TokenMergeAttnMetaDataTemplate()
    source_attn_metadata.init_token_merge(
        merge_mask=source_mask,
        topk_ids=source_topk_ids,
        topk_probs=source_topk_probs,
        residual_probs=source_residual_probs,
        mask_token_id=99,
        renormalize=True,
        mode="dmax_topk",
        weight=1.0,
    )
    replay_attn_metadata = TokenMergeAttnMetaDataTemplate()
    graph_vars = {
        "token_merge_mask": torch.zeros(3, dtype=torch.bool),
        "token_merge_topk_ids": torch.full((3, 4), 99, dtype=torch.int64),
        "token_merge_topk_probs": torch.zeros((3, 4), dtype=torch.float32),
        "token_merge_residual_probs": torch.zeros((3, 1), dtype=torch.float32),
    }

    runner._bind_decode_graph_extra_metadata(
        replay_attn_metadata,
        graph_vars,
        num_tokens=3,
        source_attn_metadata=source_attn_metadata,
    )

    assert replay_attn_metadata.token_merge_mask.data_ptr() == graph_vars["token_merge_mask"][:3].data_ptr()
    assert source_attn_metadata.token_merge_mask.data_ptr() == source_mask.data_ptr()
    assert torch.equal(source_attn_metadata.token_merge_topk_ids, source_topk_ids)


def test_warming_up_context_resets_flag_after_exception() -> None:
    assert not is_warming_up()

    with pytest.raises(RuntimeError, match="boom"):
        with warming_up_context():
            assert is_warming_up()
            raise RuntimeError("boom")

    assert not is_warming_up()


def test_prepare_prefill_req_maps_multiple_blocks_to_one_page() -> None:
    runner = _Runner()
    runner.page_size = 8
    runner.block_size = 4
    req = SimpleNamespace(
        running_sequence=list(range(8)),
        contiguous_in_cache_prefix_len=0,
        in_cache_len=0,
        running_len=8,
        page_table=[3],
        prefix_len=8,
        padded_prefix_len=8,
        dllm_blocks=[
            SimpleNamespace(start=0, end=4, rel_page_id=0, is_to_cache=True),
            SimpleNamespace(start=4, end=8, rel_page_id=0, is_to_cache=True),
        ],
    )

    prepared = runner._prepare_prefill_req(req)

    assert prepared["slot_mapping"] == list(range(24, 32))


def test_prepare_prefill_req_prefers_contiguous_cached_prefix_over_in_cache_len() -> None:
    req = SimpleNamespace(
        running_sequence=list(range(4, 12)),
        contiguous_in_cache_prefix_len=4,
        in_cache_len=8,
        running_len=12,
        page_table=[0, 1, 2],
        prefix_len=12,
        padded_prefix_len=12,
        dllm_blocks=[
            SimpleNamespace(start=0, end=4, rel_page_id=0, is_to_cache=False),
            SimpleNamespace(start=4, end=8, rel_page_id=1, is_to_cache=True),
            SimpleNamespace(start=8, end=12, rel_page_id=2, is_to_cache=True),
        ],
    )

    prepared = _Runner()._prepare_prefill_req(req)

    assert prepared["context_len"] == 4
    assert prepared["positions"] == list(range(4, 12))
    assert prepared["slot_mapping"] == list(range(4, 12))


def test_prepare_prefill_req_keeps_active_uncached_tail_in_valid_slice() -> None:
    req = SimpleNamespace(
        running_sequence=list(range(4, 8)),
        contiguous_in_cache_prefix_len=4,
        in_cache_len=4,
        running_len=8,
        page_table=[0, 1],
        prefix_len=8,
        padded_prefix_len=8,
        dllm_blocks=[
            SimpleNamespace(start=0, end=4, rel_page_id=0, is_to_cache=False),
            SimpleNamespace(start=4, end=8, rel_page_id=1, is_to_cache=False),
        ],
    )

    prepared = _Runner()._prepare_prefill_req(req)

    assert prepared["context_len"] == 4
    assert prepared["valid_slice"] == 4
    assert prepared["slot_mapping"] == [-1, -1, -1, -1]


def test_sampler_prefill_localizes_mask_token_indices_after_cached_prefix() -> None:
    sampler = SDARSampler()
    sampler.fetch_attn_metadata = lambda: SimpleNamespace(is_prefill=[True], cu_seqlens_q=None)

    block = SimpleNamespace(
        block_id=1,
        is_active=True,
        num_mask_tokens=4,
        mask_token_id=-1,
        mask_token_global_ids=[4, 5, 6, 7],
        mask_token_relative_ids=[0, 1, 2, 3],
        should_force_decode_topk=False,
    )
    req = SimpleNamespace(
        req_id=0,
        running_sequence=[-1, -1, -1, -1],
        chunk_size=4,
        dllm_blocks=[block],
        contiguous_in_cache_prefix_len=4,
        in_cache_len=4,
    )

    logits = torch.zeros(4, 8)
    logits[:, 0] = 1.0
    temperatures = torch.tensor([0.0])

    out = sampler([req], logits, temperatures, threshold=0.95)

    assert out.sampled_tokens_map["0"]["0"] == [0, 0, 0, 0]
    assert out.mask_token_rel_ids_map["0"]["0"] == [0, 1, 2, 3]


def test_sampler_greedy_tie_break_prefers_higher_token_id() -> None:
    sampler = SDARSampler()

    logits = torch.tensor([[1.0, 3.0, 3.0]], dtype=torch.float32)

    confidence, sampled_tokens, initial_confidence = sampler.sample_tokens(logits, temperature=0.0)

    assert sampled_tokens.tolist() == [2]
    assert confidence.tolist() == initial_confidence.tolist()


def test_sdar_sampler_forces_topk_for_initial_block() -> None:
    sampler = SDARSampler()
    block = SimpleNamespace(block_id=0, prev_block=None, should_force_decode_topk=False)
    confidence = torch.tensor([0.2, 0.7, 0.4])
    initial_confidence = torch.tensor([0.2, 0.7, 0.4])
    sampled_tokens = torch.tensor([10, 11, 12])

    accepted = sampler._compute_accepted_ids(block, confidence, initial_confidence, sampled_tokens, threshold=0.95)

    assert accepted.tolist() == [1]


def test_sdar_sampler_does_not_force_topk_for_later_block_without_ready_prev() -> None:
    sampler = SDARSampler()
    block = SimpleNamespace(block_id=1, prev_block=object(), should_force_decode_topk=False)
    confidence = torch.tensor([0.2, 0.7, 0.4])
    initial_confidence = torch.tensor([0.2, 0.7, 0.4])
    sampled_tokens = torch.tensor([10, 11, 12])

    accepted = sampler._compute_accepted_ids(block, confidence, initial_confidence, sampled_tokens, threshold=0.95)

    assert accepted.tolist() == []


def test_attention_uses_kernel_for_cached_prefix_prefill(monkeypatch) -> None:
    attn = Attention(num_heads=2, head_dim=4, scale=0.5, num_kv_heads=2)
    attn.k_cache = torch.zeros(1, 4, 2, 4)
    attn.v_cache = torch.zeros(1, 4, 2, 4)
    metadata = SimpleNamespace(
        kv_cache_layout="unified",
        need_kv_cache_store=False,
        use_multi_block=True,
        status_table=torch.tensor([0], dtype=torch.int32),
        context_lens=torch.tensor([4], dtype=torch.int32),
        page_tables=torch.tensor([[0]], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int32),
        valid_slices=torch.tensor([4], dtype=torch.int32),
        prefix_lens=torch.tensor([0], dtype=torch.int32),
        padded_prefix_lens=torch.tensor([0], dtype=torch.int32),
        page_size=4,
        block_size=4,
        is_block_causal=True,
        is_prefix_full=False,
    )
    attn.fetch_attn_metadata = lambda: metadata

    kernel_called = {"value": False}

    def _fake_kernel(q, k, v, k_cache, v_cache, attn_metadata):
        kernel_called["value"] = True
        return torch.zeros_like(q)

    monkeypatch.setattr(attn_impl, "chunked_prefill_attn_unified", _fake_kernel)

    qkv = torch.randn(4, 8)
    out = attn(qkv, qkv, qkv)

    assert kernel_called["value"] is True
    assert out.shape == qkv.shape


def test_attention_reference_path_is_factored_and_skips_kernel(monkeypatch) -> None:
    attn = Attention(num_heads=2, head_dim=4, scale=0.5, num_kv_heads=1, attn_impl="naive")

    kernel_called = {"value": False}

    def _fake_kernel(*args, **kwargs):
        kernel_called["value"] = True
        raise AssertionError("reference attention should not call the external kernel")

    monkeypatch.setattr(attn_impl, "chunked_prefill_attn_unified", _fake_kernel)

    q = torch.randn(4, 8)
    k = torch.randn(4, 4)
    v = torch.randn(4, 4)
    out = attn(q, k, v)
    expected = reference_torch_attention(
        q.reshape(4, 2, 4),
        k.reshape(4, 1, 4),
        v.reshape(4, 1, 4),
        num_heads=2,
        num_kv_heads=1,
        scale=0.5,
    ).reshape(4, 8)

    assert kernel_called["value"] is False
    torch.testing.assert_close(out, expected)


def test_attention_rejects_unknown_impl() -> None:
    with pytest.raises(ValueError, match="attn_impl must be one of"):
        Attention(num_heads=2, head_dim=4, scale=0.5, num_kv_heads=1, attn_impl="flash")


def test_run_multi_block_keeps_prefill_batch_together() -> None:
    runner = _BatchRunner()
    reqs = [
        SimpleNamespace(req_id=1, is_decoding=False, is_prefilling=True, contiguous_in_cache_prefix_len=0),
        SimpleNamespace(req_id=2, is_decoding=False, is_prefilling=True, contiguous_in_cache_prefix_len=4),
    ]

    runner.run_multi_block(reqs)

    assert runner.calls == [[1, 2]]


def test_llada_and_llada2_use_separate_sampler_classes() -> None:
    llada_cfg = SimpleNamespace(model_name="llada")
    llada2_cfg = SimpleNamespace(model_name="llada2", sampling_mode="naive", token_merge_top_k=1)
    llada2_moe_cfg = SimpleNamespace(model_name="llada2_moe", sampling_mode="naive", token_merge_top_k=1)
    llada2_mini_cfg = SimpleNamespace(model_name="llada2_mini", sampling_mode="naive", token_merge_top_k=1)
    edit_llada2_cfg = SimpleNamespace(model_name="llada2", sampling_mode="edit", token_merge_top_k=1)
    dmax_llada2_cfg = SimpleNamespace(
        model_name="llada2",
        sampling_mode="edit",
        token_merge_top_k=1,
        decoding_strategy="dmax",
    )

    llada_sampler = AutoSampler.from_config(llada_cfg)
    llada2_sampler = AutoSampler.from_config(llada2_cfg)
    llada2_moe_sampler = AutoSampler.from_config(llada2_moe_cfg)
    llada2_mini_sampler = AutoSampler.from_config(llada2_mini_cfg)
    edit_llada2_sampler = AutoSampler.from_config(edit_llada2_cfg)
    dmax_llada2_sampler = AutoSampler.from_config(dmax_llada2_cfg)

    assert isinstance(llada_sampler, LLaDASampler)
    assert isinstance(llada2_sampler, LLaDA2Sampler)
    assert isinstance(llada2_moe_sampler, LLaDA2Sampler)
    assert isinstance(llada2_mini_sampler, LLaDA2Sampler)
    assert isinstance(edit_llada2_sampler, LLaDA2dot1Sampler)
    assert isinstance(edit_llada2_sampler, EditSamplerMixin)
    assert isinstance(dmax_llada2_sampler, LLaDA2DMaxSampler)
    assert isinstance(dmax_llada2_sampler, EditSamplerMixin)
    assert isinstance(dmax_llada2_sampler, TokenMergeSamplerMixin)
    assert type(llada_sampler) is not type(llada2_sampler)
    assert type(llada2_sampler) is LLaDA2Sampler
    assert type(edit_llada2_sampler) is LLaDA2dot1Sampler
    assert type(dmax_llada2_sampler) is LLaDA2DMaxSampler


def test_auto_sampler_rejects_invalid_sampling_mode_combo() -> None:
    with pytest.raises(ValueError, match="Unsupported sampler configuration"):
        AutoSampler.from_config(
            SimpleNamespace(model_name="llada2dot1_mini", decoding_strategy=None, sampling_mode="naive")
        )


def test_llada2dot1_edit_sampler_emits_edit_writes_map() -> None:
    sampler = LLaDA2dot1Sampler(SimpleNamespace(sampling_mode="edit", token_merge_top_k=1))
    sampler.fetch_attn_metadata = lambda: SimpleNamespace(is_prefill=[True], cu_seqlens_q=None)

    block = SimpleNamespace(
        block_id=1,
        start=4,
        end=6,
        block_size=2,
        is_active=True,
        num_mask_tokens=2,
        mask_token_global_ids=[4, 5],
        mask_token_relative_ids=[0, 1],
        token_ids=[0, 0],
        mask_token_id=0,
        prev_block=SimpleNamespace(is_semi_complete=True),
        thresholds=SimpleNamespace(accept_threshold=0.6, remask_threshold=0.4),
    )
    req = SimpleNamespace(
        req_id=0,
        running_sequence=[-1, -1],
        chunk_size=2,
        dllm_blocks=[block],
        contiguous_in_cache_prefix_len=4,
        in_cache_len=4,
    )

    logits = torch.tensor(
        [
            [0.1, 3.0, 0.0],
            [0.1, 0.0, 3.0],
        ],
        dtype=torch.float32,
    )
    temperatures = torch.tensor([0.0], dtype=torch.float32)

    out = sampler([req], logits, temperatures)

    assert out.edit_writes_map["0"]["0"] == {0: 1, 1: 2}
    assert not hasattr(out, "token_merge_map")


def test_llada2dmax_sampler_emits_edit_writes_and_token_merge_map() -> None:
    sampler = LLaDA2DMaxSampler(
        SimpleNamespace(
            sampling_mode="edit",
            token_merge_top_k=1,
            token_merge_mode="dmax_topk",
            token_merge_weight=1.0,
        )
    )
    sampler.fetch_attn_metadata = lambda: SimpleNamespace(is_prefill=[True], cu_seqlens_q=None)

    block = SimpleNamespace(
        block_id=0,
        start=4,
        end=7,
        block_size=3,
        is_active=True,
        num_mask_tokens=2,
        mask_token_global_ids=[5, 6],
        mask_token_relative_ids=[1, 2],
        token_ids=[9, 0, 0],
        mask_token_id=0,
        prev_block=SimpleNamespace(is_semi_complete=True),
        thresholds=SimpleNamespace(accept_threshold=0.9, remask_threshold=0.5),
    )
    req = SimpleNamespace(
        req_id=0,
        running_sequence=[9, 0, 0],
        chunk_size=3,
        dllm_blocks=[block],
        contiguous_in_cache_prefix_len=4,
        in_cache_len=4,
    )

    logits = torch.tensor([[0.0, 0.4, 0.3], [0.0, 4.0, 0.0], [0.0, 0.0, 3.0]], dtype=torch.float32)
    temperatures = torch.tensor([0.0], dtype=torch.float32)

    out = sampler([req], logits, temperatures)

    assert out.edit_writes_map["0"]["0"] == {0: 1, 1: 1, 2: 2}
    req_merge = out.token_merge_map["0"]
    assert req_merge[4]["topk_ids"] == [1]
    assert req_merge[4]["residual_prob"] == 0.0
    assert req_merge[5]["topk_ids"] == [1]
    assert req_merge[6]["topk_ids"] == [2]


def test_llada2dmax_sampler_does_not_expose_probability_path_toggle() -> None:
    sampler = LLaDA2DMaxSampler(
        SimpleNamespace(
            sampling_mode="edit",
            token_merge_top_k=1,
            token_merge_mode="dmax_topk",
            token_merge_weight=1.0,
        )
    )

    assert not hasattr(sampler, "_fast_prob_path")


def test_dllm_block_editable_start_limits_mask_ids_and_writes() -> None:
    class _Req:
        page_size = 4

        def __init__(self) -> None:
            self.token_ids = [101, 102, 0, 0]

        def __getitem__(self, s):
            return self.token_ids[s]

    req = _Req()

    block = DllmBlock(
        block_id=0,
        start=0,
        end=4,
        block_size=4,
        mask_token_id=0,
        thresholds=DecodingThresholds(0.1, 0.9, 0.95, 0.4),
        editable_start=2,
    )
    block.post_init_dllm_block(req, None)

    assert block.mask_token_relative_ids == [2, 3]
    assert block.mask_token_global_ids == [2, 3]
    assert block.num_mask_tokens == 2

    with pytest.raises(ValueError, match="non-editable token"):
        block.write_token(999, 1)


def test_dllm_block_edit_state_is_derived_from_token_snapshots_and_confidence() -> None:
    class _Req:
        page_size = 4

        def __init__(self) -> None:
            self.token_ids = [101, 102, 0, 0]

        def __getitem__(self, s):
            return self.token_ids[s]

    block = DllmBlock(
        block_id=0,
        start=0,
        end=4,
        block_size=4,
        mask_token_id=0,
        thresholds=DecodingThresholds(0.1, 0.9, 0.95, 0.4),
        editable_start=2,
    )
    block.post_init_dllm_block(_Req(), None)

    block.observe_edit_state([101, 102, 7, 8], confidences=[1.0, 1.0, 0.95, 0.96])
    assert block.previous_token_ids == [101, 102, 0, 0]
    assert block.current_token_ids == [101, 102, 7, 8]
    assert block.same_token_ratio == 0.0
    assert block.same_as_previous is False
    assert block.all_confident is True
    assert block.commit_ready is True

    block.observe_edit_state([101, 102, 7, 9], confidences=[1.0, 1.0, 0.95, 0.40])
    assert block.same_token_ratio == 0.5
    assert block.same_as_previous is False
    assert block.all_confident is False
    assert block.commit_ready is False


def test_llada2dmax_sampler_respects_block_editable_start() -> None:
    sampler = LLaDA2DMaxSampler(
        SimpleNamespace(
            sampling_mode="edit",
            token_merge_top_k=1,
            token_merge_mode="dmax_topk",
            token_merge_weight=1.0,
        )
    )
    sampler.fetch_attn_metadata = lambda: SimpleNamespace(is_prefill=[True], cu_seqlens_q=None)

    class _Block:
        block_id = 0
        start = 4
        end = 8
        block_size = 4
        is_active = True
        token_ids = [11, 12, 0, 0]
        mask_token_id = 0
        prev_block = SimpleNamespace(is_semi_complete=True)
        thresholds = SimpleNamespace(accept_threshold=0.9, remask_threshold=0.5)
        editable_start = 2

        @property
        def num_mask_tokens(self):
            return 2

        @property
        def mask_token_global_ids(self):
            return [6, 7]

        @property
        def mask_token_relative_ids(self):
            return [2, 3]

    block = _Block()
    req = SimpleNamespace(
        req_id=0,
        running_sequence=[11, 12, 0, 0],
        chunk_size=4,
        dllm_blocks=[block],
        contiguous_in_cache_prefix_len=4,
        in_cache_len=4,
    )

    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=torch.float32,
    )
    temperatures = torch.tensor([0.0], dtype=torch.float32)

    out = sampler([req], logits, temperatures)

    assert out.edit_writes_map["0"]["0"] == {2: 1, 3: 2}
    req_merge = out.token_merge_map["0"]
    assert req_merge[4] is None
    assert req_merge[5] is None
    assert req_merge[6]["topk_ids"] == [1]
    assert req_merge[7]["topk_ids"] == [2]


def test_llada2dmax_sampler_does_not_treat_mask_predictions_as_progress() -> None:
    sampler = LLaDA2DMaxSampler(
        SimpleNamespace(
            sampling_mode="edit",
            token_merge_top_k=1,
            token_merge_mode="dmax_topk",
            token_merge_weight=1.0,
        )
    )
    sampler.fetch_attn_metadata = lambda: SimpleNamespace(is_prefill=[True], cu_seqlens_q=None)

    block = SimpleNamespace(
        block_id=0,
        start=0,
        end=3,
        block_size=3,
        is_active=True,
        num_mask_tokens=2,
        mask_token_global_ids=[1, 2],
        mask_token_relative_ids=[1, 2],
        token_ids=[9, 0, 0],
        mask_token_id=0,
        prev_block=SimpleNamespace(is_semi_complete=True),
        thresholds=SimpleNamespace(accept_threshold=0.9, remask_threshold=0.5),
        editable_start=1,
    )
    req = SimpleNamespace(
        req_id=0,
        running_sequence=[9, 0, 0],
        chunk_size=3,
        dllm_blocks=[block],
        contiguous_in_cache_prefix_len=0,
        in_cache_len=0,
    )

    logits = torch.tensor(
        [
            [0.0, 2.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=torch.float32,
    )
    temperatures = torch.tensor([0.0], dtype=torch.float32)

    out = sampler([req], logits, temperatures)

    assert out.edit_writes_map["0"]["0"] == {2: 2}
    req_merge = out.token_merge_map["0"]
    assert req_merge[0] is None
    assert req_merge[1] is None
    assert req_merge[2]["topk_ids"] == [2]


def test_llada2dmax_sampler_decodes_leftmost_mask_prefix() -> None:
    sampler = LLaDA2DMaxSampler(
        SimpleNamespace(
            sampling_mode="edit",
            token_merge_top_k=1,
            token_merge_mode="dmax_topk",
            token_merge_weight=1.0,
        )
    )
    sampler.fetch_attn_metadata = lambda: SimpleNamespace(is_prefill=[True], cu_seqlens_q=None)

    block = SimpleNamespace(
        block_id=0,
        start=0,
        end=3,
        block_size=3,
        is_active=True,
        num_mask_tokens=3,
        mask_token_global_ids=[0, 1, 2],
        mask_token_relative_ids=[0, 1, 2],
        token_ids=[0, 0, 0],
        mask_token_id=0,
        prev_block=SimpleNamespace(is_semi_complete=True),
        thresholds=SimpleNamespace(accept_threshold=0.95, remask_threshold=0.4),
    )
    req = SimpleNamespace(
        req_id=0,
        running_sequence=[0, 0, 0],
        chunk_size=3,
        dllm_blocks=[block],
        contiguous_in_cache_prefix_len=0,
        in_cache_len=0,
    )

    logits = torch.tensor(
        [
            [0.0, 5.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.1, 0.0],
        ],
        dtype=torch.float32,
    )

    out = sampler([req], logits, torch.tensor([0.0], dtype=torch.float32))

    assert out.edit_writes_map["0"]["0"] == {0: 1, 1: 1}
    req_merge = out.token_merge_map["0"]
    assert req_merge[0]["topk_ids"] == [1]
    assert req_merge[1]["topk_ids"] == [1]
    assert req_merge[2] is None


def test_llada2dmax_sampler_refreshes_editable_non_mask_tokens() -> None:
    sampler = LLaDA2DMaxSampler(
        SimpleNamespace(
            sampling_mode="edit",
            token_merge_top_k=1,
            token_merge_mode="dmax_topk",
            token_merge_weight=1.0,
        )
    )
    sampler.fetch_attn_metadata = lambda: SimpleNamespace(is_prefill=[True], cu_seqlens_q=None)

    block = SimpleNamespace(
        block_id=0,
        start=10,
        end=14,
        block_size=4,
        is_active=True,
        num_mask_tokens=2,
        mask_token_global_ids=[12, 13],
        mask_token_relative_ids=[2, 3],
        token_ids=[7, 8, 0, 0],
        mask_token_id=0,
        prev_block=SimpleNamespace(is_semi_complete=True),
        thresholds=SimpleNamespace(accept_threshold=0.8, remask_threshold=0.4),
        editable_start=0,
    )
    req = SimpleNamespace(
        req_id=0,
        running_sequence=[7, 8, 0, 0],
        chunk_size=4,
        dllm_blocks=[block],
        contiguous_in_cache_prefix_len=10,
        in_cache_len=10,
    )

    logits = torch.tensor(
        [
            [0.0, 0.0, 5.0],
            [0.0, 4.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.1, 0.0],
        ],
        dtype=torch.float32,
    )

    out = sampler([req], logits, torch.tensor([0.0], dtype=torch.float32))

    assert out.edit_writes_map["0"]["0"] == {0: 2, 1: 1, 2: 1}
    req_merge = out.token_merge_map["0"]
    assert req_merge[10]["topk_ids"] == [2]
    assert req_merge[11]["topk_ids"] == [1]
    assert req_merge[12]["topk_ids"] == [1]
    assert req_merge[13] is None


def test_llada2dmax_sampler_emits_confidence_mask_blend_descriptors_on_dmax_path() -> None:
    sampler = LLaDA2DMaxSampler(
        SimpleNamespace(
            sampling_mode="edit",
            token_merge_top_k=1,
            token_merge_mode="dmax_topk",
            token_merge_weight=1.0,
        )
    )
    sampler.fetch_attn_metadata = lambda: SimpleNamespace(is_prefill=[True], cu_seqlens_q=None)

    block = SimpleNamespace(
        block_id=0,
        start=0,
        end=2,
        block_size=2,
        is_active=True,
        num_mask_tokens=1,
        mask_token_global_ids=[1],
        mask_token_relative_ids=[1],
        token_ids=[9, 0],
        mask_token_id=0,
        prev_block=SimpleNamespace(is_semi_complete=True),
        thresholds=SimpleNamespace(accept_threshold=0.5, remask_threshold=0.4),
    )
    req = SimpleNamespace(
        req_id=0,
        running_sequence=[9, 0],
        chunk_size=2,
        dllm_blocks=[block],
        contiguous_in_cache_prefix_len=0,
        in_cache_len=0,
    )

    logits = torch.tensor(
        [
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=torch.float32,
    )

    out = sampler([req], logits, torch.tensor([0.0], dtype=torch.float32))

    assert out.edit_writes_map["0"]["0"] == {0: 1, 1: 2}
    assert out.token_merge_map["0"][0]["topk_ids"] == [1]
    assert out.token_merge_map["0"][0]["residual_prob"] == 0.0
    assert out.token_merge_map["0"][1]["topk_ids"] == [2]
    assert out.token_merge_map["0"][1]["residual_prob"] > 0.0


def test_llada_sampler_does_not_resample_mask_token() -> None:
    sampler = LLaDASampler()
    sampler.fetch_attn_metadata = lambda: SimpleNamespace(is_prefill=[True], cu_seqlens_q=None)

    block = SimpleNamespace(
        block_id=0,
        is_active=True,
        num_mask_tokens=1,
        mask_token_id=0,
        mask_token_global_ids=[0],
        mask_token_relative_ids=[0],
        prev_block=None,
        should_force_decode_topk=False,
        thresholds=SimpleNamespace(accept_threshold=0.99),
    )
    req = SimpleNamespace(
        req_id=0,
        running_sequence=[0],
        chunk_size=1,
        dllm_blocks=[block],
        contiguous_in_cache_prefix_len=0,
    )

    logits = torch.tensor([[10.0, 5.0, 4.0]], dtype=torch.float32)
    temperatures = torch.tensor([0.0], dtype=torch.float32)

    out = sampler([req], logits, temperatures)

    assert out.sampled_tokens_map["0"]["0"] == [1]


def test_llada_sampler_does_not_sample_tokenizer_padding_vocab() -> None:
    sampler = LLaDASampler()
    sampler.tokenizer_vocab_size = 3

    logits = torch.tensor([[0.1, 0.2, 0.3, 10.0]], dtype=torch.float32)

    _, sampled_tokens, _ = sampler.sample_tokens(logits, temperature=0.0)

    assert sampled_tokens.tolist() == [2]


def test_llada_sampler_sanitizes_nan_logits_before_sampling() -> None:
    sampler = LLaDASampler()
    sampler.tokenizer_vocab_size = 3

    logits = torch.tensor([[1.0, 2.0, 3.0, float("nan")]], dtype=torch.float32)

    _, sampled_tokens, _ = sampler.sample_tokens(logits, temperature=0.0)

    assert sampled_tokens.tolist() == [2]


def test_llada_sampler_forces_topk_for_initial_block() -> None:
    sampler = LLaDASampler()
    block = SimpleNamespace(
        block_id=0,
        prev_block=None,
        should_force_decode_topk=False,
        thresholds=SimpleNamespace(accept_threshold=0.95),
    )
    confidence = torch.tensor([0.2, 0.7, 0.4])
    initial_confidence = torch.tensor([0.2, 0.7, 0.4])
    sampled_tokens = torch.tensor([10, 11, 12])

    accepted = sampler._compute_accepted_ids(block, confidence, initial_confidence, sampled_tokens)

    assert accepted.tolist() == [1]


def test_dream_sampler_forces_topk_when_prev_block_is_complete() -> None:
    sampler = DreamSampler()
    block = SimpleNamespace(
        prev_block=SimpleNamespace(is_semi_complete=True),
        thresholds=SimpleNamespace(accept_threshold=0.95),
    )
    confidence = torch.tensor([0.2, 0.7, 0.4])
    initial_confidence = torch.tensor([0.2, 0.7, 0.4])
    sampled_tokens = torch.tensor([10, 11, 12])

    accepted = sampler._compute_accepted_ids(block, confidence, initial_confidence, sampled_tokens)

    assert accepted.tolist() == [1]


def test_dream_sampler_does_not_force_topk_when_prev_block_not_complete() -> None:
    sampler = DreamSampler()
    block = SimpleNamespace(
        prev_block=SimpleNamespace(is_semi_complete=False),
        thresholds=SimpleNamespace(accept_threshold=0.95),
    )
    confidence = torch.tensor([0.2, 0.7, 0.4])
    initial_confidence = torch.tensor([0.2, 0.7, 0.4])
    sampled_tokens = torch.tensor([10, 11, 12])

    accepted = sampler._compute_accepted_ids(block, confidence, initial_confidence, sampled_tokens)

    assert accepted.tolist() == []


def test_shift_sampler_caches_last_logits_as_independent_tensor() -> None:
    sampler = SamplerShiftLogits()
    req = SimpleNamespace(req_id=7, has_to_cache_block=False)
    logits = torch.randn(4, 8, dtype=torch.float32)

    cached = sampler._fetch_last_logits(logits, req)

    assert cached.shape == logits[-1].shape
    assert cached.data_ptr() != logits[-1].data_ptr()


def test_shift_sampler_evict_req_states_removes_cached_logits() -> None:
    sampler = SamplerShiftLogits()
    req = SimpleNamespace(req_id=9, has_to_cache_block=False)
    logits = torch.randn(2, 4, dtype=torch.float32)
    sampler._fetch_last_logits(logits, req)

    assert "9" in sampler.req_last_logits_map
    sampler.evict_req_states([9])
    assert "9" not in sampler.req_last_logits_map
