from types import SimpleNamespace

import pytest
import torch
import diffulex.attention.attn_impl as attn_impl
from diffulex.attention.attn_impl import Attention
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

    logits = torch.tensor(
        [
            [0.0, 0.4, 0.3],  # low confidence non-mask position -> remask
            [0.0, 4.0, 0.0],  # confident fill
            [0.0, 0.0, 3.0],  # confident fill
        ],
        dtype=torch.float32,
    )
    temperatures = torch.tensor([0.0], dtype=torch.float32)

    out = sampler([req], logits, temperatures)

    assert out.edit_writes_map["0"]["0"] == {0: 0, 1: 1, 2: 2}
    req_merge = out.token_merge_map["0"]
    assert req_merge[4] is None
    assert req_merge[5]["topk_ids"] == [1]
    assert req_merge[6]["topk_ids"] == [2]


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


def test_llada2dmax_sampler_respects_block_editable_start() -> None:
    sampler = LLaDA2DMaxSampler(
        SimpleNamespace(
            sampling_mode="edit",
            token_merge_top_k=1,
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
