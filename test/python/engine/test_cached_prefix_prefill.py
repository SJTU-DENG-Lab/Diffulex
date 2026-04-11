from types import SimpleNamespace

import torch
import torch.nn.functional as F
from einops import rearrange

import diffulex.attention.attn_impl as attn_impl
from diffulex.attention.attn_impl import Attention, reference_multi_block_attn
from diffulex.mixin.multi_block.engine.model_runner import ModelRunnerMultiBlockMixin
from diffulex.sampler.sdar import SDARSampler
from diffulex.sampler.base import SampleOutputBase


class _Runner(ModelRunnerMultiBlockMixin):
    page_size = 4
    block_size = 4


class _BatchRunner(ModelRunnerMultiBlockMixin):
    rank = 0
    _serialize_multi_block_reqs = False

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
    sampler.fetch_attn_metadata = lambda: SimpleNamespace(is_prefill=[True])

    block = SimpleNamespace(
        block_id=1,
        is_active=True,
        num_mask_tokens=4,
        mask_token_global_ids=[4, 5, 6, 7],
        mask_token_relative_ids=[0, 1, 2, 3],
        should_force_decode_topk=False,
    )
    req = SimpleNamespace(
        req_id=0,
        running_sequence=[-1, -1, -1, -1],
        chunk_size=4,
        dllm_blocks=[block],
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


def test_reference_cached_prefix_attn_matches_full_sequence_block_causal() -> None:
    torch.manual_seed(0)

    seq_len = 8
    ctx_len = 4
    num_heads = 2
    head_dim = 4

    q_full = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float32)
    k_full = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float32)
    v_full = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float32)

    q_suffix = q_full[ctx_len:].clone()
    k_suffix = k_full[ctx_len:].clone()
    v_suffix = v_full[ctx_len:].clone()

    k_cache = torch.zeros(1, ctx_len, num_heads, head_dim, dtype=torch.float32)
    v_cache = torch.zeros(1, ctx_len, num_heads, head_dim, dtype=torch.float32)
    k_cache[0, :ctx_len] = k_full[:ctx_len]
    v_cache[0, :ctx_len] = v_full[:ctx_len]

    metadata = SimpleNamespace(
        page_tables=torch.tensor([[0]], dtype=torch.int32),
        context_lens=torch.tensor([ctx_len], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, seq_len - ctx_len], dtype=torch.int32),
        valid_slices=torch.tensor([seq_len - ctx_len], dtype=torch.int32),
        page_size=ctx_len,
        block_size=ctx_len,
        is_prefix_full=False,
        status_table=torch.tensor([0], dtype=torch.int32),
        prefix_lens=torch.tensor([0], dtype=torch.int32),
        padded_prefix_lens=torch.tensor([0], dtype=torch.int32),
    )

    cached_out = reference_multi_block_attn(
        q_suffix,
        k_suffix,
        v_suffix,
        k_cache,
        v_cache,
        metadata,
        scale=1.0 / (head_dim**0.5),
    )

    abs_q = torch.arange(seq_len)
    abs_k = torch.arange(seq_len)
    block_mask = abs_k[None, :] < ((((abs_q // ctx_len) + 1) * ctx_len)[:, None])
    full_out = F.scaled_dot_product_attention(
        rearrange(q_full, "s h d -> 1 h s d"),
        rearrange(k_full, "s h d -> 1 h s d"),
        rearrange(v_full, "s h d -> 1 h s d"),
        attn_mask=block_mask.unsqueeze(0).unsqueeze(0),
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / (head_dim**0.5),
        enable_gqa=True,
    )
    full_out = rearrange(full_out, "1 h s d -> s h d")[ctx_len:]

    torch.testing.assert_close(cached_out, full_out, atol=1e-5, rtol=1e-5)


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

    def _fail_reference(*args, **kwargs):
        raise AssertionError("cached-prefix prefill should stay on the Triton kernel path")

    monkeypatch.setattr(attn_impl, "chunked_prefill_attn_unified", _fake_kernel)
    monkeypatch.setattr(attn_impl, "reference_multi_block_attn", _fail_reference)

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
