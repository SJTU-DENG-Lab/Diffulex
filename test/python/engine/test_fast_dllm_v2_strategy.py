from types import SimpleNamespace

import torch

from diffulex.config import Config
from diffulex.engine.kv_cache_manager import AutoKVCacheManager
from diffulex.engine.model_runner import AutoModelRunner
from diffulex.engine.request import AutoReq
from diffulex.engine.scheduler import AutoScheduler
from diffulex.engine.status import DllmBlockStatus, DllmReqStatus
from diffulex.sampler.fast_dllm_v2 import FastdLLMV2Sampler
from diffulex.strategy.fast_dllm_v2.engine.kv_cache_manager import FastDLLMV2KVCacheManager
from diffulex.strategy.fast_dllm_v2.engine.model_runner import FastDLLMV2ModelRunner
from diffulex.strategy.fast_dllm_v2.engine.request import FastDLLMV2Mode, FastDLLMV2Req
from diffulex.strategy.fast_dllm_v2.engine.scheduler import FastDLLMV2Scheduler
from diffulex.strategy.multi_bd.config import MultiBDStrategyConfig
from diffulex.strategy.multi_bd.engine.kv_cache_manager import MultiBDKVCacheManager
from diffulex.strategy.multi_bd.engine.model_runner import MultiBDModelRunner
from diffulex.strategy.multi_bd.engine.request import MultiBDReq
from diffulex.strategy.multi_bd.engine.scheduler import MultiBDScheduler


def _runtime_config(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        """
        {
          "model_type": "llama",
          "hidden_size": 16,
          "intermediate_size": 32,
          "num_attention_heads": 2,
          "num_hidden_layers": 1,
          "num_key_value_heads": 2,
          "vocab_size": 128,
          "max_position_embeddings": 4096
        }
        """,
        encoding="utf-8",
    )
    return Config(
        str(model_dir),
        model_name="fast_dllm_v2",
        decoding_strategy="fast_dllm_v2",
        block_size=8,
        buffer_size=4,
        tensor_parallel_size=1,
        data_parallel_size=1,
        device_ids=[0],
        num_pages=8,
    )


def _req_config():
    return SimpleNamespace(
        block_size=8,
        buffer_size=4,
        mask_token_id=99,
        decoding_thresholds=SimpleNamespace(
            add_block_threshold=0.1,
            semi_complete_threshold=0.9,
            accept_threshold=0.95,
            token_stability_threshold=0.0,
        ),
        eos=-1,
        max_model_len=256,
    )


def _init_decode_req() -> FastDLLMV2Req:
    req = FastDLLMV2Req([1, 2, 3, 4, 5, 6, 7, 8])
    req.page_size = 8
    req.init_multi_block(_req_config())
    req.dllm_blocks[0].in_cache()
    req.status = DllmReqStatus.DECODING
    req.status_history.append(DllmReqStatus.DECODING)
    return req


def test_fast_dllm_v2_strategy_registers_runtime_components(tmp_path):
    cfg = _runtime_config(tmp_path)

    assert cfg.multi_block_prefix_full is False
    assert cfg.enable_prefix_caching is True
    assert cfg.strategy.name == "fast_dllm_v2"
    assert cfg.strategy.sub_block_size == 8
    assert cfg.strategy.block_size == 32

    assert isinstance(AutoReq.create(cfg, [1, 2, 3]), FastDLLMV2Req)
    assert isinstance(AutoKVCacheManager.from_config(cfg), FastDLLMV2KVCacheManager)
    assert isinstance(AutoScheduler.from_config(cfg), FastDLLMV2Scheduler)
    assert AutoModelRunner._MODULE_MAPPING["fast_dllm_v2"] is FastDLLMV2ModelRunner

    auto_cfg = Config(
        cfg.model,
        model_name="fast_dllm_v2",
        block_size=8,
        buffer_size=4,
        tensor_parallel_size=1,
        data_parallel_size=1,
        device_ids=[0],
        num_pages=8,
    )
    assert auto_cfg.decoding_strategy == "fast_dllm_v2"


def test_fast_dllm_v2_can_use_plain_multi_bd_strategy(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        """
        {
          "model_type": "llama",
          "hidden_size": 16,
          "intermediate_size": 32,
          "num_attention_heads": 2,
          "num_hidden_layers": 1,
          "num_key_value_heads": 2,
          "vocab_size": 128,
          "max_position_embeddings": 4096
        }
        """,
        encoding="utf-8",
    )

    cfg = Config(
        str(model_dir),
        model_name="fast_dllm_v2",
        decoding_strategy="multi_bd",
        block_size=8,
        buffer_size=1,
        tensor_parallel_size=1,
        data_parallel_size=1,
        device_ids=[0],
        num_pages=8,
    )

    assert cfg.decoding_strategy == "multi_bd"
    assert isinstance(cfg.strategy, MultiBDStrategyConfig)
    assert isinstance(AutoReq.create(cfg, [1, 2, 3]), MultiBDReq)
    assert isinstance(AutoKVCacheManager.from_config(cfg), MultiBDKVCacheManager)
    assert isinstance(AutoScheduler.from_config(cfg), MultiBDScheduler)
    assert AutoModelRunner._MODULE_MAPPING["multi_bd"] is MultiBDModelRunner


def test_fast_dllm_v2_request_activates_whole_buffer_and_advances_modes():
    req = _init_decode_req()

    req.step()

    assert req.fdv2_mode == FastDLLMV2Mode.FULL_BUFFER_INIT
    assert all(block.is_active for block in req.dllm_block_buffer.dllm_blocks)

    req.postprocess()
    assert req.fdv2_current_buffer_initialized is True

    first_block = req.fdv2_current_sub_block
    for rel_idx in first_block.mask_token_relative_ids:
        first_block.write_token(10 + rel_idx, rel_idx)

    req.step()
    assert req.fdv2_mode == FastDLLMV2Mode.FULL_BUFFER_INIT
    assert req.fdv2_current_sub_block_idx == 2

    req.fdv2_current_sub_block.write_token(777, 0)
    req.step()
    assert req.fdv2_mode == FastDLLMV2Mode.SUB_BLOCK_REFINE

    for block in req.dllm_block_buffer.dllm_blocks:
        for rel_idx in block.mask_token_relative_ids:
            block.write_token(20 + rel_idx, rel_idx)

    req.step()
    assert req.fdv2_mode == FastDLLMV2Mode.FINAL_COMMIT


def test_fast_dllm_v2_final_commit_seeds_next_block_first_token():
    req = _init_decode_req()

    for block in req.dllm_block_buffer.dllm_blocks:
        for rel_idx in block.mask_token_relative_ids:
            block.write_token(20 + rel_idx, rel_idx)
        block.status = DllmBlockStatus.TO_CACHE
        block.commit_ready = True

    req.fdv2_mode = FastDLLMV2Mode.FINAL_COMMIT
    req.fdv2_pending_next_token_id = 777

    req.postprocess()

    first_block = req.dllm_block_buffer.dllm_blocks[0]
    assert first_block.block_id == 4
    assert req.token_ids[first_block.start] == 777
    assert req.fdv2_pending_next_token_id is None
    assert req.new_tokens == 1


def test_fast_dllm_v2_kv_manager_allocates_current_pages_without_hashing():
    manager = FastDLLMV2KVCacheManager(SimpleNamespace(num_pages=8, page_size=8, enable_prefix_caching=True))
    req = _init_decode_req()
    req.status = DllmReqStatus.PREFILLING
    req.page_table = [0]
    manager._allocate_page(0)
    prefix_page = req.token_ids[: req.page_size]
    manager.pages[0].update(manager.compute_hash(prefix_page), prefix_page)
    req.status = DllmReqStatus.DECODING

    manager.may_append(req)

    assert len(req.page_table) == req.fdv2_read_cache_pages
    for page_id in req.page_table[1:]:
        assert manager.pages[page_id].hash == -1


def test_fast_dllm_v2_sampler_uses_current_sub_block_and_local_shift(monkeypatch):
    req = _init_decode_req()
    req.step()
    req.postprocess()
    req.fdv2_mode = FastDLLMV2Mode.SUB_BLOCK_REFINE
    req.fdv2_current_sub_block_idx = 2

    block = req.fdv2_current_sub_block
    block.status = DllmBlockStatus.ACTIVE
    token_ids = list(block.token_ids)
    for rel_idx in range(4):
        block.write_token(10 + rel_idx, rel_idx)
    for rel_idx in range(4, 8):
        req.token_ids[block.start + rel_idx] = block.mask_token_id

    sampler = FastdLLMV2Sampler()
    monkeypatch.setattr(
        sampler,
        "fetch_attn_metadata",
        lambda: SimpleNamespace(
            cu_seqlens_q=torch.tensor([0, 8], dtype=torch.int32),
            is_prefill=[False],
        ),
    )

    logits = torch.zeros(8, 16)
    for row in range(8):
        logits[row, row] = 10.0

    out = sampler([req], logits, torch.tensor([0.0]))
    req_out = out.true_local_ids_map[str(req.req_id)]

    assert set(req_out.keys()) == {str(block.block_id)}
    assert req_out[str(block.block_id)]
    assert min(req_out[str(block.block_id)]) >= 4
    assert all(str(other.block_id) not in req_out for other in req.dllm_block_buffer.dllm_blocks if other is not block)

    req.token_ids[block.start : block.end] = token_ids


def test_fast_dllm_v2_native_sampler_forces_top1_without_prev_block_gate():
    sampler = FastdLLMV2Sampler()
    block = SimpleNamespace(
        thresholds=SimpleNamespace(accept_threshold=1.0),
        prev_block=SimpleNamespace(is_semi_complete=False),
    )
    block.req = SimpleNamespace(fdv2_current_sub_block=block)
    confidence = torch.tensor([0.1, 0.7, 0.2])
    initial_confidence = torch.tensor([0.1, 0.7, 0.2])
    sampled_tokens = torch.tensor([10, 11, 12])

    accepted = sampler._compute_accepted_ids(block, confidence, initial_confidence, sampled_tokens)

    assert accepted.tolist() == [1]


def test_fast_dllm_v2_plain_multibd_sampler_uses_prev_block_gate():
    sampler = FastdLLMV2Sampler()
    block = SimpleNamespace(
        thresholds=SimpleNamespace(accept_threshold=1.0),
        prev_block=SimpleNamespace(is_semi_complete=False),
    )
    confidence = torch.tensor([0.1, 0.7, 0.2])
    initial_confidence = torch.tensor([0.1, 0.7, 0.2])
    sampled_tokens = torch.tensor([10, 11, 12])

    accepted = sampler._compute_accepted_ids(block, confidence, initial_confidence, sampled_tokens)

    assert accepted.tolist() == []


class _FakeGraph:
    def __init__(self, outputs, value):
        self.outputs = outputs
        self.value = value
        self.replayed = False

    def replay(self):
        self.outputs.fill_(self.value)
        self.replayed = True


class _FakeModel:
    @staticmethod
    def compute_logits(hidden_states):
        return hidden_states + 2


def _fake_graph_runner():
    runner = FastDLLMV2ModelRunner.__new__(FastDLLMV2ModelRunner)
    runner.config = SimpleNamespace(
        block_size=8,
        buffer_size=4,
        enable_full_static_runner=True,
    )
    runner.enforce_eager = False
    runner.fdv2_attention_block_size = 32
    runner.model = _FakeModel()
    outputs = torch.zeros(96, 3)
    runner.graph_vars = {
        "input_ids": torch.zeros(96, dtype=torch.int64),
        "positions": torch.zeros(96, dtype=torch.int64),
        "slot_mapping": torch.full((96,), -1, dtype=torch.int32),
        "context_lens": torch.zeros(8, dtype=torch.int32),
        "cu_seqlens_q": torch.zeros(9, dtype=torch.int32),
        "cu_seqlens_k": torch.zeros(9, dtype=torch.int32),
        "valid_slices": torch.zeros(8, dtype=torch.int32),
        "status_table": torch.zeros(8, dtype=torch.int32),
        "prefix_lens": torch.zeros(8, dtype=torch.int32),
        "padded_prefix_lens": torch.zeros(8, dtype=torch.int32),
        "page_tables": torch.full((8, 4), -1, dtype=torch.int32),
        "outputs": outputs,
    }
    runner.graph_bs = {
        "full_buffer_init": [32, 64, 96],
        "sub_block_cache_only": [8, 16, 32],
        "final_commit": [32, 64, 96],
    }
    runner.graphs = {
        ("full_buffer_init", 64): _FakeGraph(outputs, 5),
        ("sub_block_cache_only", 32): _FakeGraph(outputs, 7),
        ("final_commit", 64): _FakeGraph(outputs, 11),
    }
    runner.graph_outputs_are_logits = False
    return runner


def _fake_attn_metadata(*, cache_only: bool, num_reqs: int, q_len: int, mode: FastDLLMV2Mode | None = None):
    if mode is None:
        mode = FastDLLMV2Mode.SUB_BLOCK_REFINE if cache_only else FastDLLMV2Mode.FULL_BUFFER_INIT
    return SimpleNamespace(
        has_prefill=False,
        fdv2_cache_only=cache_only,
        fdv2_mode=int(mode),
        block_size=32,
        num_reqs=num_reqs,
        slot_mapping=torch.arange(num_reqs * q_len, dtype=torch.int32),
        context_lens=torch.arange(num_reqs, dtype=torch.int32) + 32,
        cu_seqlens_q=torch.arange(num_reqs + 1, dtype=torch.int32) * q_len,
        cu_seqlens_k=torch.arange(num_reqs + 1, dtype=torch.int32) * 64,
        valid_slices=(torch.arange(num_reqs, dtype=torch.int32) + 1) * q_len,
        status_table=torch.full((num_reqs,), 2 if cache_only else 1, dtype=torch.int32),
        prefix_lens=torch.zeros(num_reqs, dtype=torch.int32),
        padded_prefix_lens=torch.zeros(num_reqs, dtype=torch.int32),
        page_tables=torch.zeros(num_reqs, 2, dtype=torch.int32),
    )


def test_fast_dllm_v2_cuda_graph_replay_uses_sub_block_bucket_and_static_metadata():
    runner = _fake_graph_runner()
    attn_metadata = _fake_attn_metadata(cache_only=True, num_reqs=3, q_len=8)
    input_ids = torch.arange(24, dtype=torch.int64)
    positions = torch.arange(128, 152, dtype=torch.int64)

    assert runner._fdv2_can_run_decode_graph(input_ids, attn_metadata)

    logits = runner._fdv2_run_decode_graph(input_ids, positions, attn_metadata)

    assert runner.graphs[("sub_block_cache_only", 32)].replayed is True
    assert runner.graphs[("full_buffer_init", 64)].replayed is False
    assert runner.graphs[("final_commit", 64)].replayed is False
    assert logits.shape == (24, 3)
    assert torch.all(logits == 9)
    assert torch.equal(runner.graph_vars["input_ids"][:24], input_ids)
    assert torch.equal(runner.graph_vars["positions"][:24], positions)
    assert runner.graph_vars["cu_seqlens_q"][4] == runner.graph_vars["cu_seqlens_q"][3]
    assert runner.graph_vars["cu_seqlens_k"][4] == runner.graph_vars["cu_seqlens_k"][3]
    assert attn_metadata.fdv2_cache_only is True
    assert attn_metadata.block_size == 32


def test_fast_dllm_v2_cuda_graph_replay_uses_full_buffer_bucket():
    runner = _fake_graph_runner()
    attn_metadata = _fake_attn_metadata(cache_only=False, num_reqs=2, q_len=32)
    input_ids = torch.arange(64, dtype=torch.int64)
    positions = torch.arange(64, dtype=torch.int64)

    assert runner._fdv2_can_run_decode_graph(input_ids, attn_metadata)

    logits = runner._fdv2_run_decode_graph(input_ids, positions, attn_metadata)

    assert runner.graphs[("full_buffer_init", 64)].replayed is True
    assert runner.graphs[("sub_block_cache_only", 32)].replayed is False
    assert runner.graphs[("final_commit", 64)].replayed is False
    assert logits.shape == (64, 3)
    assert torch.all(logits == 7)
    assert attn_metadata.fdv2_cache_only is False
    assert attn_metadata.block_size == 32


def test_fast_dllm_v2_cuda_graph_replay_uses_final_commit_bucket():
    runner = _fake_graph_runner()
    attn_metadata = _fake_attn_metadata(
        cache_only=False,
        mode=FastDLLMV2Mode.FINAL_COMMIT,
        num_reqs=2,
        q_len=32,
    )
    input_ids = torch.arange(64, dtype=torch.int64)
    positions = torch.arange(64, dtype=torch.int64)

    assert runner._fdv2_can_run_decode_graph(input_ids, attn_metadata)

    logits = runner._fdv2_run_decode_graph(input_ids, positions, attn_metadata)

    assert runner.graphs[("final_commit", 64)].replayed is True
    assert runner.graphs[("full_buffer_init", 64)].replayed is False
    assert runner.graphs[("sub_block_cache_only", 32)].replayed is False
    assert logits.shape == (64, 3)
    assert torch.all(logits == 13)
    assert attn_metadata.fdv2_cache_only is False
    assert attn_metadata.fdv2_mode == int(FastDLLMV2Mode.FINAL_COMMIT)


def test_fast_dllm_v2_cuda_graph_uses_exact_batch_buckets():
    assert FastDLLMV2ModelRunner._graph_seq_batch_sizes(0) == []
    assert FastDLLMV2ModelRunner._graph_seq_batch_sizes(5) == [1, 2, 3, 4, 5]
