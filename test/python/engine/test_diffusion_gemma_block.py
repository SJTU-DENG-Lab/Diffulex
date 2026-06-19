from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

import diffulex.distributed.parallel_state as parallel_state
from diffulex.config import Config
from diffulex.engine.status import DllmReqStatus
from diffulex.model.diffusion_gemma import DiffusionGemmaForDiffusionLM
from diffulex.sampler.diffusion_gemma import DiffusionGemmaSampler
from diffulex.sampling_params import SamplingParams
from diffulex.strategy.diffusion_gemma.engine.model_runner import DiffusionGemmaModelRunner
from diffulex.strategy.diffusion_gemma.engine.request import DiffusionGemmaReq
from diffulex.utils.loader import apply_resolved_weight, resolve_weight_spec


@pytest.fixture
def config_no_model_load(monkeypatch, tmp_path):
    model_dir = tmp_path / "fake_model"
    model_dir.mkdir()
    del monkeypatch
    return model_dir


def _write_diffusion_gemma_config(model_dir):
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "diffusion_gemma",
                "canvas_length": 256,
                "text_config": {
                    "vocab_size": 32,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "head_dim": 8,
                    "max_position_embeddings": 128,
                    "rms_norm_eps": 1e-6,
                },
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "generation_config.json").write_text(
        json.dumps(
            {
                "max_denoising_steps": 7,
                "stability_threshold": 3,
                "t_min": 0.2,
                "t_max": 0.8,
                "confidence_threshold": 0.4,
                "sampler_config": {
                    "_cls_name": "EntropyBound",
                    "entropy_bound": 2.5,
                },
            }
        ),
        encoding="utf-8",
    )


def test_diffusion_gemma_forces_gemma_block_runtime(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_diffusion_gemma_config(model_dir)

    cfg = Config(
        str(model_dir),
        model_name="diffusion_gemma",
        decoding_strategy="d2f",
        block_size=32,
        page_size=32,
        buffer_size=4,
        tensor_parallel_size=1,
        data_parallel_size=1,
        device_ids=[0],
        max_num_batched_tokens=256,
    )

    assert cfg.decoding_strategy == "diffusion_gemma"
    assert cfg.block_size == 256
    assert cfg.page_size == 256
    assert cfg.buffer_size == 1
    assert cfg.diffusion_gemma_max_denoising_steps == 7
    assert cfg.diffusion_gemma_stability_threshold == 3
    assert cfg.diffusion_gemma_entropy_bound == 2.5
    assert cfg.hf_config.hidden_size == 8


def test_non_gemma_rejects_page_size_256(config_no_model_load):
    with pytest.raises(ValueError, match="page_size must be one of"):
        Config(
            str(config_no_model_load),
            model_name="sdar",
            decoding_strategy="multi_bd",
            page_size=256,
            block_size=32,
            tensor_parallel_size=1,
            data_parallel_size=1,
            device_ids=[0],
        )


def test_gemma_block_prefill_uses_real_prefix_only(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_diffusion_gemma_config(model_dir)
    cfg = Config(
        str(model_dir),
        model_name="diffusion_gemma",
        tensor_parallel_size=1,
        data_parallel_size=1,
        device_ids=[0],
        max_num_batched_tokens=256,
    )

    req = DiffusionGemmaReq([11, 12, 13], SamplingParams(max_tokens=5))
    req.page_size = cfg.page_size
    req.init_multi_block(cfg)
    req.page_table = [7, 8]
    req.make_pending()
    req.step()

    runner = SimpleNamespace(
        page_size=cfg.page_size,
        _cached_prefix_len=lambda request: request.contiguous_in_cache_prefix_len,
    )
    prepared = DiffusionGemmaModelRunner._prepare_prefill_req(runner, req)

    assert req.prefix_len == 3
    assert req.padded_prefix_len == 256
    assert req.dllm_block_buffer.first_running_block.start == 256
    assert prepared["input_ids"] == [11, 12, 13]
    assert prepared["positions"] == [0, 1, 2]
    assert prepared["slot_mapping"] == [7 * 256, 7 * 256 + 1, 7 * 256 + 2]
    assert prepared["valid_slice"] == 3


def test_diffusion_gemma_rewrite_hook_defers_token_count_to_commit(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_diffusion_gemma_config(model_dir)
    cfg = Config(
        str(model_dir),
        model_name="diffusion_gemma",
        tensor_parallel_size=1,
        data_parallel_size=1,
        device_ids=[0],
        max_num_batched_tokens=256,
    )
    req = DiffusionGemmaReq([11, 12, 13], SamplingParams(max_tokens=5))
    req.page_size = cfg.page_size
    req.init_multi_block(cfg)
    block = req.dllm_block_buffer.active_blocks[0]

    req.on_block_token_rewrite(block, rel_idx=0, old_token=block.mask_token_id, new_token=7)

    assert req.new_tokens == 0


def test_diffusion_gemma_model_constructs_k_eq_v_attention():
    parallel_state.reset_parallel_state()
    parallel_state.PARALLEL_STATE = parallel_state.build_parallel_state_for_test(
        tp_size=1,
        ep_size=1,
        dp_size=1,
        world_size=1,
        global_rank=0,
    )
    text_config = SimpleNamespace(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        attention_bias=False,
        layer_types=["full_attention"],
        rope_parameters={
            "full_attention": {
                "rope_theta": 1000000.0,
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
            }
        },
        attention_k_eq_v=True,
        router_uses_prenormed_input=False,
        tie_word_embeddings=False,
        final_logit_softcapping=None,
        hidden_activation="gelu_pytorch_tanh",
    )
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            text_config=text_config,
            self_conditioning_size=32,
        )
    )

    model = DiffusionGemmaForDiffusionLM(config)
    attn = model.model.layers[0].self_attn

    assert attn.v_proj is None
    assert attn.num_heads == 2
    assert attn.num_kv_heads == 2
    assert attn.head_dim == 8
    assert attn.attn.scale == 1.0
    assert type(attn.rotary_emb).__name__ == "Gemma4ProportionalRotaryEmbedding"
    assert attn.rotary_emb.rope_angles == 1
    assert attn.rotary_emb.nope_angles == 3


def test_diffusion_gemma_resolves_fused_moe_expert_weights():
    parallel_state.reset_parallel_state()
    parallel_state.PARALLEL_STATE = parallel_state.build_parallel_state_for_test(
        tp_size=1,
        ep_size=1,
        dp_size=1,
        world_size=1,
        global_rank=0,
    )
    text_config = SimpleNamespace(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=4,
        num_experts=2,
        top_k_experts=1,
        enable_moe_block=True,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        attention_bias=False,
        layer_types=["sliding_attention"],
        attention_k_eq_v=True,
        router_uses_prenormed_input=False,
        tie_word_embeddings=False,
        final_logit_softcapping=None,
        hidden_activation="gelu_pytorch_tanh",
    )
    model = DiffusionGemmaForDiffusionLM(
        SimpleNamespace(
            hf_config=SimpleNamespace(
                text_config=text_config,
                self_conditioning_size=32,
            )
        )
    )

    weight_name = "model.decoder.layers.0.experts.gate_up_proj"
    spec = resolve_weight_spec(model, weight_name, config=SimpleNamespace())
    weight = torch.arange(2 * 8 * 16, dtype=model.model.layers[0].moe.w13.dtype).view(2, 8, 16)

    assert spec is not None
    assert spec.loader is not None
    apply_resolved_weight(spec, weight)
    assert torch.equal(model.model.layers[0].moe.w13, weight)


def test_diffusion_gemma_sampler_builds_self_conditioning_embeds():
    sampler = DiffusionGemmaSampler(
        SimpleNamespace(
            diffusion_gemma_max_denoising_steps=2,
            diffusion_gemma_stability_threshold=1,
            diffusion_gemma_t_min=0.0,
            diffusion_gemma_t_max=0.0,
            diffusion_gemma_confidence_threshold=999.0,
            diffusion_gemma_entropy_bound=1.0,
            tokenizer_vocab_size=4,
            hf_config=SimpleNamespace(vocab_size=4),
        )
    )
    sampler.bind_model(
        SimpleNamespace(
            model=SimpleNamespace(
                embed_tokens=SimpleNamespace(weight=torch.arange(12, dtype=torch.float32).view(4, 3)),
                normalizer=torch.tensor(2.0),
            )
        )
    )

    probs = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    embeds = sampler._compute_self_conditioning_embeds(probs, valid_commit_len=2, block_size=3)

    assert embeds.shape == (3, 3)
    assert torch.equal(embeds[0], torch.tensor([0.0, 2.0, 4.0]))
    assert torch.equal(embeds[1], torch.tensor([6.0, 8.0, 10.0]))
    assert torch.equal(embeds[2], torch.zeros(3))


def test_diffusion_gemma_runner_sets_self_conditioning_context(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_diffusion_gemma_config(model_dir)
    cfg = Config(
        str(model_dir),
        model_name="diffusion_gemma",
        tensor_parallel_size=1,
        data_parallel_size=1,
        device_ids=[0],
        max_num_batched_tokens=256,
    )
    req = DiffusionGemmaReq([11, 12, 13], SamplingParams(max_tokens=5))
    req.page_size = cfg.page_size
    req.init_multi_block(cfg)
    req.make_pending()
    req.step()
    req.status = DllmReqStatus.DECODING
    active = req.dllm_block_buffer.active_blocks[0]
    soft = torch.ones(active.block_size, 8)

    captured = {}
    runner = SimpleNamespace(
        config=cfg,
        sampler=SimpleNamespace(
            get_self_conditioning_embeds=lambda req_id, block_id: soft
        ),
        model=SimpleNamespace(
            set_self_conditioning_context=lambda context: captured.setdefault("context", context)
        ),
    )

    DiffusionGemmaModelRunner._before_multi_block_model_forward(runner, [req])

    assert captured["context"][0]["start"] == 0
    assert captured["context"][0]["end"] == active.block_size
    assert captured["context"][0]["soft_embeds"] is soft
