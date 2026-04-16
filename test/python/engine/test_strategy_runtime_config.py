from pathlib import Path

import yaml

import pytest

from diffulex.config import Config
from diffulex_bench.arg_parser import create_argument_parser
from diffulex_bench.main import load_config_from_args


MODEL_PATH = "/data1/ckpts/JetLM/SDAR-1.7B-Chat-b32"


def test_runtime_forces_d2f_prefix_settings():
    cfg = Config(
        MODEL_PATH,
        decoding_strategy="d2f",
        enable_prefix_caching=True,
        multi_block_prefix_full=False,
        tensor_parallel_size=1,
        data_parallel_size=1,
    )

    assert cfg.multi_block_prefix_full is True
    assert cfg.enable_prefix_caching is False


def test_runtime_forces_multi_bd_prefix_full_off():
    cfg = Config(
        MODEL_PATH,
        decoding_strategy="multi_bd",
        enable_prefix_caching=True,
        multi_block_prefix_full=True,
        tensor_parallel_size=1,
        data_parallel_size=1,
    )

    assert cfg.multi_block_prefix_full is False
    assert cfg.enable_prefix_caching is True


def test_config_file_dataset_not_overridden_by_cli_defaults(tmp_path):
    config_path = Path(tmp_path) / "bench.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "engine": {
                    "model_path": MODEL_PATH,
                    "model_name": "sdar",
                    "decoding_strategy": "multi_bd",
                    "mask_token_id": 151669,
                    "tensor_parallel_size": 1,
                    "data_parallel_size": 1,
                },
                "eval": {
                    "dataset_name": "math500_diffulex_4shot",
                    "output_dir": "custom_results",
                    "max_nfe": 77,
                },
            }
        ),
        encoding="utf-8",
    )

    parser = create_argument_parser()
    args = parser.parse_args(["--config", str(config_path)])
    config = load_config_from_args(args)

    assert config.eval.dataset_name == "math500_diffulex_4shot"
    assert config.eval.output_dir == "custom_results"
    assert config.eval.max_nfe == 77


def test_runtime_builds_default_decoding_thresholds_when_flat_keys_are_none():
    cfg = Config(
        MODEL_PATH,
        decoding_strategy="multi_bd",
        decoding_thresholds=None,
        add_block_threshold=None,
        semi_complete_threshold=None,
        accept_threshold=None,
        tensor_parallel_size=1,
        data_parallel_size=1,
    )

    assert cfg.decoding_thresholds.add_block_threshold == 0.1
    assert cfg.decoding_thresholds.semi_complete_threshold == 0.9
    assert cfg.decoding_thresholds.accept_threshold == 0.9


def test_bench_cli_forwards_sampling_mode_to_engine_config():
    parser = create_argument_parser()
    args = parser.parse_args(
        [
            "--model-path",
            MODEL_PATH,
            "--sampling-mode",
            "edit",
        ]
    )

    config = load_config_from_args(args)

    assert config.engine.sampling_mode == "edit"


def test_bench_cli_accepts_llada_and_dmax_choices():
    parser = create_argument_parser()
    args = parser.parse_args(
        [
            "--model-path",
            MODEL_PATH,
            "--model-name",
            "llada",
            "--decoding-strategy",
            "dmax",
            "--sampling-mode",
            "edit",
        ]
    )

    config = load_config_from_args(args)

    assert config.engine.model_name == "llada"
    assert config.engine.decoding_strategy == "dmax"
    assert config.engine.sampling_mode == "edit"


def test_bench_cli_forwards_explicit_engine_fields():
    parser = create_argument_parser()
    args = parser.parse_args(
        [
            "--model-path",
            MODEL_PATH,
            "--expert-parallel-size",
            "2",
            "--page-size",
            "16",
            "--token-merge-mode",
            "iter_smooth_topk",
            "--token-merge-top-k",
            "3",
            "--no-token-merge-renormalize",
            "--token-merge-weight",
            "0.75",
            "--no-enable-prefix-caching",
        ]
    )

    config = load_config_from_args(args)

    assert config.engine.expert_parallel_size == 2
    assert config.engine.page_size == 16
    assert config.engine.token_merge_mode == "iter_smooth_topk"
    assert config.engine.token_merge_top_k == 3
    assert config.engine.token_merge_renormalize is False
    assert config.engine.token_merge_weight == 0.75
    assert config.engine.enable_prefix_caching is False


def test_config_rejects_edit_sampling_for_non_llada2_model(config_no_model_load):
    with pytest.raises(ValueError, match="sampling_mode='edit' is only supported"):
        Config(
            str(config_no_model_load),
            model_name="dream",
            sampling_mode="edit",
            tensor_parallel_size=1,
            data_parallel_size=1,
            device_ids=[0],
        )


def test_config_rejects_dmax_without_edit_sampling(config_no_model_load):
    with pytest.raises(ValueError, match="decoding_strategy='dmax' requires sampling_mode='edit'"):
        Config(
            str(config_no_model_load),
            model_name="llada2",
            decoding_strategy="dmax",
            sampling_mode="naive",
            tensor_parallel_size=1,
            data_parallel_size=1,
            device_ids=[0],
        )


def test_config_rejects_llada2dot1_mini_without_edit_sampling(config_no_model_load):
    with pytest.raises(ValueError, match="model_name='llada2dot1_mini' requires sampling_mode='edit'"):
        Config(
            str(config_no_model_load),
            model_name="llada2dot1_mini",
            sampling_mode="naive",
            tensor_parallel_size=1,
            data_parallel_size=1,
            device_ids=[0],
        )


def test_config_accepts_sdar_naive_sampling(config_no_model_load):
    cfg = Config(
        str(config_no_model_load),
        model_name="sdar",
        sampling_mode="naive",
        tensor_parallel_size=1,
        data_parallel_size=1,
        device_ids=[0],
    )

    assert cfg.model_name == "sdar"
    assert cfg.sampling_mode == "naive"


def test_config_accepts_fast_dllm_v2_naive_sampling(config_no_model_load):
    cfg = Config(
        str(config_no_model_load),
        model_name="fast_dllm_v2",
        sampling_mode="naive",
        tensor_parallel_size=1,
        data_parallel_size=1,
        device_ids=[0],
    )

    assert cfg.model_name == "fast_dllm_v2"
    assert cfg.sampling_mode == "naive"


def test_config_accepts_distinct_kv_cache_layout_independently(config_no_model_load):
    cfg = Config(
        str(config_no_model_load),
        model_name="sdar",
        kv_cache_layout="distinct",
        tensor_parallel_size=1,
        data_parallel_size=1,
        device_ids=[0],
    )

    assert cfg.model_name == "sdar"
    assert cfg.kv_cache_layout == "distinct"


@pytest.fixture
def config_no_model_load(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setattr("diffulex.config.AutoConfig.from_pretrained", lambda *args, **kwargs: type(
        "FakeHFConfig",
        (),
        {"max_position_embeddings": 4096},
    )())
    return model_dir


@pytest.mark.parametrize(
    "page_size,block_size",
    [(page_size, block_size) for block_size in (4, 8, 16, 32) for page_size in (4, 8, 16, 32) if block_size <= page_size],
)
def test_config_accepts_supported_page_block_matrix(config_no_model_load, page_size, block_size):
    cfg = Config(
        str(config_no_model_load),
        decoding_strategy="multi_bd",
        page_size=page_size,
        block_size=block_size,
        tensor_parallel_size=1,
        data_parallel_size=1,
        device_ids=[0],
    )

    assert cfg.page_size == page_size
    assert cfg.block_size == block_size


def test_config_rejects_block_larger_than_page(config_no_model_load):
    with pytest.raises(ValueError, match="block_size must be <= page_size"):
        Config(
            str(config_no_model_load),
            decoding_strategy="multi_bd",
            page_size=4,
            block_size=8,
            tensor_parallel_size=1,
            data_parallel_size=1,
            device_ids=[0],
        )


def test_config_rejects_unsupported_page_block_size(config_no_model_load):
    with pytest.raises(ValueError, match="page_size must be one of"):
        Config(
            str(config_no_model_load),
            decoding_strategy="multi_bd",
            page_size=64,
            block_size=32,
            tensor_parallel_size=1,
            data_parallel_size=1,
            device_ids=[0],
        )
