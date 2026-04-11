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
        decoding_threshold=None,
        tensor_parallel_size=1,
        data_parallel_size=1,
    )

    assert cfg.decoding_thresholds.add_block_threshold == 0.1
    assert cfg.decoding_thresholds.semi_complete_threshold == 0.9
    assert cfg.decoding_thresholds.decoding_threshold == 0.9


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
