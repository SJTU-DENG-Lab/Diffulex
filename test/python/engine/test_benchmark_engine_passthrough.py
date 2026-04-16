import argparse

from types import SimpleNamespace

from lm_eval import utils
from lm_eval._cli.utils import MergeDictAction

from diffulex_bench.config import BenchmarkConfig, EngineConfig
from diffulex_bench.lm_eval_model import DiffulexLM
from diffulex_bench.main import (
    _decode_lm_eval_model_arg_dict,
    _install_lm_eval_model_arg_decoder,
    config_to_model_args,
)


def test_engine_config_preserves_extra_diffulex_fields() -> None:
    engine = EngineConfig.from_dict(
        {
            "model_path": "/tmp/model",
            "model_name": "sdar",
            "decoding_strategy": "multi_bd",
            "block_size": 4,
            "buffer_size": 4,
            "page_size": 64,
            "device_ids": [0, 1],
            "k_cache_hdim_split_factor_x": 4,
            "decoding_thresholds": {
                "add_block_threshold": 0.2,
                "semi_complete_threshold": 0.8,
                "accept_threshold": 0.95,
            },
        }
    )

    kwargs = engine.get_diffulex_kwargs()

    assert kwargs["page_size"] == 64
    assert kwargs["device_ids"] == [0, 1]
    assert kwargs["k_cache_hdim_split_factor_x"] == 4
    assert kwargs["decoding_thresholds"]["semi_complete_threshold"] == 0.8


def test_model_args_round_trip_extra_engine_fields(monkeypatch) -> None:
    captured = {}

    class FakeRunner:
        def __init__(self, model_path, tokenizer_path=None, wait_ready=True, **diffulex_kwargs):
            captured["model_path"] = model_path
            captured["tokenizer_path"] = tokenizer_path
            captured["wait_ready"] = wait_ready
            captured["diffulex_kwargs"] = diffulex_kwargs
            self.tokenizer = SimpleNamespace(name_or_path="fake-tokenizer", bos_token=None)

    monkeypatch.setattr("diffulex_bench.lm_eval_model.BenchmarkRunner", FakeRunner)

    config = BenchmarkConfig.from_dict(
        {
            "engine": {
                "model_path": "/tmp/model",
                "model_name": "sdar",
                "decoding_strategy": "multi_bd",
                "block_size": 4,
                "buffer_size": 4,
                "page_size": 64,
                "device_ids": [0, 1],
                "decoding_thresholds": {
                    "add_block_threshold": 0.2,
                    "semi_complete_threshold": 0.8,
                    "accept_threshold": 0.95,
                },
            },
            "eval": {
                "dataset_name": "gsm8k",
                "max_tokens": 123,
                "max_nfe": 17,
                "max_repetition_run": 9,
                "temperature": 0.0,
            },
        }
    )

    model_args = config_to_model_args(config, result_output_dir="/tmp/out")
    lm = DiffulexLM.create_from_arg_string(model_args)

    forwarded = captured["diffulex_kwargs"]

    assert captured["model_path"] == "/tmp/model"
    assert captured["tokenizer_path"] == "/tmp/model"
    assert captured["wait_ready"] is True
    assert forwarded["page_size"] == 64
    assert forwarded["device_ids"] == [0, 1]
    assert forwarded["block_size"] == 4
    assert forwarded["decoding_thresholds"]["accept_threshold"] == 0.95
    assert lm.max_new_tokens == 123
    assert lm.max_nfe == 17
    assert lm.max_repetition_run == 9
    assert lm.sampling_params.max_nfe == 17
    assert lm.sampling_params.max_repetition_run == 9


def test_model_arg_obj_round_trip_extra_engine_fields(monkeypatch) -> None:
    captured = {}

    class FakeRunner:
        def __init__(self, model_path, tokenizer_path=None, wait_ready=True, **diffulex_kwargs):
            captured["model_path"] = model_path
            captured["tokenizer_path"] = tokenizer_path
            captured["wait_ready"] = wait_ready
            captured["diffulex_kwargs"] = diffulex_kwargs
            self.tokenizer = SimpleNamespace(name_or_path="fake-tokenizer", bos_token=None)

    monkeypatch.setattr("diffulex_bench.lm_eval_model.BenchmarkRunner", FakeRunner)

    config = BenchmarkConfig.from_dict(
        {
            "engine": {
                "model_path": "/tmp/model",
                "model_name": "sdar",
                "decoding_strategy": "multi_bd",
                "block_size": 4,
                "buffer_size": 4,
                "page_size": 64,
                "device_ids": [0, 1],
                "decoding_thresholds": {
                    "add_block_threshold": 0.2,
                    "semi_complete_threshold": 0.8,
                    "accept_threshold": 0.95,
                },
            },
            "eval": {
                "dataset_name": "gsm8k",
                "max_tokens": 123,
                "temperature": 0.0,
            },
        }
    )

    encoded_args = utils.simple_parse_args_string(
        config_to_model_args(config, result_output_dir="/tmp/out")
    )

    DiffulexLM.create_from_arg_obj(encoded_args)

    forwarded = captured["diffulex_kwargs"]

    assert captured["model_path"] == "/tmp/model"
    assert captured["tokenizer_path"] == "/tmp/model"
    assert captured["wait_ready"] is True
    assert forwarded["page_size"] == 64
    assert forwarded["device_ids"] == [0, 1]
    assert forwarded["decoding_thresholds"]["semi_complete_threshold"] == 0.8


def test_model_args_parser_decodes_complex_values_for_logging() -> None:
    config = BenchmarkConfig.from_dict(
        {
            "engine": {
                "model_path": "/tmp/model",
                "model_name": "sdar",
                "decoding_strategy": "multi_bd",
                "page_size": 64,
                "device_ids": [0, 1],
                "decoding_thresholds": {
                    "add_block_threshold": 0.2,
                    "semi_complete_threshold": 0.8,
                    "accept_threshold": 0.95,
                },
            },
            "eval": {
                "dataset_name": "gsm8k",
                "max_tokens": 16,
            },
        }
    )

    decoded = _decode_lm_eval_model_arg_dict(utils.simple_parse_args_string(config_to_model_args(config)))

    assert decoded["page_size"] == 64
    assert decoded["device_ids"] == [0, 1]
    assert decoded["decoding_thresholds"]["add_block_threshold"] == 0.2


def test_lm_eval_merge_dict_action_decodes_complex_values() -> None:
    _install_lm_eval_model_arg_decoder()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_args", nargs="+", action=MergeDictAction, default=None)

    ns = parser.parse_args(
        [
            "--model_args",
            "decoding_thresholds=b64json:eyJhZGRfYmxvY2tfdGhyZXNob2xkIjowLjEsInNlbWlfY29tcGxldGVfdGhyZXNob2xkIjowLjUsImFjY2VwdF90aHJlc2hvbGQiOjAuOTV9,page_size=32",
        ]
    )

    assert ns.model_args["page_size"] == 32
    assert ns.model_args["decoding_thresholds"]["semi_complete_threshold"] == 0.5
