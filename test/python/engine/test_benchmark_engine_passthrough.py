import argparse

from types import SimpleNamespace

from lm_eval import utils
from lm_eval._cli.utils import MergeDictAction

from diffulex_bench.config import BenchmarkConfig, EngineConfig
from diffulex import SamplingParams
from diffulex_bench.lm_eval_model import DiffulexLM, _sampling_params_from_generation_kwargs
from diffulex_bench.main import (
    _decode_lm_eval_model_arg_dict,
    _install_lm_eval_model_arg_decoder,
    config_to_model_args,
)
from diffulex_bench.runner import BenchmarkRunner


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
            "enable_vectorized_sampler": True,
            "enable_vectorized_sampler_compile": True,
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
    assert kwargs["enable_vectorized_sampler"] is True
    assert kwargs["enable_vectorized_sampler_compile"] is True
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
                "expert_parallel_size": 2,
                "attn_impl": "naive",
                "enable_prefix_caching": False,
                "token_merge_mode": "iter_smooth_topk",
                "token_merge_top_k": 3,
                "token_merge_renormalize": False,
                "token_merge_weight": 0.75,
                "dmax_sampler_fast_path": False,
                "dmax_force_prefill_active": True,
                "enable_vectorized_sampler": True,
                "enable_vectorized_sampler_compile": True,
                "skip_warmup": True,
                "profiler_config": {
                    "profiler": "torch",
                    "torch_profiler_dir": "/tmp/diffulex-profile",
                    "max_iterations": 3,
                },
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
    assert forwarded["expert_parallel_size"] == 2
    assert forwarded["attn_impl"] == "naive"
    assert forwarded["enable_prefix_caching"] is False
    assert forwarded["token_merge_mode"] == "iter_smooth_topk"
    assert forwarded["token_merge_top_k"] == 3
    assert forwarded["token_merge_renormalize"] is False
    assert forwarded["token_merge_weight"] == 0.75
    assert forwarded["dmax_sampler_fast_path"] is False
    assert forwarded["dmax_force_prefill_active"] is True
    assert forwarded["enable_vectorized_sampler"] is True
    assert forwarded["enable_vectorized_sampler_compile"] is True
    assert forwarded["skip_warmup"] is True
    assert forwarded["profiler_config"]["torch_profiler_dir"] == "/tmp/diffulex-profile"
    assert forwarded["profiler_config"]["max_iterations"] == 3
    assert forwarded["block_size"] == 4
    assert forwarded["decoding_thresholds"]["accept_threshold"] == 0.95
    assert lm.max_new_tokens == 123
    assert lm.max_nfe == 17
    assert lm.max_repetition_run == 9
    assert lm.sampling_params.max_nfe == 17
    assert lm.sampling_params.max_repetition_run == 9


def test_benchmark_runner_starts_profile_once(monkeypatch) -> None:
    class FakeLLM:
        def __init__(self, model_path, **kwargs):
            self.model_path = model_path
            self.kwargs = kwargs
            self.profile_starts = []

        def start_profile(self, profile_prefix=None):
            self.profile_starts.append(profile_prefix)

        def generate(self, prompts, sampling_params, use_tqdm=True):
            return [{"token_ids": [1, 2], "text": "ok"} for _ in prompts]

    captured = {}

    def fake_diffulex(model_path, **kwargs):
        captured["llm"] = FakeLLM(model_path, **kwargs)
        return captured["llm"]

    monkeypatch.setattr("diffulex_bench.runner.Diffulex", fake_diffulex)
    monkeypatch.setattr(
        "diffulex_bench.runner.auto_tokenizer_from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(name_or_path="fake-tokenizer"),
    )

    runner = BenchmarkRunner(
        "/tmp/model",
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": "/tmp/profile",
            "run_id": "bench-run",
        },
    )

    runner.generate(["a"], SimpleNamespace(), use_tqdm=False)
    runner.generate(["b"], SimpleNamespace(), use_tqdm=False)

    assert captured["llm"].profile_starts == ["bench-run"]


def test_lm_eval_generation_kwargs_override_sampling_params() -> None:
    default = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        max_nfe=None,
        max_repetition_run=32,
        ignore_eos=False,
    )

    params = _sampling_params_from_generation_kwargs(
        default,
        {
            "max_gen_toks": 8192,
            "temperature": 0.7,
            "ignore_eos": "true",
        },
    )

    assert params.max_tokens == 8192
    assert params.temperature == 0.7
    assert params.max_repetition_run == 32
    assert params.ignore_eos is True


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
                "expert_parallel_size": 2,
                "enable_prefix_caching": False,
                "token_merge_mode": "iter_smooth_topk",
                "token_merge_top_k": 3,
                "token_merge_renormalize": False,
                "token_merge_weight": 0.75,
                "enable_vectorized_sampler": True,
                "enable_vectorized_sampler_compile": False,
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
    assert forwarded["expert_parallel_size"] == 2
    assert forwarded["enable_prefix_caching"] is False
    assert forwarded["token_merge_mode"] == "iter_smooth_topk"
    assert forwarded["token_merge_top_k"] == 3
    assert forwarded["token_merge_renormalize"] is False
    assert forwarded["token_merge_weight"] == 0.75
    assert forwarded["enable_vectorized_sampler"] is True
    assert forwarded["enable_vectorized_sampler_compile"] is False
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
