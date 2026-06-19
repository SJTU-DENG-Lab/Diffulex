from __future__ import annotations

from diffulex.server.args import parse_args


def test_server_args_forward_recent_engine_config_fields():
    args = parse_args(
        [
            "--model",
            "/tmp/model",
            "--disable-prefill-cudagraph",
            "--prefill-cudagraph-max-len",
            "4096",
            "--disable-torch-compile",
            "--enable-cudagraph-torch-compile",
            "--torch-compile-mode",
            "max-autotune",
            "--auto-max-nfe-warmup-steps",
            "6",
            "--auto-max-nfe-tpf-floor",
            "1.5",
            "--attn-impl",
            "naive",
            "--moe-gemm-impl",
            "vllm",
            "--profiler",
            "torch",
            "--torch-profiler-dir",
            "/tmp/diffulex-profile",
            "--profiler-record-shapes",
            "--profiler-max-iterations",
            "3",
        ]
    )

    kwargs = args.engine_kwargs()

    assert kwargs["enable_prefill_cudagraph"] is False
    assert kwargs["prefill_cudagraph_max_len"] == 4096
    assert kwargs["enable_torch_compile"] is False
    assert kwargs["enable_cudagraph_torch_compile"] is True
    assert kwargs["torch_compile_mode"] == "max-autotune"
    assert kwargs["auto_max_nfe_warmup_steps"] == 6
    assert kwargs["auto_max_nfe_tpf_floor"] == 1.5
    assert kwargs["attn_impl"] == "naive"
    assert kwargs["moe_gemm_impl"] == "vllm"
    assert kwargs["profiler_config"] == {
        "profiler": "torch",
        "torch_profiler_dir": "/tmp/diffulex-profile",
        "record_shapes": True,
        "profile_memory": False,
        "with_stack": False,
        "delay_iterations": 0,
        "max_iterations": 3,
    }
