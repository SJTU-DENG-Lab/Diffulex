from __future__ import annotations

from diffulex.server.args import parse_args


def test_server_args_forward_recent_engine_config_fields():
    args = parse_args(
        [
            "--model",
            "/tmp/model",
            "--auto-max-nfe-warmup-steps",
            "6",
            "--auto-max-nfe-tpf-floor",
            "1.5",
            "--attn-impl",
            "naive",
            "--moe-gemm-impl",
            "flashinfer",
            "--moe-topk-impl",
            "flashinfer",
        ]
    )

    kwargs = args.engine_kwargs()

    assert kwargs["auto_max_nfe_warmup_steps"] == 6
    assert kwargs["auto_max_nfe_tpf_floor"] == 1.5
    assert kwargs["attn_impl"] == "naive"
    assert kwargs["moe_gemm_impl"] == "flashinfer"
    assert kwargs["moe_topk_impl"] == "flashinfer"
