"""
Argument Parser - Command line argument parsing for benchmark
"""

import argparse
from pathlib import Path

MODEL_NAME_CHOICES = [
    "dream",
    "sdar",
    "sdar_moe",
    "fast_dllm_v2",
    "llada",
    "llada2",
    "llada2_moe",
    "llada2_mini",
    "llada2dot1_mini",
    "llada2_mini_dmax",
]

DECODING_STRATEGY_CHOICES = ["d2f", "multi_bd", "dmax"]
TOKEN_MERGE_MODE_CHOICES = ["dmax_topk", "iter_smooth_topk"]
ATTN_IMPL_CHOICES = ["triton", "naive"]
MOE_GEMM_IMPL_CHOICES = ["triton", "vllm", "naive"]
MOE_DISPATCHER_BACKEND_CHOICES = ["standard", "naive", "deepep"]
DEEP_EP_MODE_CHOICES = ["normal", "low_latency", "auto"]


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure argument parser for benchmark

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Diffulex Benchmark using lm-evaluation-harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using configuration file (recommended)
  python -m diffulex_bench.main --config diffulex_bench/configs/example.yml

  # Using command line arguments
  python -m diffulex_bench.main \\
    --model-path /path/to/model \\
    --dataset gsm8k \\
    --dataset-limit 100 \\
    --output-dir ./results

  # With custom model settings
  python -m diffulex_bench.main \\
    --model-path /path/to/model \\
    --model-name dream \\
    --decoding-strategy d2f \\
    --dataset gsm8k \\
    --temperature 0.0 \\
    --max-tokens 256
        """,
    )

    # Logging arguments
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (optional)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file path (YAML or JSON). Default: configs/example.yml",
    )

    # Model arguments
    parser.add_argument(
        "--model-path",
        type=str,
        help="Model path",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Tokenizer path (defaults to model-path)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="dream",
        choices=MODEL_NAME_CHOICES,
        help="Model name",
    )
    parser.add_argument(
        "--decoding-strategy",
        type=str,
        default="d2f",
        choices=DECODING_STRATEGY_CHOICES,
        help="Decoding strategy (d2f, multi_bd, dmax)",
    )
    parser.add_argument(
        "--sampling-mode",
        type=str,
        default=None,
        choices=["naive", "edit"],
        help="Sampler behavior mode",
    )
    parser.add_argument(
        "--mask-token-id",
        type=int,
        default=151666,
        help="Mask token ID",
    )

    # Inference arguments
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=1,
        help="Data parallel size",
    )
    parser.add_argument(
        "--expert-parallel-size",
        type=int,
        default=None,
        help="Expert parallel size",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum model length",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=4096,
        help="Maximum number of batched tokens",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="(Deprecated) Maximum number of sequences; use --max-num-reqs",
    )
    parser.add_argument(
        "--max-num-reqs",
        type=int,
        default=None,
        help="Maximum number of requests",
    )

    # Sampling arguments
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--max-nfe",
        type=int,
        default=None,
        help="Maximum number of forward evaluations (NFE) allowed per request",
    )
    parser.add_argument(
        "--max-repetition-run",
        type=int,
        default=None,
        help="Kill a request when its generated suffix ends with this many identical consecutive tokens",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Ignore EOS token",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k_diffulex",
        help="lm-eval task name (bundled offline: gsm8k_diffulex, math500_diffulex, humaneval_diffulex, ...)",
    )
    parser.add_argument(
        "--include-path",
        type=str,
        default=None,
        help="lm-eval --include_path for external tasks (default: packaged diffulex_bench/tasks). Set to empty to disable.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="test",
        help="Dataset split",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        help="Limit number of samples",
    )
    parser.add_argument(
        "--dataset-data-files",
        type=str,
        default=None,
        help="Override task YAML `dataset_kwargs.data_files` with this JSON path",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output base directory (each run may create a run_* subfolder; see --use-run-subdirectory)",
    )
    parser.add_argument(
        "--use-run-subdirectory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write this run under output_dir/run_<timestamp>_<task>/ (default: true; override YAML when set)",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        default=True,
        help="Save results to file",
    )
    parser.add_argument(
        "--no-save-results",
        dest="save_results",
        action="store_false",
        help="Do not save results to file",
    )

    # LoRA arguments
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="",
        help="LoRA path",
    )
    parser.add_argument(
        "--pre-merge-lora",
        action="store_true",
        dest="pre_merge_lora",
        help="Merge LoRA into base weights at load to avoid per-forward compute",
    )

    # Engine arguments
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Enforce eager mode (disable CUDA graphs)",
    )
    parser.add_argument(
        "--no-enforce-eager",
        dest="enforce_eager",
        action="store_false",
        help="Disable eager mode (enable CUDA graphs when supported)",
    )
    parser.set_defaults(enforce_eager=None)
    parser.add_argument(
        "--kv-cache-layout",
        type=str,
        default="unified",
        choices=["unified", "distinct"],
        help="KV cache layout",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to enable prefix caching",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=None,
        help="KV cache page size",
    )
    parser.add_argument(
        "--token-merge-mode",
        type=str,
        default=None,
        choices=TOKEN_MERGE_MODE_CHOICES,
        help="Token merge mode for DMax/token-merge strategies",
    )
    parser.add_argument(
        "--token-merge-top-k",
        type=int,
        default=None,
        help="Top-k count for token merge metadata",
    )
    parser.add_argument(
        "--token-merge-renormalize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to renormalize token merge probabilities",
    )
    parser.add_argument(
        "--token-merge-weight",
        type=float,
        default=None,
        help="Interpolation weight for token merge",
    )
    parser.add_argument(
        "--attn-impl",
        type=str,
        default=None,
        choices=ATTN_IMPL_CHOICES,
        help="Attention implementation",
    )
    parser.add_argument(
        "--moe-dispatcher-backend",
        type=str,
        default=None,
        choices=MOE_DISPATCHER_BACKEND_CHOICES,
        help="MoE token dispatcher backend",
    )
    parser.add_argument(
        "--moe-gemm-impl",
        type=str,
        default=None,
        choices=MOE_GEMM_IMPL_CHOICES,
        help="MoE GEMM implementation",
    )
    parser.add_argument(
        "--deepep-mode",
        type=str,
        default=None,
        choices=DEEP_EP_MODE_CHOICES,
        help="DeepEP dispatcher mode",
    )
    parser.add_argument(
        "--deepep-num-max-dispatch-tokens-per-rank",
        type=int,
        default=None,
        help="DeepEP max dispatch tokens per rank",
    )

    # D2F-specific arguments
    parser.add_argument(
        "--add-block-threshold",
        type=float,
        default=0.1,
        help="Add block threshold for D2F",
    )
    parser.add_argument(
        "--semi-complete-threshold",
        type=float,
        default=0.9,
        help="Semi-complete threshold for D2F",
    )
    parser.add_argument(
        "--accept-threshold",
        type=float,
        default=0.9,
        help="Accept threshold for D2F",
    )
    parser.add_argument(
        "--remask-threshold",
        type=float,
        default=0.4,
        help="Remask threshold for DMax-style edit decode",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        dest="block_size",
        help="Diffusion block size (aligned with diffulex Config.block_size, default 32)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="Number of active diffusion blocks in buffer",
    )
    parser.add_argument(
        "--enable-prefill-cudagraph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable lazy CUDA graph capture for block-aligned prefill buckets",
    )
    parser.add_argument(
        "--prefill-cudagraph-max-len",
        type=int,
        default=None,
        help="Maximum prefill bucket length to capture; 0 uses max_model_len",
    )
    parser.add_argument(
        "--enable-torch-compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable torch.compile where supported by the engine",
    )
    parser.add_argument(
        "--enable-cudagraph-torch-compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Experimental: allow torch.compile inside decode CUDA graph capture",
    )
    parser.add_argument(
        "--torch-compile-mode",
        type=str,
        default=None,
        help="torch.compile mode",
    )
    parser.add_argument(
        "--auto-max-nfe-warmup-steps",
        type=int,
        default=None,
        help="Warmup steps before deriving max_nfe from per-request average TPF when max_nfe is unset",
    )
    parser.add_argument(
        "--auto-max-nfe-tpf-floor",
        type=float,
        default=None,
        help="Minimum TPF used when deriving max_nfe from max_tokens",
    )
    parser.add_argument(
        "--multi-block-prefix-full",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether multi_bd should treat the prefix region as fully visible",
    )
    parser.add_argument(
        "--engine-arg",
        dest="engine_args",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Diffulex engine override. May be repeated; values are parsed with YAML semantics.",
    )
    return parser


def get_default_config_path() -> Path:
    """
    Get default configuration file path

    Returns:
        Path to default config file
    """
    config_dir = Path(__file__).parent / "configs"
    default_config = config_dir / "example.yml"
    return default_config
