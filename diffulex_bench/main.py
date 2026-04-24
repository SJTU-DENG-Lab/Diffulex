"""
Benchmark Main Entry - Main entry point for benchmark using lm-evaluation-harness
"""

import sys
import logging
import os
import time
import re
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from diffulex_bench.config import (
    BenchmarkConfig,
    EngineConfig,
    EvalConfig,
    decode_model_arg_value,
    encode_model_arg_value,
    parse_engine_arg_override,
)
from diffulex.logger import setup_logger, get_logger
from diffulex_bench.arg_parser import create_argument_parser, get_default_config_path

try:
    from lm_eval.__main__ import cli_evaluate
except ImportError:
    cli_evaluate = None


def _decode_lm_eval_model_arg_dict(args_dict: dict) -> dict:
    return {k: decode_model_arg_value(v) for k, v in args_dict.items()}


def _install_lm_eval_model_arg_decoder():
    """Patch lm-eval CLI parsing so encoded complex model_args are decoded before logging/init."""
    import lm_eval._cli.utils as lm_eval_cli_utils
    import lm_eval.config.evaluate_config as lm_eval_config
    import lm_eval.evaluator as lm_eval_evaluator
    import lm_eval.utils as lm_eval_utils

    original = getattr(lm_eval_utils, "_diffulex_orig_simple_parse_args_string", None)
    if original is None:
        original = lm_eval_utils.simple_parse_args_string
        lm_eval_utils._diffulex_orig_simple_parse_args_string = original

    def decoded_parse(args_string: str | None) -> dict:
        return _decode_lm_eval_model_arg_dict(original(args_string))

    lm_eval_utils.simple_parse_args_string = decoded_parse
    lm_eval_evaluator.simple_parse_args_string = decoded_parse
    lm_eval_config.simple_parse_args_string = decoded_parse

    original_key_val_to_dict = getattr(lm_eval_cli_utils, "_diffulex_orig_key_val_to_dict", None)
    if original_key_val_to_dict is None:
        original_key_val_to_dict = lm_eval_cli_utils.key_val_to_dict
        lm_eval_cli_utils._diffulex_orig_key_val_to_dict = original_key_val_to_dict

    def decoded_key_val_to_dict(args: str) -> dict:
        return _decode_lm_eval_model_arg_dict(original_key_val_to_dict(args))

    original_try_parse_json = getattr(lm_eval_cli_utils, "_diffulex_orig_try_parse_json", None)
    if original_try_parse_json is None:
        original_try_parse_json = lm_eval_cli_utils.try_parse_json
        lm_eval_cli_utils._diffulex_orig_try_parse_json = original_try_parse_json

    def decoded_try_parse_json(value):
        result = original_try_parse_json(value)
        if isinstance(result, dict):
            return _decode_lm_eval_model_arg_dict(result)
        return result

    lm_eval_cli_utils.key_val_to_dict = decoded_key_val_to_dict
    lm_eval_cli_utils.try_parse_json = decoded_try_parse_json

    evaluator_config_cls = lm_eval_config.EvaluatorConfig
    original_parse_dict_args = getattr(evaluator_config_cls, "_diffulex_orig_parse_dict_args", None)
    if original_parse_dict_args is None:
        original_parse_dict_args = evaluator_config_cls._parse_dict_args
        evaluator_config_cls._diffulex_orig_parse_dict_args = original_parse_dict_args

    def decoded_parse_dict_args(self):
        parsed = original_parse_dict_args(self)
        if getattr(parsed, "model_args", None) is not None:
            parsed.model_args = _decode_lm_eval_model_arg_dict(parsed.model_args)
        if getattr(parsed, "metadata", None) is not None:
            parsed.metadata = _decode_lm_eval_model_arg_dict(parsed.metadata)
        return parsed

    evaluator_config_cls._parse_dict_args = decoded_parse_dict_args
    return decoded_parse


def config_to_model_args(config: BenchmarkConfig, *, result_output_dir: Optional[str] = None) -> str:
    """
    Convert BenchmarkConfig to lm_eval model_args string format

    Args:
        config: Benchmark configuration
        result_output_dir: If set, used as model save_dir (trajectory/stats); else eval.output_dir

    Returns:
        Model arguments string in key=value format
    """
    engine = config.engine
    eval_config = config.eval
    save_dir = result_output_dir if result_output_dir is not None else eval_config.output_dir

    args_dict = {"pretrained": engine.model_path}
    args_dict.update(engine.get_diffulex_kwargs())
    args_dict = {
        **args_dict,
        "temperature": eval_config.temperature,
        "max_new_tokens": eval_config.max_tokens,
        "max_nfe": eval_config.max_nfe,
        "max_repetition_run": eval_config.max_repetition_run,
        "wait_ready": True,
    }

    if engine.tokenizer_path:
        args_dict["tokenizer_path"] = engine.tokenizer_path

    if save_dir and eval_config.save_results:
        args_dict["save_dir"] = save_dir

    if eval_config.add_bos_token is not None:
        args_dict["add_bos_token"] = eval_config.add_bos_token

    # Convert to string format: key1=value1,key2=value2
    args_list = []
    for k, v in args_dict.items():
        if v is None:
            continue
        args_list.append(f"{k}={encode_model_arg_value(v)}")
    return ",".join(args_list)


def _resolve_lm_eval_include_path(config: BenchmarkConfig) -> Optional[Path]:
    """
    lm-eval TaskManager include_path for bundled Lightning JSON tasks.
    None → diffulex_bench/tasks (sibling of this file). Empty string → disabled.
    """
    raw = config.eval.include_path
    if raw is not None and str(raw).strip() == "":
        return None
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = Path(os.getcwd()) / p
        return p.resolve()
    return (Path(__file__).resolve().parent / "tasks").resolve()


def _task_name_to_yaml_map(include_root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for yml in include_root.rglob("*.yaml"):
        try:
            text = yml.read_text(encoding="utf-8")
        except Exception:
            continue
        m = re.search(r"(?m)^\s*task:\s*([^\s#]+)\s*$", text)
        if m:
            mapping.setdefault(m.group(1).strip(), yml)
    return mapping


def _rewrite_task_data_files(task_yaml: Path, data_files: str) -> bool:
    text = task_yaml.read_text(encoding="utf-8")
    data_files_value = str(Path(data_files).expanduser())
    if Path(data_files_value).exists():
        data_files_value = str(Path(data_files_value).resolve())
    replacement_value = json.dumps(data_files_value)
    replaced, n = re.subn(r"(?m)^(\s*data_files:\s*).*$", rf"\1{replacement_value}", text, count=1)
    if n == 0:
        return False
    task_yaml.write_text(replaced, encoding="utf-8")
    return True


def _resolve_include_path_with_data_files_override(
    config: BenchmarkConfig, logger
) -> tuple[Optional[Path], Optional[Path]]:
    include_path = _resolve_lm_eval_include_path(config)
    data_files = config.eval.dataset_data_files
    if not data_files:
        return include_path, None
    if include_path is None or not include_path.is_dir():
        logger.warning(
            "dataset_data_files is set but include_path is unavailable; "
            "cannot rewrite task YAML data_files."
        )
        return include_path, None

    tmp_root = Path(tempfile.mkdtemp(prefix="diffulex_tasks_override_")).resolve()
    tmp_tasks = tmp_root / "tasks"
    shutil.copytree(include_path, tmp_tasks, dirs_exist_ok=True)
    task_map = _task_name_to_yaml_map(tmp_tasks)
    requested = [name.strip() for name in str(config.eval.dataset_name).split(",") if name.strip()]

    rewritten = 0
    for task_name in requested:
        task_yaml = task_map.get(task_name)
        if task_yaml is None:
            logger.warning(f"Task '{task_name}' not found under include_path={include_path}")
            continue
        if _rewrite_task_data_files(task_yaml, data_files):
            rewritten += 1
        else:
            logger.warning(f"Task '{task_name}' has no data_files field to override: {task_yaml}")

    if rewritten == 0:
        shutil.rmtree(tmp_root, ignore_errors=True)
        logger.warning("No task YAML was rewritten by dataset_data_files; using original include_path.")
        return include_path, None

    logger.info(f"Overrode dataset data_files for {rewritten} task(s) -> {data_files}")
    return tmp_tasks, tmp_root


def _sanitize_for_dir(name: str, max_len: int = 96) -> str:
    s = "".join(c if c.isalnum() or c in "._-" else "_" for c in name.strip())
    return s[:max_len] if s else "run"


def resolve_run_output_dir(config: BenchmarkConfig) -> str:
    """
    Root directory for this benchmark invocation: either output_dir or
    output_dir/run_<timestamp>_<task>/ when use_run_subdirectory is True.
    """
    base = Path(config.eval.output_dir).expanduser()
    if not config.eval.use_run_subdirectory:
        base.mkdir(parents=True, exist_ok=True)
        return str(base.resolve())
    task_part = _sanitize_for_dir(config.eval.dataset_name.replace(",", "+"))
    run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{task_part}"
    run_path = (base / run_name).resolve()
    run_path.mkdir(parents=True, exist_ok=True)
    return str(run_path)


def run_benchmark(config: BenchmarkConfig) -> None:
    """
    Run benchmark using lm-evaluation-harness

    Args:
        config: Benchmark configuration
    """
    logger = get_logger(__name__)

    if cli_evaluate is None:
        logger.error("lm-evaluation-harness is not installed. Please install it with: pip install lm-eval")
        sys.exit(1)
    decoded_model_arg_parser = _install_lm_eval_model_arg_decoder()

    benchmark_info = [
        "=" * 80,
        "Diffulex Benchmark (using lm-evaluation-harness)",
        "=" * 80,
        f"Model: {config.engine.model_path}",
        f"Model Name: {config.engine.model_name}",
        f"Decoding Strategy: {config.engine.decoding_strategy}",
        f"Tasks: {config.eval.dataset_name}",
        f"Output base directory: {config.eval.output_dir}",
        "=" * 80,
    ]
    run_output_dir = resolve_run_output_dir(config)
    benchmark_info.insert(-1, f"This run directory: {run_output_dir}")
    logger.info("\n".join(benchmark_info))

    # Convert config to lm_eval arguments (stats + trajectory share run_output_dir with lm-eval)
    model_args = config_to_model_args(config, result_output_dir=run_output_dir)
    decoded_model_args = decoded_model_arg_parser(model_args)
    tasks = config.eval.dataset_name

    # Prepare sys.argv for lm_eval
    original_argv = sys.argv.copy()

    # try:
    sys.argv = [
        "lm_eval",
        "--model",
        "diffulex",
        "--model_args",
        model_args,
        "--tasks",
        tasks,
        "--batch_size",
        "1",
        "--output_path",
        run_output_dir,
    ]

    inc, tmp_include_root = _resolve_include_path_with_data_files_override(config, logger)
    if inc is not None and inc.is_dir():
        sys.argv.extend(["--include_path", str(inc)])

    if config.eval.dataset_limit:
        sys.argv.extend(["--limit", str(config.eval.dataset_limit)])

    if config.eval.save_results:
        sys.argv.extend(["--log_samples"])

    # Add any additional lm_eval arguments from config if needed
    # For now, we use default batch_size=1

    lm_eval_info = [
        "=" * 80,
        "Starting lm-evaluation-harness evaluation...",
        "=" * 80,
        f"Model args: {decoded_model_args}",
        f"Tasks: {tasks}",
        "=" * 80,
    ]
    logger.info("\n".join(lm_eval_info))

    try:
        cli_evaluate()
        logger.success("Evaluation completed successfully")
    finally:
        sys.argv = original_argv
        if tmp_include_root is not None:
            shutil.rmtree(tmp_include_root, ignore_errors=True)

    # except Exception as e:
    #     logger.error(f"Evaluation failed: {e}", exc_info=True)
    #     sys.exit(1)
    # finally:
    #     # Restore original argv
    #     sys.argv = original_argv


def load_config_from_args(args) -> BenchmarkConfig:
    """
    Load configuration from command line arguments

    Args:
        args: Parsed command line arguments

    Returns:
        BenchmarkConfig instance
    """
    logger = get_logger(__name__)
    default_args = create_argument_parser().parse_args([])

    def was_provided(name: str) -> bool:
        return getattr(args, name) != getattr(default_args, name)

    if getattr(args, "max_num_reqs", None) is None and getattr(args, "max_num_seqs", None) is not None:
        logger.warning(
            "--max-num-seqs is deprecated and will be removed in a future release; please use --max-num-reqs instead."
        )
    max_num_reqs = (
        args.max_num_reqs if getattr(args, "max_num_reqs", None) is not None else getattr(args, "max_num_seqs", None)
    )
    engine_override_args = getattr(args, "engine_args", None) or []

    def apply_engine_arg_overrides(engine: EngineConfig) -> None:
        for raw in engine_override_args:
            if "=" not in raw:
                logger.error(f"Invalid --engine-arg '{raw}'. Expected KEY=VALUE.")
                sys.exit(1)
            key, raw_value = raw.split("=", 1)
            key = key.strip()
            if not key:
                logger.error(f"Invalid --engine-arg '{raw}'. Empty key.")
                sys.exit(1)
            engine.apply_updates({key: parse_engine_arg_override(raw_value)})

    # Try to load from config file
    if args.config:
        config_path = Path(args.config)
    else:
        # Try default config path
        default_config = get_default_config_path()
        if default_config.exists():
            config_path = default_config
            logger.info(f"Using default config: {config_path}")
        else:
            config_path = None

    if config_path and config_path.exists():
        if config_path.suffix in [".yaml", ".yml"]:
            config = BenchmarkConfig.from_yaml(str(config_path))
        elif config_path.suffix == ".json":
            config = BenchmarkConfig.from_json(str(config_path))
        else:
            logger.error(f"Unsupported config file format: {config_path.suffix}")
            sys.exit(1)
        logger.info(f"Loaded configuration from: {config_path}")

        # Override with command line arguments if provided
        if was_provided("model_path") and args.model_path:
            config.engine.model_path = args.model_path
        if was_provided("tokenizer_path") and getattr(args, "tokenizer_path", None):
            config.engine.tokenizer_path = args.tokenizer_path
        if was_provided("model_name") and getattr(args, "model_name", None):
            config.engine.model_name = args.model_name
        if was_provided("decoding_strategy") and getattr(args, "decoding_strategy", None):
            config.engine.decoding_strategy = args.decoding_strategy
        if was_provided("mask_token_id") and getattr(args, "mask_token_id", None) is not None:
            config.engine.mask_token_id = args.mask_token_id
        if was_provided("tensor_parallel_size") and getattr(args, "tensor_parallel_size", None) is not None:
            config.engine.tensor_parallel_size = args.tensor_parallel_size
        if was_provided("data_parallel_size") and getattr(args, "data_parallel_size", None) is not None:
            config.engine.data_parallel_size = args.data_parallel_size
        if was_provided("gpu_memory_utilization") and getattr(args, "gpu_memory_utilization", None) is not None:
            config.engine.gpu_memory_utilization = args.gpu_memory_utilization
        if was_provided("use_lora"):
            config.engine.use_lora = bool(args.use_lora)
        if was_provided("lora_path"):
            config.engine.lora_path = args.lora_path
        if was_provided("pre_merge_lora"):
            config.engine.pre_merge_lora = bool(args.pre_merge_lora)
        if was_provided("dataset") and args.dataset:
            config.eval.dataset_name = args.dataset
        if was_provided("dataset_limit") and args.dataset_limit is not None:
            config.eval.dataset_limit = args.dataset_limit
        if was_provided("max_tokens") and getattr(args, "max_tokens", None) is not None:
            config.eval.max_tokens = args.max_tokens
        if was_provided("max_nfe") and getattr(args, "max_nfe", None) is not None:
            config.eval.max_nfe = args.max_nfe
        if was_provided("max_repetition_run") and getattr(args, "max_repetition_run", None) is not None:
            config.eval.max_repetition_run = args.max_repetition_run
        if was_provided("temperature") and getattr(args, "temperature", None) is not None:
            config.eval.temperature = args.temperature
        if was_provided("output_dir") and args.output_dir:
            config.eval.output_dir = args.output_dir
        if getattr(args, "include_path", None) is not None:
            config.eval.include_path = args.include_path
        if getattr(args, "dataset_data_files", None) is not None:
            config.eval.dataset_data_files = args.dataset_data_files
        if getattr(args, "use_run_subdirectory", None) is not None:
            config.eval.use_run_subdirectory = bool(args.use_run_subdirectory)

        # Engine overrides (make bench configs reusable for eager vs CUDA Graph comparisons)
        if getattr(args, "enforce_eager", None) is not None:
            config.engine.enforce_eager = bool(args.enforce_eager)
        if was_provided("kv_cache_layout") and getattr(args, "kv_cache_layout", None) is not None:
            config.engine.kv_cache_layout = args.kv_cache_layout
        if getattr(args, "enable_prefix_caching", None) is not None:
            config.engine.enable_prefix_caching = bool(args.enable_prefix_caching)
        if getattr(args, "sampling_mode", None) is not None:
            config.engine.sampling_mode = args.sampling_mode
        if getattr(args, "expert_parallel_size", None) is not None:
            config.engine.expert_parallel_size = args.expert_parallel_size
        if was_provided("max_model_len") and getattr(args, "max_model_len", None) is not None:
            config.engine.max_model_len = args.max_model_len
        if max_num_reqs is not None:
            config.engine.max_num_reqs = max_num_reqs
        if (
            was_provided("max_num_batched_tokens")
            and getattr(args, "max_num_batched_tokens", None) is not None
        ):
            config.engine.max_num_batched_tokens = args.max_num_batched_tokens
        if getattr(args, "enable_prefill_cudagraph", None) is not None:
            config.engine.enable_prefill_cudagraph = bool(args.enable_prefill_cudagraph)
        if (
            was_provided("prefill_cudagraph_max_len")
            and getattr(args, "prefill_cudagraph_max_len", None) is not None
        ):
            config.engine.prefill_cudagraph_max_len = args.prefill_cudagraph_max_len
        if getattr(args, "enable_torch_compile", None) is not None:
            config.engine.enable_torch_compile = bool(args.enable_torch_compile)
        if getattr(args, "enable_cudagraph_torch_compile", None) is not None:
            config.engine.enable_cudagraph_torch_compile = bool(args.enable_cudagraph_torch_compile)
        if getattr(args, "torch_compile_mode", None) is not None:
            config.engine.torch_compile_mode = args.torch_compile_mode
        if getattr(args, "auto_max_nfe_warmup_steps", None) is not None:
            config.engine.auto_max_nfe_warmup_steps = args.auto_max_nfe_warmup_steps
        if getattr(args, "auto_max_nfe_tpf_floor", None) is not None:
            config.engine.auto_max_nfe_tpf_floor = args.auto_max_nfe_tpf_floor
        if getattr(args, "page_size", None) is not None:
            config.engine.page_size = args.page_size
        if getattr(args, "buffer_size", None) is not None:
            config.engine.buffer_size = args.buffer_size
        if getattr(args, "block_size", None) is not None:
            config.engine.block_size = args.block_size
        if getattr(args, "token_merge_mode", None) is not None:
            config.engine.token_merge_mode = args.token_merge_mode
        if getattr(args, "token_merge_top_k", None) is not None:
            config.engine.token_merge_top_k = args.token_merge_top_k
        if getattr(args, "token_merge_renormalize", None) is not None:
            config.engine.token_merge_renormalize = bool(args.token_merge_renormalize)
        if getattr(args, "token_merge_weight", None) is not None:
            config.engine.token_merge_weight = args.token_merge_weight
        if getattr(args, "attn_impl", None) is not None:
            config.engine.attn_impl = args.attn_impl
        if getattr(args, "moe_dispatcher_backend", None) is not None:
            config.engine.moe_dispatcher_backend = args.moe_dispatcher_backend
        if getattr(args, "moe_gemm_impl", None) is not None:
            config.engine.moe_gemm_impl = args.moe_gemm_impl
        if getattr(args, "deepep_mode", None) is not None:
            config.engine.deepep_mode = args.deepep_mode
        if getattr(args, "deepep_num_max_dispatch_tokens_per_rank", None) is not None:
            config.engine.deepep_num_max_dispatch_tokens_per_rank = args.deepep_num_max_dispatch_tokens_per_rank
        if getattr(args, "multi_block_prefix_full", None) is not None:
            config.engine.multi_block_prefix_full = bool(args.multi_block_prefix_full)
        apply_engine_arg_overrides(config.engine)
    else:
        if not args.model_path:
            logger.error("Either --config or --model-path must be provided")
            sys.exit(1)

        # Create config from command line arguments
        engine = EngineConfig(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            model_name=args.model_name,
            decoding_strategy=args.decoding_strategy,
            sampling_mode=getattr(args, "sampling_mode", None) or "naive",
            mask_token_id=args.mask_token_id,
            tensor_parallel_size=args.tensor_parallel_size,
            data_parallel_size=args.data_parallel_size,
            expert_parallel_size=(
                getattr(args, "expert_parallel_size", None)
                if getattr(args, "expert_parallel_size", None) is not None
                else 1
            ),
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=getattr(args, "max_num_batched_tokens", 4096),
            max_num_reqs=max_num_reqs if max_num_reqs is not None else 128,
            enable_prefill_cudagraph=(
                bool(getattr(args, "enable_prefill_cudagraph", True))
                if getattr(args, "enable_prefill_cudagraph", None) is not None
                else True
            ),
            prefill_cudagraph_max_len=(getattr(args, "prefill_cudagraph_max_len", None) or 0),
            enable_torch_compile=(
                bool(getattr(args, "enable_torch_compile", True))
                if getattr(args, "enable_torch_compile", None) is not None
                else True
            ),
            enable_cudagraph_torch_compile=bool(getattr(args, "enable_cudagraph_torch_compile", False)),
            torch_compile_mode=(getattr(args, "torch_compile_mode", None) or "reduce-overhead"),
            auto_max_nfe_warmup_steps=(getattr(args, "auto_max_nfe_warmup_steps", None) or 8),
            auto_max_nfe_tpf_floor=(getattr(args, "auto_max_nfe_tpf_floor", None) or 1.0),
            use_lora=args.use_lora,
            lora_path=args.lora_path,
            pre_merge_lora=getattr(args, "pre_merge_lora", True),
            enable_prefix_caching=(
                bool(args.enable_prefix_caching)
                if getattr(args, "enable_prefix_caching", None) is not None
                else True
            ),
            kv_cache_layout=getattr(args, "kv_cache_layout", "unified"),
            page_size=(args.page_size if getattr(args, "page_size", None) is not None else 32),
            token_merge_mode=(
                getattr(args, "token_merge_mode", None) or "dmax_topk"
            ),
            token_merge_top_k=(
                getattr(args, "token_merge_top_k", None)
                if getattr(args, "token_merge_top_k", None) is not None
                else 1
            ),
            token_merge_renormalize=(
                bool(args.token_merge_renormalize)
                if getattr(args, "token_merge_renormalize", None) is not None
                else True
            ),
            token_merge_weight=(
                getattr(args, "token_merge_weight", None)
                if getattr(args, "token_merge_weight", None) is not None
                else 1.0
            ),
            attn_impl=(getattr(args, "attn_impl", None) or "triton"),
            moe_dispatcher_backend=(getattr(args, "moe_dispatcher_backend", None) or "standard"),
            moe_gemm_impl=(getattr(args, "moe_gemm_impl", None) or "triton"),
            deepep_mode=(getattr(args, "deepep_mode", None) or "auto"),
            deepep_num_max_dispatch_tokens_per_rank=(
                getattr(args, "deepep_num_max_dispatch_tokens_per_rank", None)
                if getattr(args, "deepep_num_max_dispatch_tokens_per_rank", None) is not None
                else 256
            ),
            decoding_thresholds={
                "add_block_threshold": getattr(args, "add_block_threshold", 0.1),
                "semi_complete_threshold": getattr(args, "semi_complete_threshold", 0.9),
                "accept_threshold": getattr(args, "accept_threshold", 0.9),
                "remask_threshold": getattr(args, "remask_threshold", 0.4),
            },
            block_size=(args.block_size if getattr(args, "block_size", None) is not None else 32),
            buffer_size=getattr(args, "buffer_size", 4),
            multi_block_prefix_full=(
                bool(args.multi_block_prefix_full)
                if getattr(args, "multi_block_prefix_full", None) is not None
                else False
            ),
            enforce_eager=args.enforce_eager if hasattr(args, "enforce_eager") else False,
        )

        eval_config = EvalConfig(
            dataset_name=args.dataset,
            dataset_split=getattr(args, "dataset_split", "test"),
            dataset_limit=args.dataset_limit,
            dataset_data_files=getattr(args, "dataset_data_files", None),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_nfe=getattr(args, "max_nfe", None),
            max_repetition_run=getattr(args, "max_repetition_run", None),
            ignore_eos=getattr(args, "ignore_eos", False),
            output_dir=args.output_dir,
            use_run_subdirectory=(
                bool(args.use_run_subdirectory)
                if getattr(args, "use_run_subdirectory", None) is not None
                else True
            ),
            save_results=args.save_results,
            include_path=getattr(args, "include_path", None),
        )

        apply_engine_arg_overrides(engine)
        config = BenchmarkConfig(engine=engine, eval=eval_config)

    return config


def main():
    """Main function"""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup logger
    log_level = getattr(logging, args.log_level.upper())
    setup_logger("diffulex_bench", level=log_level, log_file=args.log_file)

    # Load configuration
    config = load_config_from_args(args)

    # Run benchmark using lm_eval
    run_benchmark(config)


if __name__ == "__main__":
    main()
