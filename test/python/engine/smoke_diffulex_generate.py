import os
import socket

from transformers import AutoTokenizer

from diffulex import Diffulex, SamplingParams


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return bool(_env_int(name, int(default)))


def _env_device_ids(name: str, default_count: int) -> list[int]:
    value = os.environ.get(name)
    if not value:
        return list(range(default_count))
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _pick_free_port() -> int:
    configured = os.environ.get("DIFFULEX_MASTER_PORT")
    if configured:
        return int(configured)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_prompts(model_path: str) -> list[str] | list[list[int]]:
    prompt = os.environ.get("DIFFULEX_SMOKE_PROMPT", "What is 2+2? Answer briefly.")
    prompts = [part.strip() for part in os.environ.get("DIFFULEX_SMOKE_PROMPTS", prompt).split("||") if part.strip()]
    if not _env_bool("DIFFULEX_USE_CHAT_TEMPLATE", False):
        return prompts

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    system_prompt = os.environ.get("DIFFULEX_SYSTEM_PROMPT", "You are a helpful assistant.")
    prompt_ids = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            tokenize=True,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    return [ids["input_ids"] if hasattr(ids, "__contains__") and "input_ids" in ids else ids for ids in prompt_ids]


def main() -> None:
    model_path = os.environ["DIFFULEX_MODEL_PATH"]
    model_name = os.environ["DIFFULEX_MODEL_NAME"]
    decoding_strategy = os.environ.get("DIFFULEX_DECODING_STRATEGY", "multi_bd")
    tp_size = _env_int("DIFFULEX_TP_SIZE", 1)
    dp_size = _env_int("DIFFULEX_DP_SIZE", 1)
    ep_size = _env_int("DIFFULEX_EP_SIZE", 1)
    device_ids = _env_device_ids("DIFFULEX_DEVICE_IDS", tp_size * dp_size * ep_size)
    max_tokens = _env_int("DIFFULEX_MAX_TOKENS", 4)
    max_nfe = _env_int("DIFFULEX_MAX_NFE", 2)

    kwargs = dict(
        model_name=model_name,
        decoding_strategy=decoding_strategy,
        sampling_mode=os.environ.get("DIFFULEX_SAMPLING_MODE", "naive"),
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
        expert_parallel_size=ep_size,
        device_ids=device_ids,
        master_port=_pick_free_port(),
        distributed_backend=os.environ.get("DIFFULEX_DISTRIBUTED_BACKEND", "nccl"),
        distributed_timeout_seconds=_env_int("DIFFULEX_DISTRIBUTED_TIMEOUT_SECONDS", 3600),
        gpu_memory_utilization=_env_float("DIFFULEX_GPU_MEMORY_UTILIZATION", 0.45),
        max_model_len=_env_int("DIFFULEX_MAX_MODEL_LEN", 512),
        max_num_batched_tokens=_env_int("DIFFULEX_MAX_BATCHED_TOKENS", 512),
        max_num_reqs=_env_int("DIFFULEX_MAX_NUM_REQS", 1),
        block_size=_env_int("DIFFULEX_BLOCK_SIZE", 32),
        page_size=_env_int("DIFFULEX_PAGE_SIZE", 32),
        buffer_size=_env_int("DIFFULEX_BUFFER_SIZE", 1),
        kv_cache_layout=os.environ.get("DIFFULEX_KV_CACHE_LAYOUT", "unified"),
        attn_impl=os.environ.get("DIFFULEX_ATTN_IMPL", "triton"),
        moe_dispatcher_backend=os.environ.get("DIFFULEX_MOE_DISPATCHER_BACKEND", "standard"),
        moe_gemm_impl=os.environ.get("DIFFULEX_MOE_GEMM_IMPL", "vllm_modular"),
        enable_vllm_layers=_env_bool("DIFFULEX_ENABLE_VLLM_LAYERS", True),
        enable_prefix_caching=_env_bool("DIFFULEX_ENABLE_PREFIX_CACHING", True),
        enforce_eager=_env_bool("DIFFULEX_ENFORCE_EAGER", True),
        skip_warmup=_env_bool("DIFFULEX_SKIP_WARMUP", True),
        enable_prefill_cudagraph=_env_bool("DIFFULEX_ENABLE_PREFILL_CUDAGRAPH", False),
        enable_full_static_runner=_env_bool("DIFFULEX_ENABLE_FULL_STATIC_RUNNER", False),
        enable_torch_compile=_env_bool("DIFFULEX_ENABLE_TORCH_COMPILE", False),
        enable_cudagraph_torch_compile=_env_bool("DIFFULEX_ENABLE_CUDAGRAPH_TORCH_COMPILE", False),
        diffusion_gemma_max_denoising_steps=_env_int("DIFFULEX_DIFFUSION_GEMMA_MAX_DENOISING_STEPS", 4),
        diffusion_gemma_stability_threshold=_env_int("DIFFULEX_DIFFUSION_GEMMA_STABILITY_THRESHOLD", 1),
        diffusion_gemma_confidence_threshold=_env_float("DIFFULEX_DIFFUSION_GEMMA_CONFIDENCE_THRESHOLD", 999.0),
        num_pages=_env_int("DIFFULEX_NUM_PAGES", -1),
    )
    if _env_bool("DIFFULEX_USE_LORA", False):
        kwargs["use_lora"] = True
        kwargs["lora_path"] = os.environ["DIFFULEX_LORA_PATH"]
        kwargs["pre_merge_lora"] = _env_bool("DIFFULEX_PRE_MERGE_LORA", True)

    printed = {k: v for k, v in kwargs.items() if k not in {"hf_config"}}
    printed["model_path"] = model_path
    print(printed, flush=True)

    prompts = _build_prompts(model_path)
    print("prompt_lens", [len(prompt) if not isinstance(prompt, str) else len(prompt) for prompt in prompts], flush=True)

    llm = Diffulex(model_path, **kwargs)
    try:
        outputs = llm.generate(
            prompts,
            SamplingParams(temperature=0.0, max_tokens=max_tokens, max_nfe=max_nfe, ignore_eos=False),
            use_tqdm=False,
        )
        print("outputs_len", len(outputs.trajectories), flush=True)
        for idx, trajectory in enumerate(outputs.trajectories):
            print(
                f"output{idx}: reason={trajectory.completion_reason} "
                f"tokens={len(trajectory.token_ids)} text={trajectory.text[:160]!r}",
                flush=True,
            )
        print("benchmark_outputs", outputs.to_benchmark_format(), flush=True)
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
