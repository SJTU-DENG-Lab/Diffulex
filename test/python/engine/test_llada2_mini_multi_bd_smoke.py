import os
import socket

from transformers import AutoTokenizer

from diffulex import Diffulex, SamplingParams


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _pick_free_port() -> int:
    configured = os.environ.get("DIFFULEX_MASTER_PORT")
    if configured:
        return int(configured)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    model_path = os.environ.get("DIFFULEX_MODEL_PATH", "/data1/ckpts/inclusionAI/LLaDA2.0-mini")
    model_name = os.environ.get("DIFFULEX_MODEL_NAME", "llada2_mini")
    system_prompt = os.environ.get("DIFFULEX_SYSTEM_PROMPT", "You are a helpful assistant.")
    user_prompt = os.environ.get("DIFFULEX_SMOKE_PROMPT", "What is 2 + 3? Answer with a short sentence.")

    tp_size = _env_int("DIFFULEX_TP_SIZE", 4)
    dp_size = _env_int("DIFFULEX_DP_SIZE", 1)
    ep_size = _env_int("DIFFULEX_EP_SIZE", 8)
    buffer_size = _env_int("DIFFULEX_BUFFER_SIZE", 1)
    max_model_len = _env_int("DIFFULEX_MAX_MODEL_LEN", 512)
    max_num_batched_tokens = _env_int("DIFFULEX_MAX_BATCHED_TOKENS", 512)
    max_num_reqs = _env_int("DIFFULEX_MAX_NUM_REQS", 1)
    max_tokens = _env_int("DIFFULEX_MAX_TOKENS", 16)
    max_nfe = _env_int("DIFFULEX_MAX_NFE", 16)
    distributed_timeout_seconds = _env_int("DIFFULEX_DISTRIBUTED_TIMEOUT_SECONDS", 3600)
    gpu_memory_utilization = _env_float("DIFFULEX_GPU_MEMORY_UTILIZATION", 0.35)
    moe_dispatcher_backend = os.environ.get("DIFFULEX_MOE_DISPATCHER_BACKEND", "standard")
    deepep_mode = os.environ.get("DIFFULEX_DEEPEP_MODE", "auto")
    deepep_num_max_dispatch_tokens_per_rank = _env_int("DIFFULEX_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 256)
    master_port = _pick_free_port()

    print(
        {
            "model_path": model_path,
            "model_name": model_name,
            "tp_size": tp_size,
            "dp_size": dp_size,
            "ep_size": ep_size,
            "buffer_size": buffer_size,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_num_reqs": max_num_reqs,
            "max_tokens": max_tokens,
            "max_nfe": max_nfe,
            "distributed_timeout_seconds": distributed_timeout_seconds,
            "gpu_memory_utilization": gpu_memory_utilization,
            "moe_dispatcher_backend": moe_dispatcher_backend,
            "deepep_mode": deepep_mode,
            "deepep_num_max_dispatch_tokens_per_rank": deepep_num_max_dispatch_tokens_per_rank,
            "master_port": master_port,
        },
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    print("prompt_len", len(prompt_ids), flush=True)

    llm = Diffulex(
        model_path,
        model_name=model_name,
        decoding_strategy="multi_bd",
        buffer_size=buffer_size,
        sampling_mode="naive",
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
        expert_parallel_size=ep_size,
        master_port=master_port,
        distributed_timeout_seconds=distributed_timeout_seconds,
        moe_dispatcher_backend=moe_dispatcher_backend,
        deepep_mode=deepep_mode,
        deepep_num_max_dispatch_tokens_per_rank=deepep_num_max_dispatch_tokens_per_rank,
        enforce_eager=bool(_env_int("DIFFULEX_ENFORCE_EAGER", 1)),
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    try:
        outputs = llm.generate(
            [prompt_ids],
            SamplingParams(temperature=0.0, max_tokens=max_tokens, max_nfe=max_nfe),
            use_tqdm=False,
        )
        print("outputs_type", type(outputs).__name__, flush=True)
        print("outputs_len", len(outputs.trajectories), flush=True)
        if outputs.trajectories:
            trajectory = outputs.trajectories[0]
            print("output0_type", type(trajectory).__name__, flush=True)
            print("output0_reason", trajectory.completion_reason, flush=True)
            print("output0_token_ids", trajectory.token_ids, flush=True)
        print("benchmark_outputs", outputs.to_benchmark_format(), flush=True)
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
