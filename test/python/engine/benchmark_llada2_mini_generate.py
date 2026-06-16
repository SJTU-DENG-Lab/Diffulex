from __future__ import annotations

import argparse
import json
import os
import socket
import time

from transformers import AutoTokenizer

from diffulex import Diffulex, SamplingParams


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _bool_env(name: str, default: bool) -> bool:
    return bool(int(os.environ.get(name, "1" if default else "0")))


def _make_prompt_ids(tokenizer, prompt: str) -> list[int]:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)


def _make_warmup_prompt_ids(token_ids: list[int], tokenizer) -> list[int]:
    warmup_ids = list(token_ids)
    if not warmup_ids:
        return warmup_ids
    replacement = getattr(tokenizer, "unk_token_id", None)
    if replacement is None:
        replacement = getattr(tokenizer, "eos_token_id", None)
    if replacement is None:
        replacement = (int(warmup_ids[0]) + 1) % int(getattr(tokenizer, "vocab_size", 32000))
    if int(replacement) == int(warmup_ids[0]):
        replacement = (int(replacement) + 1) % int(getattr(tokenizer, "vocab_size", 32000))
    warmup_ids[0] = int(replacement)
    return warmup_ids


def _summarize_outputs(name: str, outputs, wall_time: float) -> dict:
    total_tokens = sum(len(t.token_ids) for t in outputs.trajectories)
    total_full_tokens = sum(len(t.full_token_ids) for t in outputs.trajectories)
    reasons = [t.completion_reason for t in outputs.trajectories]
    return {
        "case": name,
        "num_reqs": len(outputs.trajectories),
        "total_tokens": total_tokens,
        "total_full_tokens": total_full_tokens,
        "batch_steps": outputs.batch_step_count,
        "internal_total_time_s": outputs.total_time,
        "wall_time_s": wall_time,
        "tpf": outputs.tpf,
        "ttft_s": outputs.ttft,
        "tpot_s": outputs.tpot,
        "prefill_tokens": outputs.prefill_tokens,
        "prefill_time_s": outputs.prefill_time,
        "prefill_throughput_tok_s": outputs.prefill_throughput,
        "decode_tokens": outputs.decode_tokens,
        "decode_time_s": outputs.decode_time,
        "e2e_throughput_tok_s": outputs.e2e_throughput,
        "decode_throughput_tok_s": outputs.decode_throughput,
        "avg_e2e_tps_tok_s": outputs.avg_e2e_tps,
        "avg_decode_tps_tok_s": outputs.avg_decode_tps,
        "completion_reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data1/ckpts/inclusionAI/LLaDA2.0-mini")
    parser.add_argument("--model-name", default="llada2_mini")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-reqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--distributed-backend", default=os.environ.get("DIFFULEX_DISTRIBUTED_BACKEND", "gloo"))
    parser.add_argument("--attn-impl", default=os.environ.get("DIFFULEX_ATTN_IMPL", "triton"))
    parser.add_argument("--moe-gemm-impl", default=os.environ.get("DIFFULEX_MOE_GEMM_IMPL", "triton"))
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-nfe", type=int, default=16)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=_bool_env("DIFFULEX_ENFORCE_EAGER", True))
    parser.add_argument("--vllm-layers", action=argparse.BooleanOptionalAction, default=_bool_env("DIFFULEX_ENABLE_VLLM_LAYERS", True))
    parser.add_argument("--disable-vllm-tp-group", action="store_true")
    parser.add_argument(
        "--cases",
        default="bs1_short,bs1_long,bs4_short,bs4_long",
        help="Comma-separated benchmark cases to run.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=1,
        help="Untimed warmup iterations per selected case, using same-length mutated prompts.",
    )
    args = parser.parse_args()

    if args.disable_vllm_tp_group:
        os.environ["DIFFULEX_DISABLE_VLLM_TP_GROUP"] = "1"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    short = "What is 2 + 3? Answer with one short sentence."
    long = (
        "Read this short context and answer in one sentence. "
        "Alice has three notebooks, Bob has five notebooks, and Clara gives Alice two more. "
        "The meeting notes say the final answer should mention Alice's total. "
    ) * 4
    short_ids = _make_prompt_ids(tokenizer, short)
    long_ids = _make_prompt_ids(tokenizer, long)

    engine_kwargs = dict(
        model_name=args.model_name,
        decoding_strategy="multi_bd",
        block_size=32,
        page_size=32,
        buffer_size=1,
        sampling_mode="naive",
        attn_impl=args.attn_impl,
        tensor_parallel_size=args.tp,
        data_parallel_size=1,
        expert_parallel_size=1,
        master_port=_pick_free_port(),
        distributed_backend=args.distributed_backend,
        distributed_timeout_seconds=3600,
        moe_dispatcher_backend="standard",
        moe_gemm_impl=args.moe_gemm_impl,
        enforce_eager=args.enforce_eager,
        enable_vllm_layers=args.vllm_layers,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_reqs=args.max_num_reqs,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    print(json.dumps({"engine": engine_kwargs, "prompt_lens": {"short": len(short_ids), "long": len(long_ids)}}), flush=True)

    t0 = time.perf_counter()
    llm = Diffulex(args.model, **engine_kwargs)
    print(json.dumps({"engine_load_wall_time_s": time.perf_counter() - t0}), flush=True)

    cases = [
        ("bs1_short", [short_ids]),
        ("bs1_long", [long_ids]),
        ("bs4_short", [short_ids, short_ids, short_ids, short_ids]),
        ("bs4_long", [long_ids, long_ids, long_ids, long_ids]),
    ]
    selected_cases = {name.strip() for name in args.cases.split(",") if name.strip()}
    cases = [(name, prompts) for name, prompts in cases if name in selected_cases]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        max_nfe=args.max_nfe,
        ignore_eos=args.ignore_eos,
    )
    try:
        for warmup_idx in range(max(0, int(args.warmup_iters))):
            for name, prompts in cases:
                warmup_prompts = [_make_warmup_prompt_ids(prompt, tokenizer) for prompt in prompts]
                llm.generate(warmup_prompts, sampling_params, use_tqdm=False)
            print(json.dumps({"warmup_iter": warmup_idx + 1, "cases": [name for name, _ in cases]}), flush=True)

        for name, prompts in cases:
            start = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            result = _summarize_outputs(name, outputs, time.perf_counter() - start)
            print(json.dumps(result), flush=True)
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
