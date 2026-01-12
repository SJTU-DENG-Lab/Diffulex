import argparse
import os
import asyncio
import time
from pathlib import Path
import sys

from tqdm import tqdm
from transformers import AutoTokenizer

from diffulex import Diffulex, SamplingParams


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


async def run_async_inference(worker, prompts, sampling_params):
    """Run async inference using generate_async."""
    outputs = await worker.generate_async(prompts, sampling_params, use_tqdm=True)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Test async inference with fast_dllm_v2")
    parser.add_argument(
        "--model",
        type=str,
        default="/data1/ckpts/Efficient-Large-Model/Fast_dLLM_v2_7B",
        help="Fast_dLLM_v2 model directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
        help="Input prompt for testing",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Number of prompts to test (will duplicate the prompt)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.25,
        help="GPU memory utilization",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Testing Async Inference with Fast_dLLM_v2")
    print("=" * 80)
    print(f"[model] {args.model}")
    print(f"[prompt] {args.prompt[:100]}..." if len(args.prompt) > 100 else f"[prompt] {args.prompt}")
    print(f"[num_prompts] {args.num_prompts}")
    print(f"[max_tokens] {args.max_tokens}")
    print(f"[temperature] {args.temperature}")
    print("=" * 80)

    # Create Diffulex engine for async inference
    print("\n[Initializing Diffulex engine...]")
    worker = Diffulex(
        model=args.model,
        use_lora=False,
        model_name="fast_dllm_v2",
        enforce_eager=True,
        data_parallel_size=1,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=2048,
        max_num_seqs=args.num_prompts,
        max_model_len=2048,
        kv_cache_layout="unified",
        decoding_strategy="block_diffusion",
        mask_token_id=151665,
        master_addr="127.0.0.1",
        master_port=2333,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)

    prompts = [args.prompt] * args.num_prompts
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    print("\n[Running async inference...]")
    print("=" * 80)

    # Run async inference
    try:
        start_time = time.time()
        outputs = asyncio.run(run_async_inference(worker, prompts, sampling_params))
        end_time = time.time()

        elapsed_time = end_time - start_time
        total_tokens = sum(len(o['token_ids']) for o in outputs)

        print("\n" + "=" * 80)
        print("[Async Inference Results]")
        print("=" * 80)
        print(f"Generated {len(outputs)} outputs")
        print(f"Total tokens: {total_tokens}")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Average TPS: {total_tokens / elapsed_time:.2f} tok/s")
        if outputs and 'n_diff_steps' in outputs[0]:
            avg_diff_steps = sum(o['n_diff_steps'] for o in outputs) / len(outputs)
            print(f"Average steps: {avg_diff_steps:.2f}")

        print("\n" + "=" * 80)
        print("[Individual Results]")
        print("=" * 80)
        for idx, output in enumerate(outputs):
            print(f"\n[Output {idx + 1}/{len(outputs)}]")
            print(f"Text: {output['text']}")
            print(f"Token IDs length: {len(output['token_ids'])}")
            if 'n_diff_steps' in output:
                print(f"Number of steps: {output['n_diff_steps']}")
            print("-" * 80)

    except Exception as e:
        print(f"\n[Error during async inference]")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n[Cleaning up...]")
        worker.exit()
        print("[Done]")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
