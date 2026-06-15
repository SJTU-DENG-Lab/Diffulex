"""DiffusionGemma-compatible GSM8K prompts and answer extraction."""

from diffulex_bench.tasks.gsm8k.sdar_utils import process_results_math


def doc_to_text_math(doc: dict) -> str:
    question = doc["question"]
    return (
        "<bos><|turn>user\n"
        f"{question}\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
        "<turn|>\n"
        "<|turn>model\n"
        "<|channel>thought\n"
        "<channel|>\n"
    )
