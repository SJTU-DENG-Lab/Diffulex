"""DiffusionGemma-compatible GSM8K prompts and answer extraction."""

from typing import Any

from diffulex_bench.tasks.gsm8k.llada2_utils import process_results_math_dmax_chat


def process_results_math(doc: dict, results: list[str]) -> dict[str, Any]:
    return process_results_math_dmax_chat(doc, results)


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
