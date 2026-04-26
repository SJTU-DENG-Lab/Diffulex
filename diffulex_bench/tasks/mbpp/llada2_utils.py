"""LLADA2/DMax-specific MBPP prompt + code postprocess utilities."""
import ast
import re
from typing import Any

from diffulex_bench.tasks.utils import evaluate_code_function, evaluate_code_stdio


def doc_to_text_code_function_dmax_chat(doc: dict) -> str:
    problem = doc.get("text", doc.get("prompt", doc.get("question", "")))
    user_content = (
        "You are an expert Python programmer, and here is your task:\n"
        f"{problem}\n\n"
        "Please enclose your code within delimiters as follows:\n"
        "```python\n# YOUR CODE HERE\n```\n\n"
    )
    return (
        "<role>SYSTEM</role>detailed thinking off<|role_end|>"
        f"<role>HUMAN</role>{user_content}<|role_end|>"
        "<role>ASSISTANT</role>"
    )


def doc_to_text_code_stdio_dmax_chat(doc: dict) -> str:
    problem = doc.get("text", doc.get("prompt", doc.get("question", "")))
    user_content = (
        "This is the problem:\n"
        f"{problem}\n\n"
        "You should put your code in ```python ```. "
        "Use input() to read input and print() to produce output in your script.\n"
    )
    return (
        "<role>SYSTEM</role>detailed thinking off<|role_end|>"
        f"<role>HUMAN</role>{user_content}<|role_end|>"
        "<role>ASSISTANT</role>"
    )


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\n(.*?)\n```", text, re.DOTALL)
    if blocks:
        text = blocks[0].strip()

    patterns = [
        r"'(.*)'\s*$$DONE$$",
        r"$$BEGIN$$\s*'(.*)'\s*$$DONE$$",
        r"BEGIN\s*'(.*)'\s*$$DONE$$",
        r"$$BEGIN$$\s*'(.*)'\s*DONE",
        r"BEGIN\s*'(.*)'\s*DONE",
        r"$$BEGIN$$\s*'(.*)\s*$$DONE$$",
        r"BEGIN\s*'(.*)\s*$$DONE$$",
        r"$$BEGIN$$\s*'(.*)\s*DONE",
        r"BEGIN\s*'(.*)\s*DONE",
        r"$$BEGIN$$\s*(.*)\s*$$DONE$$",
        r"BEGIN\s*(.*)\s*$$DONE$$",
        r"$$BEGIN$$\s*(.*)\s*DONE",
        r"BEGIN\s*(.*)\s*DONE",
        r"```python\s*(.*)\s*```",
        r"```\s*(.*)\s*```",
        r"```python\s*(.*)\s*$",
        r"```\s*(.*)\s*$",
        r"(.*)\s*```.*",
        r"\[BEGIN\]\s*'(.*)",
        r"\[BEGIN\](.*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            text = match.group(1)
            break

    text = text.split("```")[0]
    text = re.split(r"'?\s*\$\$?DONE\$\$?", text)[0]
    text = text.replace("\\_", "_").strip()

    try:
        tree = ast.parse(text)
        filtered_nodes = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))
        ]
        if filtered_nodes:
            text = "\n\n".join(ast.unparse(node) for node in filtered_nodes)
    except Exception:
        pass

    return text.strip()


def _code_text_for_prediction(doc: dict, prediction: str) -> str:
    if "```" in prediction:
        return prediction
    return doc.get("prefix", "") + prediction


def process_results_code_dmax_chat(doc: dict, results: list[str]) -> dict[str, Any]:
    prediction = results[0] if results else ""
    test_method = doc.get("test_method", "function")

    if test_method == "function":
        code = extract_code(_code_text_for_prediction(doc, prediction))
        tests = doc.get("test_list", [])
        timeout = doc.get("test_time_limit", 1)
        correctness = evaluate_code_function(code, tests, timeout)
        passed = all(correctness)
    else:
        code = extract_code(prediction)
        test_inputs = doc.get("test_input", [])
        test_outputs = doc.get("test_output", [])
        timeout = doc.get("test_time_limit", 1)
        correctness = []
        for inp, out in zip(test_inputs, test_outputs):
            correctness.append(evaluate_code_stdio(code, inp, out, timeout))
        passed = all(correctness)

    return {"exact_match": int(passed)}
