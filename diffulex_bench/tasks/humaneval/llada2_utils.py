"""LLADA2/DMax-specific HumanEval prompt + code postprocess utilities."""
import ast
import re
from typing import Any

from diffulex_bench.tasks.utils import evaluate_code_function


def doc_to_text_code_function_dmax_chat(doc: dict) -> str:
    problem = doc.get("text", doc.get("prompt", doc.get("question", "")))
    user_content = (
        "Write a solution to the following problem and make sure that it passes the tests:\n"
        f"```python\n{problem}\n```\n\n"
        "Please enclose your code within delimiters as follows:\n"
        "```python\n# YOUR CODE HERE\n```\n\n"
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


def process_results_code_dmax_chat(doc: dict, results: list[str]) -> dict[str, Any]:
    prediction = results[0] if results else ""
    prefix = doc.get("prefix", "")
    code = extract_code(prefix + prediction)
    tests = doc.get("test_list", [])
    timeout = doc.get("test_time_limit", 1)
    correctness = evaluate_code_function(code, tests, timeout)
    return {"exact_match": int(all(correctness))}
