"""LLADA2/DMax-specific MATH-500 prompt + scoring utilities."""
import re
from functools import lru_cache
from typing import Any, Iterable, Optional

from diffulex_bench.tasks.common.lightning_math_prompts import MATH_4SHOT_EXAMPLES

try:
    from math_verify import parse, verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except Exception:
    parse = None
    verify = None
    ExprExtractionConfig = None
    LatexExtractionConfig = None

try:
    import sympy
    from sympy.parsing.latex import parse_latex
except Exception:
    sympy = None
    parse_latex = None


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def doc_to_text_math_dmax_chat(doc: dict) -> str:
    question = doc["question"]
    user_content = f"{question}\nLet's think step by step\n"
    return (
        "<role>SYSTEM</role>detailed thinking off<|role_end|>"
        f"<role>HUMAN</role>{user_content}<|role_end|>"
        "<role>ASSISTANT</role>"
    )


def doc_to_text_math_4shot_dmax_chat(doc: dict) -> str:
    question = doc["question"]
    user_content = f"{MATH_4SHOT_EXAMPLES}{question}\nLet's think step by step\n"
    return (
        "<role>SYSTEM</role>detailed thinking off<|role_end|>"
        f"<role>HUMAN</role>{user_content}<|role_end|>"
        "<role>ASSISTANT</role>"
    )


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def normalize_final_answer(text: str) -> str:
    answer = text.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        answer = answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        answer = answer.replace(expr, "")
    answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", answer)
    answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", answer)
    answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", answer)
    answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", answer)
    answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", answer)
    answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", answer)
    answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", answer)
    answer = answer.replace("$", "")
    if answer.replace(",", "").isdigit():
        answer = answer.replace(",", "")
    return answer


def strip_markdown(text: str) -> str:
    stripped = text.strip()
    stripped = stripped.replace("**", "").replace("__", "")
    return stripped.strip("`")


def strip_answer_prefix(text: str) -> str:
    cleaned = strip_markdown(text)
    cleaned = re.sub(r"^\s*####\s*", "", cleaned)
    cleaned = re.sub(r"(?is)^\s*<answer>\s*", "", cleaned)
    cleaned = re.sub(r"(?is)\s*</answer>\s*$", "", cleaned)
    cleaned = re.sub(r"(?is)^\s*(?:final answer|answer)\s*[:：]\s*", "", cleaned)
    cleaned = re.sub(r"(?is)^\s*the final answer is\s*", "", cleaned)
    cleaned = re.sub(r"(?is)\s*I hope it is correct\.?\s*$", "", cleaned)
    return cleaned.strip()


def extract_braced_content(text: str, open_index: int) -> tuple[Optional[str], Optional[int]]:
    if open_index >= len(text) or text[open_index] != "{":
        return None, None
    depth = 0
    content: list[str] = []
    for idx in range(open_index, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
            if depth > 1:
                content.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(content), idx
            content.append(char)
        else:
            content.append(char)
    return None, None


def unwrap_known_wrappers(text: str) -> str:
    current = text.strip()
    while True:
        previous = current
        current = current.strip()
        if current.startswith("$$") and current.endswith("$$") and len(current) >= 4:
            current = current[2:-2].strip()
        elif current.startswith("\\[") and current.endswith("\\]"):
            current = current[2:-2].strip()
        elif current.startswith("\\(") and current.endswith("\\)"):
            current = current[2:-2].strip()
        elif current.startswith("$") and current.endswith("$") and len(current) >= 2:
            current = current[1:-1].strip()
        elif current.startswith("\\boxed"):
            brace_start = current.find("{")
            content, brace_end = extract_braced_content(current, brace_start)
            if content is not None and brace_end == len(current) - 1:
                current = content.strip()
        elif current.startswith("\\fbox"):
            brace_start = current.find("{")
            content, brace_end = extract_braced_content(current, brace_start)
            if content is not None and brace_end == len(current) - 1:
                current = content.strip()
        elif current.startswith("\\text{") or current.startswith("\\mathrm{") or current.startswith("\\textbf{"):
            brace_start = current.find("{")
            content, brace_end = extract_braced_content(current, brace_start)
            if content is not None and brace_end == len(current) - 1:
                current = content.strip()
        current = current.strip(" \n\t\r,.;:!。；，")
        if current == previous:
            break
    return current


def cleanup_candidate(text: str) -> str:
    cleaned = strip_answer_prefix(text)
    cleaned = cleaned.replace("\u2212", "-").replace("−", "-")
    cleaned = cleaned.replace("\u00d7", "*").replace("\u00f7", "/")
    cleaned = cleaned.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    cleaned = cleaned.replace("\\left", "").replace("\\right", "")
    cleaned = cleaned.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    cleaned = cleaned.replace("\\%", "%")
    cleaned = unwrap_known_wrappers(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def canonicalize_for_compare(text: str) -> str:
    cleaned = cleanup_candidate(text)
    cleaned = normalize_final_answer(cleaned)
    cleaned = cleaned.replace("\\{", "{").replace("\\}", "}")
    cleaned = cleaned.replace(" ", "")
    return cleaned.strip(" \n\t\r,.;:!。；，")


def extract_boxed_contents(text: str) -> list[str]:
    matches: list[str] = []
    for command in ("\\boxed", "\\fbox"):
        start = 0
        while True:
            idx = text.find(command, start)
            if idx == -1:
                break
            cursor = idx + len(command)
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor < len(text) and text[cursor] == "{":
                content, end_idx = extract_braced_content(text, cursor)
                if content is not None and end_idx is not None:
                    matches.append(content.strip())
                    start = end_idx + 1
                    continue
            start = cursor + 1
    return matches


def extract_answer_by_patterns(text: str) -> list[str]:
    patterns = [
        r"(?is)<answer>\s*(.*?)\s*</answer>",
        r"(?is)Final Answer\s*[:：]\s*(.*?)(?=\n\s*\n|$)",
        r"(?is)The final answer is\s*(.*?)(?:\.?\s*I hope it is correct\.?|$)",
        r"(?im)^\s*Answer\s*[:：]\s*(.+?)\s*$",
        r"(?im)^\s*####\s*(.+?)\s*$",
    ]
    matches: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            if isinstance(match, tuple):
                for piece in match:
                    if piece and piece.strip():
                        matches.append(piece.strip())
            elif match and match.strip():
                matches.append(match.strip())
    return matches


def extract_latex_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    patterns = [
        r"(?s)\$\$(.*?)\$\$",
        r"(?s)\\\[(.*?)\\\]",
        r"(?s)\\\((.*?)\\\)",
        r"(?s)(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            if match and str(match).strip():
                blocks.append(str(match).strip())
    return blocks[-6:]


def extract_line_candidates(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates: list[str] = []
    for line in lines[-8:]:
        stripped = strip_markdown(line).lstrip("-*")
        if not stripped:
            continue
        candidates.append(stripped)
        if "=" in stripped:
            rhs = stripped.split("=")[-1].strip()
            if rhs:
                candidates.append(rhs)
    return candidates[-8:]


def extract_answer_candidates(text: str) -> list[str]:
    if not text:
        return []
    raw_text = str(text)
    candidates: list[str] = []
    candidates.extend(reversed(extract_boxed_contents(raw_text)[-6:]))
    pattern_matches = extract_answer_by_patterns(raw_text)
    for match in reversed(pattern_matches[-6:]):
        candidates.append(match)
        first_line = match.splitlines()[0].strip()
        if first_line and first_line != match:
            candidates.append(first_line)
    for block in reversed(extract_latex_blocks(raw_text)):
        candidates.append(block)
    candidates.extend(reversed(extract_line_candidates(raw_text)))
    raw_text_stripped = raw_text.strip()
    if raw_text_stripped and len(raw_text_stripped) <= 200:
        candidates.append(raw_text_stripped)
    cleaned = [cleanup_candidate(item) for item in candidates]
    cleaned = [item for item in cleaned if item and item.lower() not in {"answer", "final answer"}]
    return dedupe_keep_order(cleaned)[:20]


def extract_ground_truth_answer_candidates(doc: dict[str, Any]) -> list[str]:
    answer_text = str(doc.get("ground_truth_answer", "") or "").strip()
    if not answer_text:
        return []
    candidates = [cleanup_candidate(answer_text)]
    candidates.extend(extract_answer_candidates(answer_text))
    return [item for item in dedupe_keep_order(candidates) if item]


def extract_llm_final_answer_candidates(text: str) -> list[str]:
    if not text:
        return []
    raw_text = str(text)
    tail_text = raw_text.strip()[-800:]
    tail_lines = [line.strip() for line in tail_text.splitlines() if line.strip()]
    candidates: list[str] = []
    pattern_matches = extract_answer_by_patterns(raw_text)
    if pattern_matches:
        candidates.append(cleanup_candidate(pattern_matches[-1]))
    boxed = extract_boxed_contents(raw_text)
    if boxed:
        candidates.append(cleanup_candidate(boxed[-1]))
    tail_latex_blocks = extract_latex_blocks(tail_text)
    if tail_latex_blocks:
        candidates.append(cleanup_candidate(tail_latex_blocks[-1]))
    for line in reversed(tail_lines[-3:]):
        stripped = strip_markdown(line).lstrip("-*").strip()
        if not stripped:
            continue
        candidates.append(cleanup_candidate(stripped))
        if "=" in stripped:
            rhs = stripped.split("=")[-1].strip()
            if rhs:
                candidates.append(cleanup_candidate(rhs))
    raw_text_stripped = raw_text.strip()
    if raw_text_stripped and len(raw_text_stripped) <= 200:
        candidates.append(cleanup_candidate(raw_text_stripped))
    return [item for item in dedupe_keep_order(candidates) if item][:10]


def maybe_parse_numeric(text: str) -> Optional[Any]:
    if sympy is None:
        return None
    candidate = cleanup_candidate(text)
    if not candidate:
        return None
    if candidate.endswith("%"):
        candidate = f"({candidate[:-1]})/100"
    if any(token in candidate for token in ("\\frac", "\\sqrt", "\\pi", "\\cdot", "\\pm", "{", "}")):
        try:
            return parse_latex(candidate)
        except Exception:
            pass
    ascii_candidate = candidate.replace("^", "**").replace("\\pi", "pi").replace("\\cdot", "*")
    ascii_candidate = ascii_candidate.replace("{", "(").replace("}", ")")
    try:
        return sympy.sympify(ascii_candidate)
    except Exception:
        return None


def are_sympy_equivalent(left: str, right: str) -> bool:
    left_expr = maybe_parse_numeric(left)
    right_expr = maybe_parse_numeric(right)
    if left_expr is None or right_expr is None:
        return False
    try:
        return sympy.simplify(left_expr - right_expr) == 0
    except Exception:
        return False


@lru_cache(maxsize=16384)
def cached_parse(text: str, mode: str) -> Any:
    if parse is None:
        return None
    if mode == "default":
        return parse(text)
    if mode == "snippet":
        return parse(text, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
    raise ValueError(f"Unknown parse mode: {mode}")


def try_math_verify(gold_text: str, pred_text: str) -> bool:
    if parse is None or verify is None:
        return False
    cleaned_gold = cleanup_candidate(gold_text)
    cleaned_pred = cleanup_candidate(pred_text)
    gold_variants = [x for x in dedupe_keep_order([gold_text, cleaned_gold, f"${cleaned_gold}$"]) if x]
    pred_variants = [x for x in dedupe_keep_order([pred_text, cleaned_pred, f"${cleaned_pred}$"]) if x]
    for gold_variant in gold_variants:
        gold_parsed = cached_parse(gold_variant, "snippet")
        if gold_parsed is None:
            continue
        for pred_variant in pred_variants:
            pred_parsed = cached_parse(pred_variant, "snippet")
            if pred_parsed is None:
                continue
            try:
                if verify(gold_parsed, pred_parsed):
                    return True
            except Exception:
                continue
    return False


def process_results_math_dmax_chat(doc: dict, results: list[str]) -> dict[str, Any]:
    prediction = results[0] if results else ""
    pred_candidates = extract_llm_final_answer_candidates(prediction)
    gold_candidates = extract_ground_truth_answer_candidates(doc)
    correct = False
    for gold in gold_candidates:
        gold_norm = canonicalize_for_compare(gold)
        if not gold_norm:
            continue
        for pred in pred_candidates:
            pred_norm = canonicalize_for_compare(pred)
            if gold_norm and gold_norm == pred_norm:
                correct = True
                break
        if correct:
            break
    if not correct:
        for gold in gold_candidates:
            for pred in pred_candidates:
                if are_sympy_equivalent(gold, pred) or try_math_verify(gold, pred):
                    correct = True
                    break
            if correct:
                break
    return {"exact_match": int(correct)}
