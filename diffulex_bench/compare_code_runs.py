"""Compare two lm-eval code-task run directories."""

from __future__ import annotations

import argparse
import ast
import json
import statistics
from pathlib import Path

from diffulex_bench.tasks.humaneval.llada2_utils import extract_code as extract_humaneval_code
from diffulex_bench.tasks.mbpp.llada2_utils import (
    _code_text_for_prediction as mbpp_code_text_for_prediction,
)
from diffulex_bench.tasks.mbpp.llada2_utils import extract_code as extract_mbpp_code


def _first_file(run_dir: Path, pattern: str) -> Path:
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} under {run_dir}")
    return matches[0]


def _sample_path(run_dir: Path) -> Path:
    return _first_file(run_dir, "**/samples_*.jsonl")


def _result_path(run_dir: Path) -> Path:
    return _first_file(run_dir, "**/results_*.json")


def _load_samples(run_dir: Path) -> dict[int, dict]:
    path = _sample_path(run_dir)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {int(row["doc_id"]): row for row in rows}


def _extract_code(row: dict) -> str:
    pred = row["filtered_resps"][0]
    doc = row["doc"]
    task_id = str(doc.get("task_id", "")).lower()
    if "humaneval" in task_id:
        text = pred if "```" in pred else doc.get("prefix", "") + pred
        return extract_humaneval_code(text)
    return extract_mbpp_code(mbpp_code_text_for_prediction(doc, pred))


def _syntax_error(row: dict) -> bool:
    try:
        ast.parse(_extract_code(row))
        return False
    except SyntaxError:
        return True


def _score(row: dict) -> int:
    return int(row.get("exact_match", 0))


def _short(text: str, limit: int) -> str:
    text = text.replace("\r", "")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...<truncated>"


def _print_run_summary(label: str, run_dir: Path, rows: dict[int, dict]) -> None:
    result = json.loads(_result_path(run_dir).read_text(encoding="utf-8"))
    task_name = next(iter(result["results"]))
    scores = [_score(row) for row in rows.values()]
    preds = [row["filtered_resps"][0] for row in rows.values()]
    lengths = [len(pred) for pred in preds]
    syntax = sum(_syntax_error(row) for row in rows.values())
    masks = sum("<|mask|>" in pred for pred in preds)
    no_fence = sum("```" not in pred for pred in preds)
    long = sum(len(pred) > 3000 for pred in preds)
    metadata = result["configs"][task_name].get("metadata", {})

    print(f"\n### {label}: {run_dir.name}")
    print("result:", result["results"][task_name])
    print(f"pass: {sum(scores)}/{len(scores)} = {sum(scores) / len(scores):.6f}")
    print(
        "metadata:",
        {
            k: metadata.get(k)
            for k in (
                "buffer_size",
                "block_size",
                "max_nfe",
                "max_new_tokens",
                "decoding_strategy",
                "sampling_mode",
                "accept_threshold",
                "remask_threshold",
                "token_merge_mode",
                "token_merge_top_k",
            )
        },
    )
    print(
        "response_chars mean/p50/max:",
        round(statistics.mean(lengths), 1),
        statistics.median(lengths),
        max(lengths),
    )
    print("syntax_errors:", syntax, "mask_outputs:", masks, "no_fence:", no_fence, "long>3000:", long)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_run", type=Path)
    parser.add_argument("candidate_run", type=Path)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--chars", type=int, default=900)
    args = parser.parse_args()

    base = _load_samples(args.base_run)
    cand = _load_samples(args.candidate_run)
    ids = sorted(set(base) & set(cand))

    _print_run_summary("base", args.base_run, base)
    _print_run_summary("candidate", args.candidate_run, cand)

    both_right = [i for i in ids if _score(base[i]) == 1 and _score(cand[i]) == 1]
    regress = [i for i in ids if _score(base[i]) == 1 and _score(cand[i]) == 0]
    improve = [i for i in ids if _score(base[i]) == 0 and _score(cand[i]) == 1]
    both_wrong = [i for i in ids if _score(base[i]) == 0 and _score(cand[i]) == 0]

    print("\n=== migration ===")
    print("both_right:", len(both_right), "regress:", len(regress), "improve:", len(improve), "both_wrong:", len(both_wrong))
    print("regress first:", regress[:20])
    print("improve first:", improve[:20])
    print("regress syntax in base/candidate:", sum(_syntax_error(base[i]) for i in regress), sum(_syntax_error(cand[i]) for i in regress))

    for title, selected in (("REGRESS", regress[: args.examples]), ("IMPROVE", improve[: args.examples])):
        print(f"\n=== {title} examples ===")
        for doc_id in selected:
            before = base[doc_id]
            after = cand[doc_id]
            doc = before["doc"]
            prompt = doc.get("prompt") or doc.get("text") or doc.get("question") or ""
            tests = "\n".join(doc.get("test_list", []))
            print(f"\n--- doc_id {doc_id} task_id {doc.get('task_id')} ---")
            print("prompt:", _short(prompt, 240))
            print("tests:", _short(tests, 360))
            print("[base]", _score(before), "syntax_error", _syntax_error(before), "chars", len(before["filtered_resps"][0]))
            print(_short(before["filtered_resps"][0], args.chars))
            print("[candidate]", _score(after), "syntax_error", _syntax_error(after), "chars", len(after["filtered_resps"][0]))
            print(_short(after["filtered_resps"][0], args.chars))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
