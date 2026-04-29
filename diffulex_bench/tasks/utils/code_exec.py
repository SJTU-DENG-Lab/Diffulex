"""Code execution utilities from LightningRL"""
import json
import os
import subprocess
import sys
from pathlib import Path


_WORKER_PATH = Path(__file__).with_name("code_exec_worker.py")


def _run_worker(payload: dict, timeout: float) -> dict | None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    try:
        completed = subprocess.run(
            [sys.executable, str(_WORKER_PATH)],
            input=json.dumps(payload),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    try:
        return json.loads(completed.stdout)
    except Exception:
        return None


def _check_snippet_many(snippet: str, tests: list[str], t_limit: int, spawn_slack: float = 2.0) -> list[bool]:
    payload = {"mode": "function", "code": snippet, "tests": tests}
    result = _run_worker(payload, timeout=t_limit + spawn_slack)
    if not result or not result.get("ok"):
        return [False] * len(tests)
    res = result.get("results")
    if not isinstance(res, list):
        return [False] * len(tests)
    return [bool(x) for x in res]


def evaluate_code_function(code: str, tests: list[str], timeout: int = 1) -> list[bool]:
    """Evaluate function-based code with test cases"""
    return _check_snippet_many(code, tests, timeout)


def evaluate_code_stdio(code: str, test_input: str, expected_output: str, timeout: int = 1) -> bool:
    """Evaluate stdio-based code"""
    payload = {"mode": "stdio", "code": code, "input": test_input}
    result = _run_worker(payload, timeout=timeout + 2.0)
    if not result or not result.get("ok"):
        return False
    printed_output = str(result.get("output", ""))
    return " ".join(printed_output.split()) == " ".join(expected_output.split())
