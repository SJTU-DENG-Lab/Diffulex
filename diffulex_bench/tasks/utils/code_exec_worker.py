"""Standalone worker for benchmark code execution.

Keep this module stdlib-only. It is launched as a separate Python process so
code evaluation does not inherit CUDA state from the benchmark process.
"""

import contextlib
import io
import json
import sys
import textwrap


def _eval_function(payload: dict) -> list[bool]:
    results = []
    ns = {}
    try:
        exec(textwrap.dedent(payload.get("code", "")), ns, ns)
        for stmt in payload.get("tests", []):
            try:
                exec(stmt, ns, ns)
                results.append(True)
            except SystemExit:
                results.append(True)
            except Exception:
                results.append(False)
    except SystemExit:
        results = [True] * len(payload.get("tests", []))
    except Exception:
        results = [False] * len(payload.get("tests", []))
    return results


def _eval_stdio(payload: dict) -> str:
    input_lines = iter(str(payload.get("input", "")).splitlines())

    def fake_input(prompt=""):
        try:
            return next(input_lines)
        except StopIteration:
            raise EOFError("No more input")

    stdout_capture = io.StringIO()
    context = {"__name__": "__main__", "input": fake_input}
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(payload.get("code", ""), context)
    except SystemExit:
        pass
    except Exception as exc:
        return f"error: {exc}"
    return stdout_capture.getvalue()


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        mode = payload.get("mode")
        if mode == "function":
            result = {"ok": True, "results": _eval_function(payload)}
        elif mode == "stdio":
            result = {"ok": True, "output": _eval_stdio(payload)}
        else:
            result = {"ok": False, "error": f"unknown mode: {mode}"}
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
