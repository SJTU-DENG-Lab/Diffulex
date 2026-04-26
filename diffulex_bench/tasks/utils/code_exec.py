"""Code execution utilities from LightningRL"""
import io
import sys
import textwrap
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def _run_many_pipe(snippet: str, tests: list[str], conn):
    results = []
    try:
        ns = {}
        exec(textwrap.dedent(snippet), ns, ns)
        for stmt in tests:
            try:
                exec(stmt, ns, ns)
                results.append(True)
            except SystemExit:
                results.append(True)
            except Exception:
                results.append(False)
        conn.send(results)
    except SystemExit:
        conn.send([True] * len(tests))
    except Exception:
        conn.send([False] * len(tests))
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _check_snippet_many(snippet: str, tests: list[str], t_limit: int, spawn_slack: float = 2.0) -> list[bool]:
    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(start_method)
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_run_many_pipe, args=(snippet, tests, child_conn), daemon=True)
    p.start()
    child_conn.close()

    deadline = time.monotonic() + t_limit + spawn_slack
    res = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wait = remaining if remaining < 0.05 else 0.05
            if parent_conn.poll(wait):
                try:
                    res = parent_conn.recv()
                except EOFError:
                    res = None
                break
            if not p.is_alive():
                if parent_conn.poll(0.05):
                    try:
                        res = parent_conn.recv()
                    except EOFError:
                        res = None
                break

        if res is None and parent_conn.poll(0.05):
            try:
                res = parent_conn.recv()
            except EOFError:
                res = None

        if res is None:
            if p.is_alive():
                p.terminate()
            res = [False] * len(tests)
    finally:
        try:
            p.join(timeout=0.5)
        except Exception:
            pass
        try:
            parent_conn.close()
        except Exception:
            pass

    return [bool(x) for x in res]


def evaluate_code_function(code: str, tests: list[str], timeout: int = 1) -> list[bool]:
    """Evaluate function-based code with test cases"""
    return _check_snippet_many(code, tests, timeout)


def worker_stdio(script, input_val, output_queue):
    input_lines = iter(input_val.splitlines())

    def fake_input(prompt=""):
        try:
            return next(input_lines)
        except StopIteration:
            raise EOFError("No more input")

    stdout_capture = io.StringIO()
    original_stdout = sys.stdout
    original_stdin = sys.stdin
    sys.stdout = stdout_capture
    sys.stdin = io.StringIO(input_val)

    context = {
        "__name__": "__main__",
        "input": fake_input,
    }

    try:
        exec(script, context)
        printed_output = stdout_capture.getvalue()
        output_queue.put(printed_output)
    except SystemExit:
        printed_output = stdout_capture.getvalue()
        output_queue.put(printed_output)
    except Exception as e:
        output_queue.put(f"error: {e}")
    finally:
        sys.stdout = original_stdout
        sys.stdin = original_stdin


def evaluate_code_stdio(code: str, test_input: str, expected_output: str, timeout: int = 1) -> bool:
    """Evaluate stdio-based code"""
    q = mp.Queue()
    p = mp.Process(target=worker_stdio, args=(code, test_input, q))
    p.start()
    deadline = time.time() + timeout

    while p.is_alive() and time.time() < deadline:
        time.sleep(0.001)

    if p.is_alive():
        p.terminate()
        return False

    try:
        result = q.get_nowait()
        return " ".join(result.split()) == " ".join(expected_output.split())
    except Exception:
        return False
