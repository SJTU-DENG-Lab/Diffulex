from __future__ import annotations

import multiprocessing as mp
import os
import tempfile

import uvicorn

from diffulex.server.api_server import create_app
from diffulex.server.args import ServerArgs, parse_args
from diffulex.server.backend_worker import run_sync_backend_worker
from diffulex.server.frontend import FrontendManager


def default_ipc_addr(args: ServerArgs, name: str) -> str:
    path = os.path.join(tempfile.gettempdir(), f"diffulex-{os.getpid()}-{args.port}-{name}.ipc")
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    return f"ipc://{path}"


def resolve_zmq_addrs(args: ServerArgs) -> tuple[str, str]:
    return (
        args.zmq_command_addr or default_ipc_addr(args, "commands"),
        args.zmq_event_addr or default_ipc_addr(args, "events"),
    )


def start_backend(args: ServerArgs, command_addr: str, event_addr: str):
    ctx = mp.get_context("spawn")
    ready_queue = ctx.Queue()
    process = ctx.Process(
        target=run_sync_backend_worker,
        kwargs={
            "model": args.model,
            "engine_kwargs": args.engine_kwargs(),
            "command_addr": command_addr,
            "event_addr": event_addr,
            "ready_queue": ready_queue,
        },
        daemon=False,
        name="diffulex-sync-backend",
    )
    process.start()
    ready = ready_queue.get()
    if isinstance(ready, dict) and "error" in ready:
        raise RuntimeError(f"Diffulex sync backend failed to start: {ready['error']}")
    return process


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    command_addr, event_addr = resolve_zmq_addrs(args)
    frontend = FrontendManager.from_zmq(
        model_id=args.model_name or args.model,
        command_addr=command_addr,
        event_addr=event_addr,
    )
    backend_process = start_backend(args, command_addr, event_addr)
    app = create_app(frontend)

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    finally:
        backend_process.join(timeout=5)
        if backend_process.is_alive():
            backend_process.terminate()
            backend_process.join(timeout=5)


if __name__ == "__main__":
    main()
