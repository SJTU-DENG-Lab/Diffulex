from __future__ import annotations

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, AsyncIterator, Awaitable, Callable

from diffulex.sampling_params import SamplingParams
from diffulex.server.protocol import PromptInput
from diffulex.mixin.async_engine.engine.serving_worker import (
    ServingAbort,
    ServingCommand,
    ServingError,
    ServingEvent,
    ServingGenerate,
    ServingReply,
)

GenerateResult = ServingReply


@dataclass
class QueuedCommand:
    command: ServingCommand


def default_engine_factory(model: str, **engine_kwargs):
    from diffulex import strategy as _strategy  # noqa: F401
    from diffulex.engine.tp_worker import DiffulexTPWorker

    return DiffulexTPWorker(model, **engine_kwargs)


class EngineLoop:
    """Single-owner background loop for online serving.

    The FastAPI event loop owns admission futures, while all engine calls run through
    a one-worker executor. This keeps DiffulexTPWorker mutations serialized without
    blocking request admission while a model step is running.
    """

    def __init__(
        self,
        model: str,
        engine_kwargs: dict[str, Any] | None = None,
        *,
        engine_factory: Callable[..., Any] | None = None,
        idle_sleep_s: float = 0.001,
    ) -> None:
        self.model = model
        self.engine_kwargs = dict(engine_kwargs or {})
        if self.engine_kwargs.get("data_parallel_size", 1) != 1:
            raise ValueError("Phase 1 HTTP serving supports data_parallel_size=1 only")
        self.engine_factory = engine_factory or default_engine_factory
        self.idle_sleep_s = idle_sleep_s

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="diffulex-engine")
        self._queue: asyncio.Queue[QueuedCommand] | None = None
        self._task: asyncio.Task | None = None
        self._started: asyncio.Future[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._engine = None
        self._futures: dict[str, asyncio.Future[GenerateResult]] = {}
        self._stream_queues: dict[str, asyncio.Queue[ServingEvent | None]] = {}
        self._engine_has_work = False

    @property
    def model_id(self) -> str:
        return self.engine_kwargs.get("model_name") or self.model

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.is_running:
            if self._started is not None and not self._started.done():
                await self._started
            return
        self._queue = asyncio.Queue()
        self._stop_event = asyncio.Event()
        self._started = asyncio.get_running_loop().create_future()
        self._task = asyncio.create_task(self.run_loop(), name="diffulex-engine-loop")
        await self._started

    async def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task is not None:
            await self._task
        self._executor.shutdown(wait=True)

    async def generate(self, prompt: str | list[int], sampling_params: SamplingParams) -> GenerateResult:
        if self._queue is None or self._task is None:
            raise RuntimeError("EngineLoop.start() must be called before generate()")
        if self._task.done():
            raise RuntimeError("EngineLoop stopped before generate()")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        request_id = f"diffulex-{uuid.uuid4().hex}"
        self._futures[request_id] = future
        queued = QueuedCommand(
            command=ServingGenerate(
                rid=request_id,
                input=PromptInput(prompt),
                sampling_params=sampling_params,
            ),
        )
        await self._queue.put(queued)
        return await future

    async def generate_stream(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        *,
        stream_mode: str = "denoise",
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncIterator[ServingEvent]:
        if self._queue is None or self._task is None:
            raise RuntimeError("EngineLoop.start() must be called before generate_stream()")
        if self._task.done():
            raise RuntimeError("EngineLoop stopped before generate_stream()")

        request_id = f"diffulex-{uuid.uuid4().hex}"
        stream_queue: asyncio.Queue[ServingEvent | None] = asyncio.Queue()
        self._stream_queues[request_id] = stream_queue
        await self._queue.put(
            QueuedCommand(
                command=ServingGenerate(
                    rid=request_id,
                    input=PromptInput(prompt),
                    sampling_params=sampling_params,
                    stream=True,
                    stream_mode=stream_mode,
                ),
            )
        )

        completed = False
        try:
            while True:
                try:
                    event = await asyncio.wait_for(stream_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if is_disconnected is not None and await is_disconnected():
                        break
                    continue
                if event is None:
                    break
                if isinstance(event, (ServingReply, ServingError)):
                    completed = True
                yield event
                if completed:
                    break
        finally:
            self._stream_queues.pop(request_id, None)
            if not completed and self._queue is not None:
                await self._queue.put(QueuedCommand(command=ServingAbort(request_id)))

    async def render_chat_prompt(self, messages: list[dict[str, str]]) -> str:
        if self._task is not None and self._task.done():
            raise RuntimeError("EngineLoop stopped before render_chat_prompt()")

        def render() -> str:
            if self._engine is None:
                raise RuntimeError("EngineLoop.start() must complete before render_chat_prompt()")
            return self._engine.render_chat_prompt_for_serving(messages)

        return await self.call_engine(render)

    async def run_loop(self) -> None:
        try:
            self._engine = await self.call_engine(lambda: self.engine_factory(self.model, **self.engine_kwargs))
            self.mark_started()
            await self.serve()
        except Exception as exc:
            self.mark_started(exc)
            raise
        finally:
            self.fail_pending(RuntimeError("EngineLoop stopped before request completion"))
            self.discard_queued_commands()
            if self._engine is not None:
                await self.call_engine(self._engine.exit)
            self._engine = None

    def mark_started(self, exc: Exception | None = None) -> None:
        if self._started is None or self._started.done():
            return
        if exc is None:
            self._started.set_result(None)
        else:
            self._started.set_exception(exc)

    async def serve(self) -> None:
        assert self._queue is not None
        assert self._stop_event is not None

        while not self._stop_event.is_set():
            commands = await self.collect_ready_commands(
                wait_for_one=not self.has_waiters() and not self._engine_has_work,
            )
            if commands or self.has_waiters() or self._engine_has_work:
                events = await self.call_engine(self._engine.run_serving_tick, commands)
                self._engine_has_work = await self.call_engine(lambda: not self._engine.is_finished())
                self.resolve_events(events)
                await asyncio.sleep(0)
            elif self.idle_sleep_s > 0:
                await asyncio.sleep(self.idle_sleep_s)

    async def collect_ready_commands(self, *, wait_for_one: bool) -> list[ServingCommand]:
        assert self._queue is not None
        assert self._stop_event is not None

        queued_commands: list[QueuedCommand] = []
        if wait_for_one:
            get_task = asyncio.create_task(self._queue.get())
            stop_task = asyncio.create_task(self._stop_event.wait())
            done, pending = await asyncio.wait({get_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            if stop_task in done and self._stop_event.is_set():
                if not get_task.done():
                    return []
            if get_task in done:
                queued_commands.append(get_task.result())

        while True:
            try:
                queued_commands.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        return [queued.command for queued in queued_commands if self.should_admit(queued.command)]

    def should_admit(self, command: ServingCommand) -> bool:
        if isinstance(command, ServingGenerate):
            future = self._futures.get(command.request_id)
            stream_queue = self._stream_queues.get(command.request_id)
            if future is None and stream_queue is None:
                self._futures.pop(command.request_id, None)
                return False
            if future is not None and future.cancelled():
                self._futures.pop(command.request_id, None)
                return False
        return True

    def resolve_events(self, events: list[ServingEvent]) -> None:
        for event in events:
            stream_queue = self._stream_queues.get(event.rid)
            if stream_queue is not None:
                stream_queue.put_nowait(event)
                if isinstance(event, (ServingReply, ServingError)):
                    self._stream_queues.pop(event.rid, None)
                    stream_queue.put_nowait(None)
                continue

            future = self._futures.pop(event.rid, None)
            if future is None or future.done():
                continue
            if isinstance(event, ServingError):
                future.set_exception(RuntimeError(event.message))
            elif isinstance(event, ServingReply):
                future.set_result(event)

    def has_waiters(self) -> bool:
        return bool(self._futures or self._stream_queues)

    async def call_engine(self, fn: Callable, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, partial(fn, *args, **kwargs))

    def discard_queued_commands(self) -> None:
        if self._queue is None:
            return
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    def fail_pending(self, exc: Exception) -> None:
        for future in list(self._futures.values()):
            if not future.done():
                future.set_exception(exc)
        self._futures.clear()
        for request_id, stream_queue in list(self._stream_queues.items()):
            stream_queue.put_nowait(ServingError(request_id, str(exc)))
            stream_queue.put_nowait(None)
        self._stream_queues.clear()
