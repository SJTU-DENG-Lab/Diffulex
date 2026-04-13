from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from diffulex.logger import get_logger
from diffulex.server.protocol import (
    ServingAbort,
    ServingCommand,
    ServingError,
    ServingEvent,
    ServingGenerate,
    ServingReply,
    ServingShutdown,
    serving_command_to_dict,
    serving_event_from_dict,
)
from diffulex.server.zmq_queue import ZmqAsyncPullQueue, ZmqAsyncPushQueue

logger = get_logger(__name__)


class ClientDisconnected(RuntimeError):
    pass


@dataclass
class FrontendReqState:
    rid: str
    events: list[ServingEvent] = field(default_factory=list)
    event: asyncio.Event = field(default_factory=asyncio.Event)


class FrontendManager:
    def __init__(
        self,
        *,
        model_id: str,
        send_backend,
        recv_backend,
        request_state_wait_timeout_s: float = 0.5,
    ) -> None:
        self.model_id = model_id
        self.send_backend = send_backend
        self.recv_backend = recv_backend
        self.request_state_wait_timeout_s = request_state_wait_timeout_s
        self.rid_to_state: dict[str, FrontendReqState] = {}
        self._listen_task: asyncio.Task | None = None

    @classmethod
    def from_zmq(cls, *, model_id: str, command_addr: str, event_addr: str) -> "FrontendManager":
        return cls(
            model_id=model_id,
            send_backend=ZmqAsyncPushQueue(command_addr, create=True, encoder=serving_command_to_dict),
            recv_backend=ZmqAsyncPullQueue(event_addr, create=True, decoder=serving_event_from_dict),
        )

    async def start(self) -> None:
        self._create_listener_once()

    async def stop(self) -> None:
        try:
            await self.send_backend.put(ServingShutdown())
        except Exception:
            logger.debug("Failed to send backend shutdown command", exc_info=True)

        if self._listen_task is not None:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None

        for queue in (self.send_backend, self.recv_backend):
            stop = getattr(queue, "stop", None)
            if stop is not None:
                stop()

    def _create_listener_once(self) -> None:
        if self._listen_task is None or self._listen_task.done():
            self._listen_task = asyncio.create_task(self.listen(), name="diffulex-frontend-listen")

    def new_request_id(self) -> str:
        return f"diffulex-{uuid.uuid4().hex}"

    def add_request_state(self, rid: str) -> FrontendReqState:
        if rid in self.rid_to_state:
            raise ValueError(f"Request id already exists: {rid}")
        state = FrontendReqState(rid=rid)
        self.rid_to_state[rid] = state
        return state

    def discard_request_state(self, rid: str) -> None:
        self.rid_to_state.pop(rid, None)

    async def listen(self) -> None:
        while True:
            event = await self.recv_backend.get()
            state = self.rid_to_state.get(event.rid)
            if state is None:
                logger.debug("Received event for unknown rid=%s", event.rid)
                continue
            state.events.append(event)
            state.event.set()

    async def send_one(self, command: ServingCommand) -> None:
        self._create_listener_once()
        await self.send_backend.put(command)

    async def generate(
        self,
        command: ServingGenerate,
        *,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> ServingReply:
        async for event in self.generate_stream(command, is_disconnected=is_disconnected):
            if isinstance(event, ServingError):
                raise RuntimeError(event.message)
            if isinstance(event, ServingReply):
                return event
        raise ClientDisconnected(f"Request {command.rid} disconnected before completion")

    async def generate_stream(
        self,
        command: ServingGenerate,
        *,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ):
        self.add_request_state(command.rid)
        completed = False
        try:
            await self.send_one(command)
            while True:
                event = await self.wait_for_event(command.rid, is_disconnected=is_disconnected)
                if isinstance(event, (ServingReply, ServingError)):
                    completed = True
                yield event
                if completed:
                    break
        except ClientDisconnected:
            return
        finally:
            if not completed:
                try:
                    await self.abort_request(command.rid)
                except Exception:
                    logger.debug("Failed to abort rid=%s", command.rid, exc_info=True)
            self.discard_request_state(command.rid)

    async def wait_for_event(
        self,
        rid: str,
        *,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> ServingEvent:
        state = self.rid_to_state[rid]
        while True:
            if state.events:
                return state.events.pop(0)
            try:
                await asyncio.wait_for(state.event.wait(), timeout=self.request_state_wait_timeout_s)
            except asyncio.TimeoutError:
                if is_disconnected is not None and await is_disconnected():
                    raise ClientDisconnected(f"Request {rid} disconnected from client side")
                continue
            state.event.clear()

    async def abort_request(self, rid: str) -> None:
        await self.send_one(ServingAbort(rid=rid))
