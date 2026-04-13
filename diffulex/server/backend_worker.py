from __future__ import annotations

from typing import Any, Callable

from diffulex.logger import get_logger
from diffulex.server.protocol import (
    ServingCommand,
    ServingEvent,
    ServingShutdown,
    serving_command_from_dict,
    serving_event_to_dict,
)
from diffulex.server.zmq_queue import ZmqPullQueue, ZmqPushQueue

logger = get_logger(__name__)


def default_engine_factory(model: str, **engine_kwargs):
    from diffulex import strategy as _strategy  # noqa: F401
    from diffulex.engine.tp_worker import DiffulexTPWorker

    return DiffulexTPWorker(model, **engine_kwargs)


class SyncBackendWorker:
    def __init__(
        self,
        *,
        model: str,
        engine_kwargs: dict[str, Any] | None,
        recv_frontend,
        send_frontend,
        engine_factory: Callable[..., Any] | None = None,
        ready_queue=None,
    ) -> None:
        self.model = model
        self.engine_kwargs = dict(engine_kwargs or {})
        self.recv_frontend = recv_frontend
        self.send_frontend = send_frontend
        self.engine_factory = engine_factory or default_engine_factory
        self.ready_queue = ready_queue
        self.engine = None
        self.shutdown_requested = False

    @classmethod
    def from_zmq(
        cls,
        *,
        model: str,
        engine_kwargs: dict[str, Any] | None,
        command_addr: str,
        event_addr: str,
        ready_queue=None,
    ) -> "SyncBackendWorker":
        return cls(
            model=model,
            engine_kwargs=engine_kwargs,
            recv_frontend=ZmqPullQueue(command_addr, create=False, decoder=serving_command_from_dict),
            send_frontend=ZmqPushQueue(event_addr, create=False, encoder=serving_event_to_dict),
            ready_queue=ready_queue,
        )

    def init_engine(self) -> None:
        if self.engine is not None:
            return
        self.engine = self.engine_factory(self.model, **self.engine_kwargs)
        if self.ready_queue is not None:
            self.ready_queue.put("SyncBackendWorker is ready")

    def run_forever(self) -> None:
        self.init_engine()
        try:
            while not self.shutdown_requested:
                self.normal_loop()
        finally:
            self.shutdown()

    def normal_loop(self) -> None:
        assert self.engine is not None
        commands = self.receive_commands(blocking=self.engine.is_finished())
        commands = self.process_input_commands(commands)
        if commands or not self.engine.is_finished():
            events = self.engine.run_serving_tick(commands)
            self.send_result(events)

    def receive_commands(self, *, blocking: bool) -> list[ServingCommand]:
        commands: list[ServingCommand] = []
        if blocking:
            self.run_when_idle()
            commands.append(self.recv_frontend.get())
        while not self.recv_frontend.empty():
            commands.append(self.recv_frontend.get())
        return commands

    def process_input_commands(self, commands: list[ServingCommand]) -> list[ServingCommand]:
        input_commands: list[ServingCommand] = []
        for command in commands:
            if isinstance(command, ServingShutdown):
                self.shutdown_requested = True
            else:
                input_commands.append(command)
        return input_commands

    def send_result(self, events: list[ServingEvent]) -> None:
        for event in events:
            self.send_frontend.put(event)

    def run_when_idle(self) -> None:
        logger.info("SyncBackendWorker is idle, waiting for new requests...")

    def shutdown(self) -> None:
        if self.engine is not None:
            try:
                self.engine.exit()
            finally:
                self.engine = None
        for queue in (self.recv_frontend, self.send_frontend):
            stop = getattr(queue, "stop", None)
            if stop is not None:
                stop()


def run_sync_backend_worker(
    *,
    model: str,
    engine_kwargs: dict[str, Any] | None,
    command_addr: str,
    event_addr: str,
    ready_queue=None,
) -> None:
    try:
        worker = SyncBackendWorker.from_zmq(
            model=model,
            engine_kwargs=engine_kwargs,
            command_addr=command_addr,
            event_addr=event_addr,
            ready_queue=ready_queue,
        )
        worker.run_forever()
    except Exception as exc:
        if ready_queue is not None:
            ready_queue.put({"error": repr(exc)})
        raise
