from __future__ import annotations

import asyncio

from diffulex.sampling_params import SamplingParams
from diffulex.server.backend_worker import SyncBackendWorker
from diffulex.server.frontend import FrontendManager
from diffulex.server.protocol import PromptInput, ServingGenerate, ServingReply, ServingShutdown


class _SyncRecvQueue:
    def __init__(self, items):
        self.items = list(items)

    def get(self):
        return self.items.pop(0)

    def empty(self):
        return not self.items

    def stop(self):
        pass


class _SyncSendQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)

    def stop(self):
        pass


class _AsyncSendQueue:
    def __init__(self):
        self.items = []

    async def put(self, item):
        self.items.append(item)

    def stop(self):
        pass


class _AsyncRecvQueue:
    def __init__(self):
        self.queue = asyncio.Queue()

    async def put(self, item):
        await self.queue.put(item)

    async def get(self):
        return await self.queue.get()

    def stop(self):
        pass


class _Engine:
    def __init__(self):
        self.commands = []
        self.exited = False

    def is_finished(self):
        return True

    def run_serving_tick(self, commands):
        self.commands.extend(commands)
        return [
            ServingReply(
                rid=commands[0].rid,
                text="ok",
                token_ids=[1],
                nfe=1,
                finish_reason="stop",
            )
        ]

    def exit(self):
        self.exited = True


def test_sync_backend_worker_run_forever_uses_normal_loop_names():
    command = ServingGenerate(
        rid="rid-1",
        input=PromptInput("hello"),
        sampling_params=SamplingParams(max_tokens=4),
    )
    engine = _Engine()
    recv = _SyncRecvQueue([command, ServingShutdown()])
    send = _SyncSendQueue()
    worker = SyncBackendWorker(
        model="fake",
        engine_kwargs={},
        recv_frontend=recv,
        send_frontend=send,
        engine_factory=lambda *_args, **_kwargs: engine,
    )

    worker.run_forever()

    assert engine.commands == [command]
    assert send.items == [ServingReply(rid="rid-1", text="ok", token_ids=[1], nfe=1, finish_reason="stop")]
    assert engine.exited is True


def test_frontend_manager_generate_dispatches_by_rid():
    async def exercise():
        send = _AsyncSendQueue()
        recv = _AsyncRecvQueue()
        frontend = FrontendManager(model_id="fake", send_backend=send, recv_backend=recv)
        command = ServingGenerate(
            rid="rid-1",
            input=PromptInput("hello"),
            sampling_params=SamplingParams(max_tokens=4),
        )

        task = asyncio.create_task(frontend.generate(command))
        await asyncio.sleep(0)
        await recv.put(ServingReply(rid="rid-1", text="ok", token_ids=[1], nfe=1, finish_reason="stop"))
        result = await task
        await frontend.stop()

        assert result.text == "ok"
        assert send.items[0] == command
        assert isinstance(send.items[-1], ServingShutdown)
        assert frontend.rid_to_state == {}

    asyncio.run(exercise())
