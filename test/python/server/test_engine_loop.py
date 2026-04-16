from __future__ import annotations

import asyncio

from diffulex.sampling_params import SamplingParams
from diffulex.server.protocol import PromptInput
from diffulex.mixin.async_engine.engine.serving_worker import (
    DiffulexServingWorkerMixin,
    ServingBufferSnapshot,
    ServingDelta,
    ServingGenerate,
    ServingReply,
)
from diffulex.server.engine_loop import EngineLoop


class _Tokenizer:
    eos_token = "<eos>"

    def decode(self, token_ids, skip_special_tokens=False):
        return " ".join(str(token_id) for token_id in token_ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        rendered = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        return rendered + "\nassistant:"


class _Req:
    def __init__(self, req_id: int, token_id: int):
        self.req_id = req_id
        self._token_id = token_id
        self.is_multi_block = False
        self.num_prompt_tokens = 0
        self.mask_token_id = -1
        self.is_finished = False
        self.nfe = 0
        self.completion_reason = "stop"

    @property
    def truncated_response(self):
        return [self._token_id]

    @property
    def full_response(self):
        return [self._token_id]


class _Engine(DiffulexServingWorkerMixin):
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.added = []
        self.pending = []
        self.aborted = []
        self.exited = False
        self._next_id = 0

    def add_request(self, prompt, sampling_params):
        req_id = self._next_id
        self._next_id += 1
        req = _Req(req_id, token_id=100 + req_id)
        self.added.append((prompt, sampling_params))
        self.pending.append(req)
        return req_id

    def step(self):
        reqs = list(self.pending)
        self.pending.clear()
        for req in reqs:
            req.is_finished = True
            req.nfe = 1
        return reqs, False

    def is_finished(self):
        return not self.pending

    def abort_request(self, req_id: int):
        self.aborted.append(req_id)
        self.pending = [req for req in self.pending if req.req_id != req_id]
        return True

    def exit(self):
        self.exited = True


class _Block:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end


class _BlockBuffer:
    def __init__(self, start: int, end: int):
        self.first_running_block = _Block(start, start + 2)
        self.last_running_block = _Block(end - 2, end)


class _StreamingReq:
    def __init__(self, req_id: int):
        self.req_id = req_id
        self.is_multi_block = True
        self.prefix_len = 0
        self.mask_token_id = 0
        self.token_ids = [0, 0, 0, 0, 0, 0]
        self.dllm_block_buffer = _BlockBuffer(0, 4)
        self.is_finished = False
        self.nfe = 0
        self.completion_reason = "stop"

    @property
    def truncated_response(self):
        return self.token_ids[:4]

    @property
    def full_response(self):
        return self.token_ids[:4]


class _StreamingEngine(DiffulexServingWorkerMixin):
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.pending = []
        self.step_count = 0
        self.aborted = []

    def add_request(self, prompt, sampling_params):
        req = _StreamingReq(0)
        self.pending.append(req)
        return req.req_id

    def step(self):
        req = self.pending[0]
        self.step_count += 1
        req.nfe = self.step_count
        if self.step_count == 1:
            req.token_ids = [11, 12, 0, 0, 0, 0]
            req.dllm_block_buffer = _BlockBuffer(0, 4)
        elif self.step_count == 2:
            req.token_ids = [11, 12, 13, 14, 0, 0]
            req.dllm_block_buffer = _BlockBuffer(2, 6)
        else:
            req.token_ids = [11, 12, 13, 14, 0, 0]
            req.dllm_block_buffer = _BlockBuffer(4, 6)
            req.is_finished = True
            self.pending.clear()
        return [req], False

    def is_finished(self):
        return not self.pending

    def abort_request(self, req_id: int):
        self.aborted.append(req_id)
        self.pending = [req for req in self.pending if req.req_id != req_id]
        return True

    def exit(self):
        pass


class _MaskFinishedEngine(_StreamingEngine):
    def step(self):
        req = self.pending[0]
        req.token_ids = [11, 0, 13, 14, 0, 0]
        req.is_finished = True
        self.pending.clear()
        return [req], False


class _NoEventStreamingEngine(_StreamingEngine):
    def step(self):
        req = self.pending[0]
        self.step_count += 1
        req.nfe = self.step_count
        req.token_ids = [0, 0, 0, 0, 0, 0]
        req.dllm_block_buffer = _BlockBuffer(0, 4)
        return [req], False


def test_serving_worker_tick_returns_finished_replies():
    engine = _Engine()
    events = engine.run_serving_tick(
        [
            ServingGenerate(
                rid="request-1",
                input=PromptInput("hello"),
                sampling_params=SamplingParams(max_tokens=4),
            )
        ]
    )

    assert len(events) == 1
    assert events[0].request_id == "request-1"
    assert events[0].token_ids == [100]
    assert engine.added[0][0] == "hello"


def test_serving_worker_block_append_streams_only_stable_tokens():
    engine = _StreamingEngine()
    first = engine.run_serving_tick(
        [
            ServingGenerate(
                rid="request-1",
                input=PromptInput("hello"),
                sampling_params=SamplingParams(max_tokens=4),
                stream=True,
                stream_mode="block_append",
            )
        ]
    )
    second = engine.run_serving_tick([])
    third = engine.run_serving_tick([])

    assert first == []
    assert [type(event) for event in second] == [ServingDelta]
    assert second[0].token_ids == [11, 12]
    assert [type(event) for event in third] == [ServingDelta, ServingReply]
    assert third[0].token_ids == [13, 14]
    assert third[1].token_ids == [11, 12, 13, 14]


def test_serving_worker_block_append_does_not_emit_mask_tokens():
    engine = _MaskFinishedEngine()
    events = engine.run_serving_tick(
        [
            ServingGenerate(
                rid="request-1",
                input=PromptInput("hello"),
                sampling_params=SamplingParams(max_tokens=4),
                stream=True,
                stream_mode="block_append",
            )
        ]
    )

    assert [type(event) for event in events] == [ServingDelta, ServingReply]
    assert events[0].token_ids == [11]
    assert events[0].text == "11"
    assert events[1].token_ids == [11, 13, 14]
    assert events[1].text == "11 13 14"


def test_serving_worker_denoise_streams_buffer_snapshots():
    engine = _StreamingEngine()
    first = engine.run_serving_tick(
        [
            ServingGenerate(
                rid="request-1",
                input=PromptInput("hello"),
                sampling_params=SamplingParams(max_tokens=4),
                stream=True,
                stream_mode="denoise",
            )
        ]
    )
    second = engine.run_serving_tick([])

    assert [type(event) for event in first] == [ServingBufferSnapshot]
    assert first[0].token_offset == 0
    assert first[0].token_ids == [11, 12, 0, 0]
    assert [type(event) for event in second] == [ServingBufferSnapshot]
    assert second[0].token_offset == 0
    assert second[0].token_ids == [11, 12, 13, 14, 0, 0]


def test_engine_loop_streams_events():
    async def exercise_streaming_loop():
        engine = _Engine()
        loop = EngineLoop(
            "fake",
            {"model_name": "fake", "data_parallel_size": 1},
            engine_factory=lambda *_args, **_kwargs: engine,
        )
        await loop.start()

        events = []
        async for event in loop.generate_stream("hello", SamplingParams(max_tokens=4), stream_mode="block_append"):
            events.append(event)
        await loop.stop()

        assert [type(event) for event in events] == [ServingDelta, ServingReply]
        assert events[0].token_ids == [100]
        assert events[1].token_ids == [100]

    asyncio.run(exercise_streaming_loop())


def test_engine_loop_aborts_stream_when_generator_closes():
    async def exercise_streaming_abort():
        engine = _StreamingEngine()
        loop = EngineLoop(
            "fake",
            {"model_name": "fake", "data_parallel_size": 1},
            engine_factory=lambda *_args, **_kwargs: engine,
        )
        await loop.start()

        stream = loop.generate_stream("hello", SamplingParams(max_tokens=4), stream_mode="denoise")
        event = await stream.__anext__()
        assert isinstance(event, ServingBufferSnapshot)
        await stream.aclose()
        await asyncio.sleep(0.01)
        await loop.stop()

        assert engine.aborted == [0]
        assert engine.pending == []

    asyncio.run(exercise_streaming_abort())


def test_engine_loop_aborts_stream_when_client_disconnects_without_events():
    async def exercise_streaming_abort():
        engine = _NoEventStreamingEngine()
        loop = EngineLoop(
            "fake",
            {"model_name": "fake", "data_parallel_size": 1},
            engine_factory=lambda *_args, **_kwargs: engine,
        )
        await loop.start()

        async def is_disconnected():
            return True

        events = []
        async for event in loop.generate_stream(
            "hello",
            SamplingParams(max_tokens=4),
            stream_mode="block_append",
            is_disconnected=is_disconnected,
        ):
            events.append(event)
        await asyncio.sleep(0.01)
        await loop.stop()

        assert events == []
        assert engine.aborted == [0]
        assert engine.pending == []

    asyncio.run(exercise_streaming_abort())


def test_engine_loop_batches_concurrent_requests_and_exits():
    async def exercise_engine_loop():
        engine = _Engine()
        loop = EngineLoop(
            "fake",
            {"model_name": "fake", "data_parallel_size": 1},
            engine_factory=lambda *_args, **_kwargs: engine,
        )
        await loop.start()

        results = await asyncio.gather(
            loop.generate("hello", SamplingParams(max_tokens=4)),
            loop.generate("world", SamplingParams(max_tokens=4)),
        )
        await loop.stop()

        assert [result.token_ids for result in results] == [[100], [101]]
        assert [item[0] for item in engine.added] == ["hello", "world"]
        assert engine.exited is True

    asyncio.run(exercise_engine_loop())


def test_engine_loop_renders_chat_prompt():
    async def exercise_chat_prompt():
        engine = _Engine()
        loop = EngineLoop(
            "fake",
            {"model_name": "fake", "data_parallel_size": 1},
            engine_factory=lambda *_args, **_kwargs: engine,
        )
        await loop.start()

        prompt = await loop.render_chat_prompt([{"role": "user", "content": "hi"}])
        await loop.stop()

        assert prompt == "user: hi\nassistant:"

    asyncio.run(exercise_chat_prompt())
