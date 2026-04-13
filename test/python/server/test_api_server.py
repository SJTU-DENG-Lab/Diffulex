from __future__ import annotations

from fastapi.testclient import TestClient

from diffulex.server.api_server import create_app
from diffulex.server.protocol import ChatInput, PromptInput, ServingBufferSnapshot, ServingDelta, ServingReply


class _Frontend:
    model_id = "fake-model"

    def __init__(self):
        self.started = False
        self.stopped = False
        self.commands = []
        self._next_id = 0

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    def new_request_id(self):
        self._next_id += 1
        return f"req-{self._next_id}"

    async def generate(self, command, *, is_disconnected=None):
        self.commands.append(command)
        return ServingReply(
            rid=command.rid,
            text="ok",
            token_ids=[1, 2],
            nfe=3,
            finish_reason="stop",
            full_text="ok",
            full_token_ids=[1, 2],
        )

    async def generate_stream(self, command, *, is_disconnected=None):
        self.commands.append(command)
        if command.stream_mode == "denoise":
            yield ServingBufferSnapshot(
                rid=command.rid,
                token_offset=0,
                absolute_start=4,
                absolute_end=6,
                text="draft",
                token_ids=[7, 8],
                nfe=1,
            )
        else:
            yield ServingDelta(
                rid=command.rid,
                token_offset=0,
                text="ok",
                token_ids=[1, 2],
                nfe=1,
            )
        yield ServingReply(
            rid=command.rid,
            text="ok",
            token_ids=[1, 2],
            nfe=3,
            finish_reason="stop",
            full_text="ok",
            full_token_ids=[1, 2],
        )


def test_generate_route_and_shutdown():
    frontend = _Frontend()
    with TestClient(create_app(frontend)) as client:
        response = client.post("/generate", json={"prompt": "hello", "max_tokens": 7, "temperature": 0.0})
        assert response.status_code == 200
        assert response.json()["text"] == "ok"
        assert frontend.started is True

    assert frontend.stopped is True
    assert isinstance(frontend.commands[0].input, PromptInput)
    assert frontend.commands[0].input.prompt == "hello"
    assert frontend.commands[0].sampling_params.max_tokens == 7
    assert frontend.commands[0].sampling_params.max_nfe == 512


def test_models_and_chat_routes():
    frontend = _Frontend()
    with TestClient(create_app(frontend)) as client:
        models = client.get("/v1/models")
        assert models.status_code == 200
        assert models.json()["data"][0]["id"] == "fake-model"

        response = client.post(
            "/v1/chat/completions",
            json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["id"] == "chatcmpl-req-1"
        assert payload["choices"][0]["message"]["content"] == "ok"
        assert isinstance(frontend.commands[-1].input, ChatInput)
        assert frontend.commands[-1].input.messages == [{"role": "user", "content": "hi"}]
        assert frontend.commands[-1].sampling_params.max_tokens == 512


def test_streaming_chat_block_append_uses_openai_chunks():
    frontend = _Frontend()
    with TestClient(create_app(frontend)) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "stream_mode": "block_append",
            },
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert '"object": "chat.completion.chunk"' in body
    assert '"content": "ok"' in body
    assert "data: [DONE]" in body


def test_streaming_chat_defaults_to_denoise():
    frontend = _Frontend()
    with TestClient(create_app(frontend)) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert '"object": "diffulex.chat.completion.denoise"' in body
    assert '"event": "buffer_snapshot"' in body
    assert frontend.commands[0].stream_mode == "denoise"


def test_streaming_generate_denoise_uses_buffer_snapshots():
    frontend = _Frontend()
    with TestClient(create_app(frontend)) as client:
        with client.stream(
            "POST",
            "/generate",
            json={"prompt": "hello", "stream": True, "stream_mode": "denoise"},
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert '"event": "buffer_snapshot"' in body
    assert '"absolute_start": 4' in body
    assert '"event": "final"' in body
    assert frontend.commands[0].stream_mode == "denoise"


def test_streaming_chat_denoise_uses_buffer_snapshots():
    frontend = _Frontend()
    with TestClient(create_app(frontend)) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "stream_mode": "denoise",
            },
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert '"object": "diffulex.chat.completion.denoise"' in body
    assert '"event": "buffer_snapshot"' in body
    assert '"token_offset": 0' in body
    assert '"event": "final"' in body
