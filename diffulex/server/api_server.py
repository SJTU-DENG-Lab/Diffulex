from __future__ import annotations

import inspect
import json
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from diffulex.sampling_params import SamplingParams
from diffulex.server.frontend import ClientDisconnected, FrontendManager
from diffulex.server.protocol import (
    ChatInput,
    PromptInput,
    ServingBufferSnapshot,
    ServingDelta,
    ServingError,
    ServingGenerate,
    ServingReply,
)


class GenerateRequest(BaseModel):
    prompt: str | list[int]
    max_tokens: int = 512
    temperature: float = 1.0
    max_nfe: int | None = 512
    max_repetition_run: int | None = None
    ignore_eos: bool = False
    stream: bool = False
    stream_mode: Literal["block_append", "denoise"] = "denoise"
    user: str | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    model: str | None = None
    max_tokens: int = 512
    temperature: float = 1.0
    max_nfe: int | None = 512
    max_repetition_run: int | None = None
    ignore_eos: bool = False
    stream: bool = False
    stream_mode: Literal["block_append", "denoise"] = "denoise"
    user: str | None = None


def sampling_params_from_request(
    payload: GenerateRequest | ChatCompletionRequest,
) -> SamplingParams:
    return SamplingParams(
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        max_nfe=payload.max_nfe,
        max_repetition_run=payload.max_repetition_run,
        ignore_eos=payload.ignore_eos,
    )


def sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


def create_app(frontend: FrontendManager) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        start = getattr(frontend, "start", None)
        if start is not None:
            await maybe_await(start())
        try:
            yield
        finally:
            stop = getattr(frontend, "stop", None)
            if stop is not None:
                await maybe_await(stop())

    app = FastAPI(title="Diffulex HTTP Server", lifespan=lifespan)

    @app.post("/generate", response_model=None)
    async def generate(payload: GenerateRequest, raw_request: Request) -> dict[str, Any] | StreamingResponse:
        rid = frontend.new_request_id()
        command = ServingGenerate(
            rid=rid,
            input=PromptInput(payload.prompt),
            sampling_params=sampling_params_from_request(payload),
            stream=payload.stream,
            stream_mode=payload.stream_mode,
            user=payload.user,
            created_time=time.time(),
        )
        if payload.stream:
            return StreamingResponse(generate_stream(command, raw_request), media_type="text/event-stream")

        try:
            result = await frontend.generate(command, is_disconnected=raw_request.is_disconnected)
        except ClientDisconnected as exc:
            raise HTTPException(status_code=499, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return result.to_dict()

    async def generate_stream(command: ServingGenerate, raw_request: Request) -> AsyncIterator[str]:
        try:
            async for event in frontend.generate_stream(command, is_disconnected=raw_request.is_disconnected):
                if isinstance(event, ServingError):
                    yield sse({"event": "error", "id": event.rid, "message": event.message})
                    break
                if isinstance(event, (ServingDelta, ServingBufferSnapshot, ServingReply)):
                    event_payload = event.to_dict()
                    if isinstance(event, ServingReply):
                        event_payload["event"] = "final"
                    yield sse(event_payload)
        except RuntimeError as exc:
            yield sse({"event": "error", "message": str(exc)})
        yield "data: [DONE]\n\n"

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": frontend.model_id,
                    "object": "model",
                    "owned_by": "diffulex",
                }
            ],
        }

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        payload: ChatCompletionRequest, raw_request: Request
    ) -> dict[str, Any] | StreamingResponse:
        if not payload.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")

        messages = [
            message.model_dump() if hasattr(message, "model_dump") else message.dict()
            for message in payload.messages
        ]
        rid = frontend.new_request_id()
        command = ServingGenerate(
            rid=rid,
            input=ChatInput(messages),
            sampling_params=sampling_params_from_request(payload),
            stream=payload.stream,
            stream_mode=payload.stream_mode,
            user=payload.user,
            created_time=time.time(),
        )
        if payload.stream:
            return StreamingResponse(
                chat_completion_stream(payload, command, raw_request),
                media_type="text/event-stream",
            )

        try:
            result = await frontend.generate(command, is_disconnected=raw_request.is_disconnected)
        except ClientDisconnected as exc:
            raise HTTPException(status_code=499, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "id": f"chatcmpl-{result.rid}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.model or frontend.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": result.finish_reason or "stop",
                }
            ],
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": len(result.token_ids),
                "total_tokens": None,
            },
        }

    async def chat_completion_stream(
        payload: ChatCompletionRequest, command: ServingGenerate, raw_request: Request
    ) -> AsyncIterator[str]:
        try:
            async for event in frontend.generate_stream(command, is_disconnected=raw_request.is_disconnected):
                if isinstance(event, ServingError):
                    yield sse({"error": {"message": event.message, "type": "server_error", "code": 500}})
                    break
                if payload.stream_mode == "denoise":
                    yield sse(denoise_chat_event(event, payload, frontend.model_id))
                elif isinstance(event, ServingDelta):
                    if event.text:
                        yield sse(chat_delta_chunk(event, payload, frontend.model_id))
                elif isinstance(event, ServingReply):
                    yield sse(chat_finish_chunk(event, payload, frontend.model_id))
        except RuntimeError as exc:
            yield sse({"error": {"message": str(exc), "type": "server_error", "code": 503}})
        yield "data: [DONE]\n\n"

    return app


def chat_delta_chunk(event: ServingDelta, request: ChatCompletionRequest, model_id: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{event.rid}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request.model or model_id,
        "choices": [
            {
                "index": 0,
                "delta": {"content": event.text},
                "finish_reason": None,
            }
        ],
    }


def chat_finish_chunk(event: ServingReply, request: ChatCompletionRequest, model_id: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{event.rid}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request.model or model_id,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": event.finish_reason or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": len(event.token_ids),
            "total_tokens": None,
        },
    }


def denoise_chat_event(
    event: ServingDelta | ServingBufferSnapshot | ServingReply,
    request: ChatCompletionRequest,
    model_id: str,
) -> dict[str, Any]:
    payload = event.to_dict()
    payload.update(
        {
            "id": f"chatcmpl-{event.rid}",
            "object": "diffulex.chat.completion.denoise",
            "created": int(time.time()),
            "model": request.model or model_id,
        }
    )
    if isinstance(event, ServingReply):
        payload["event"] = "final"
    return payload
