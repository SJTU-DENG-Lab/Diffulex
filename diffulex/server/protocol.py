from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from diffulex.sampling_params import SamplingParams

StreamMode = Literal["block_append", "denoise"]


@dataclass
class PromptInput:
    prompt: str | list[int]


@dataclass
class ChatInput:
    messages: list[dict[str, str]]


ServingInput = PromptInput | ChatInput


@dataclass
class ServingGenerate:
    rid: str
    input: ServingInput
    sampling_params: SamplingParams
    stream: bool = False
    stream_mode: StreamMode = "denoise"
    user: str | None = None
    created_time: float | None = None

    @property
    def request_id(self) -> str:
        return self.rid


@dataclass
class ServingAbort:
    rid: str

    @property
    def request_id(self) -> str:
        return self.rid


@dataclass
class ServingShutdown:
    pass


ServingCommand = ServingGenerate | ServingAbort | ServingShutdown


@dataclass
class ServingReply:
    rid: str
    text: str
    token_ids: list[int]
    nfe: int
    finish_reason: str | None = None
    full_text: str | None = None
    full_token_ids: list[int] | None = None
    finished: bool = True

    @property
    def request_id(self) -> str:
        return self.rid

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.rid,
            "text": self.text,
            "token_ids": self.token_ids,
            "nfe": self.nfe,
            "finish_reason": self.finish_reason,
            "full_text": self.full_text if self.full_text is not None else self.text,
            "full_token_ids": self.full_token_ids if self.full_token_ids is not None else self.token_ids,
        }


@dataclass
class ServingDelta:
    rid: str
    token_offset: int
    text: str
    token_ids: list[int]
    nfe: int
    finished: bool = False

    @property
    def request_id(self) -> str:
        return self.rid

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": "append",
            "id": self.rid,
            "token_offset": self.token_offset,
            "text": self.text,
            "token_ids": self.token_ids,
            "nfe": self.nfe,
            "finished": self.finished,
        }


@dataclass
class ServingBufferSnapshot:
    rid: str
    token_offset: int
    absolute_start: int
    absolute_end: int
    text: str
    token_ids: list[int]
    nfe: int
    finished: bool = False

    @property
    def request_id(self) -> str:
        return self.rid

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": "buffer_snapshot",
            "id": self.rid,
            "token_offset": self.token_offset,
            "absolute_start": self.absolute_start,
            "absolute_end": self.absolute_end,
            "text": self.text,
            "token_ids": self.token_ids,
            "nfe": self.nfe,
            "finished": self.finished,
        }


@dataclass
class ServingError:
    rid: str
    message: str

    @property
    def request_id(self) -> str:
        return self.rid

    def to_dict(self) -> dict[str, Any]:
        return {"event": "error", "id": self.rid, "message": self.message}


ServingEvent = ServingReply | ServingDelta | ServingBufferSnapshot | ServingError


def sampling_params_to_dict(sampling_params: SamplingParams) -> dict[str, Any]:
    return asdict(sampling_params)


def sampling_params_from_dict(payload: dict[str, Any]) -> SamplingParams:
    return SamplingParams(**payload)


def serving_input_to_dict(input_: ServingInput) -> dict[str, Any]:
    if isinstance(input_, PromptInput):
        return {"type": "prompt", "prompt": input_.prompt}
    if isinstance(input_, ChatInput):
        return {"type": "chat", "messages": input_.messages}
    raise TypeError(f"Unsupported serving input: {type(input_)!r}")


def serving_input_from_dict(payload: dict[str, Any]) -> ServingInput:
    type_name = payload.get("type")
    if type_name == "prompt":
        return PromptInput(prompt=payload["prompt"])
    if type_name == "chat":
        return ChatInput(messages=payload["messages"])
    raise ValueError(f"Unsupported serving input type: {type_name!r}")


def serving_command_to_dict(command: ServingCommand) -> dict[str, Any]:
    if isinstance(command, ServingGenerate):
        return {
            "type": "generate",
            "rid": command.rid,
            "input": serving_input_to_dict(command.input),
            "sampling_params": sampling_params_to_dict(command.sampling_params),
            "stream": command.stream,
            "stream_mode": command.stream_mode,
            "user": command.user,
            "created_time": command.created_time,
        }
    if isinstance(command, ServingAbort):
        return {"type": "abort", "rid": command.rid}
    if isinstance(command, ServingShutdown):
        return {"type": "shutdown"}
    raise TypeError(f"Unsupported serving command: {type(command)!r}")


def serving_command_from_dict(payload: dict[str, Any]) -> ServingCommand:
    type_name = payload.get("type")
    if type_name == "generate":
        return ServingGenerate(
            rid=payload["rid"],
            input=serving_input_from_dict(payload["input"]),
            sampling_params=sampling_params_from_dict(payload["sampling_params"]),
            stream=payload.get("stream", False),
            stream_mode=payload.get("stream_mode", "denoise"),
            user=payload.get("user"),
            created_time=payload.get("created_time"),
        )
    if type_name == "abort":
        return ServingAbort(rid=payload["rid"])
    if type_name == "shutdown":
        return ServingShutdown()
    raise ValueError(f"Unsupported serving command type: {type_name!r}")


def serving_event_to_dict(event: ServingEvent) -> dict[str, Any]:
    if isinstance(event, ServingReply):
        return {
            "type": "reply",
            "rid": event.rid,
            "text": event.text,
            "token_ids": event.token_ids,
            "nfe": event.nfe,
            "finish_reason": event.finish_reason,
            "full_text": event.full_text,
            "full_token_ids": event.full_token_ids,
            "finished": event.finished,
        }
    if isinstance(event, ServingDelta):
        return {
            "type": "delta",
            "rid": event.rid,
            "token_offset": event.token_offset,
            "text": event.text,
            "token_ids": event.token_ids,
            "nfe": event.nfe,
            "finished": event.finished,
        }
    if isinstance(event, ServingBufferSnapshot):
        return {
            "type": "buffer_snapshot",
            "rid": event.rid,
            "token_offset": event.token_offset,
            "absolute_start": event.absolute_start,
            "absolute_end": event.absolute_end,
            "text": event.text,
            "token_ids": event.token_ids,
            "nfe": event.nfe,
            "finished": event.finished,
        }
    if isinstance(event, ServingError):
        return {"type": "error", "rid": event.rid, "message": event.message}
    raise TypeError(f"Unsupported serving event: {type(event)!r}")


def serving_event_from_dict(payload: dict[str, Any]) -> ServingEvent:
    type_name = payload.get("type")
    if type_name == "reply":
        return ServingReply(
            rid=payload["rid"],
            text=payload["text"],
            token_ids=payload["token_ids"],
            nfe=payload["nfe"],
            finish_reason=payload.get("finish_reason"),
            full_text=payload.get("full_text"),
            full_token_ids=payload.get("full_token_ids"),
            finished=payload.get("finished", True),
        )
    if type_name == "delta":
        return ServingDelta(
            rid=payload["rid"],
            token_offset=payload["token_offset"],
            text=payload["text"],
            token_ids=payload["token_ids"],
            nfe=payload["nfe"],
            finished=payload.get("finished", False),
        )
    if type_name == "buffer_snapshot":
        return ServingBufferSnapshot(
            rid=payload["rid"],
            token_offset=payload["token_offset"],
            absolute_start=payload["absolute_start"],
            absolute_end=payload["absolute_end"],
            text=payload["text"],
            token_ids=payload["token_ids"],
            nfe=payload["nfe"],
            finished=payload.get("finished", False),
        )
    if type_name == "error":
        return ServingError(rid=payload["rid"], message=payload["message"])
    raise ValueError(f"Unsupported serving event type: {type_name!r}")
