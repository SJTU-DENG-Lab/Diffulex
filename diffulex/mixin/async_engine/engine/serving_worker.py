from __future__ import annotations

from dataclasses import dataclass, field

from diffulex.logger import get_logger
from diffulex.server.protocol import (
    ChatInput,
    PromptInput,
    ServingAbort,
    ServingBufferSnapshot,
    ServingCommand,
    ServingDelta,
    ServingError,
    ServingEvent,
    ServingGenerate,
    ServingReply,
)
from diffulex.utils.output import decode_token_ids_robust

logger = get_logger(__name__)

SUPPORTED_STREAM_MODES = {"block_append", "denoise"}


@dataclass
class ServingRequestState:
    rid: str
    engine_req_id: int
    stream: bool = False
    stream_mode: str = "denoise"
    emitted_token_count: int = 0
    emitted_text_len: int = 0


@dataclass
class ServingState:
    requests: dict[int, ServingRequestState] = field(default_factory=dict)


class DiffulexServingWorkerMixin:
    """Serving-only owner-loop helpers for DiffulexTPWorker.

    The async HTTP frontend should enqueue ServingCommand objects and call
    run_serving_tick() from exactly one engine owner thread. The mixin keeps
    scheduler/model_runner mutations synchronous and serialized.
    """

    def init_serving_state(self) -> None:
        if not hasattr(self, "serving_state"):
            self.serving_state = ServingState()

    def run_serving_tick(self, commands: list[ServingCommand]) -> list[ServingEvent]:
        self.init_serving_state()
        events: list[ServingEvent] = []

        for command in commands:
            if isinstance(command, ServingGenerate):
                event = self.add_serving_request(command)
                if event is not None:
                    events.append(event)
            elif isinstance(command, ServingAbort):
                self.abort_serving_request(command)
            else:
                raise TypeError(f"Unsupported serving command: {type(command)!r}")

        if not self.is_finished():
            events.extend(self.step_serving_requests())

        return events

    def add_serving_request(self, command: ServingGenerate) -> ServingError | None:
        if command.stream and command.stream_mode not in SUPPORTED_STREAM_MODES:
            return ServingError(command.rid, f"Unsupported stream_mode: {command.stream_mode}")

        try:
            prompt = self.prepare_serving_prompt(command)
            engine_req_id = self.add_request(prompt, command.sampling_params)
        except Exception as exc:
            return ServingError(command.rid, str(exc))

        self.serving_state.requests[engine_req_id] = ServingRequestState(
            rid=command.rid,
            engine_req_id=engine_req_id,
            stream=command.stream,
            stream_mode=command.stream_mode,
        )
        return None

    def prepare_serving_prompt(self, command: ServingGenerate) -> str | list[int]:
        if isinstance(command.input, PromptInput):
            return command.input.prompt
        if isinstance(command.input, ChatInput):
            return self.render_chat_prompt_for_serving(command.input.messages)
        raise TypeError(f"Unsupported serving input: {type(command.input)!r}")

    def abort_serving_request(self, command: ServingAbort) -> None:
        for engine_req_id, state in list(self.serving_state.requests.items()):
            if state.rid == command.rid:
                del self.serving_state.requests[engine_req_id]
                abort_request = getattr(self, "abort_request", None)
                if abort_request is not None:
                    abort_request(engine_req_id)
                break

    def step_serving_requests(self) -> list[ServingEvent]:
        reqs, _ = self.step()
        events: list[ServingEvent] = []

        for req in reqs:
            state = self.serving_state.requests.get(req.req_id)
            if state is None:
                continue

            if state.stream:
                events.extend(self.build_stream_events(state, req))

            if not req.is_finished:
                continue

            del self.serving_state.requests[req.req_id]
            events.append(self.build_serving_reply(state.rid, req))

        return events

    def build_stream_events(self, state: ServingRequestState, req) -> list[ServingEvent]:
        if state.stream_mode == "block_append":
            event = self.build_block_append_delta(state, req)
            return [event] if event is not None else []
        if state.stream_mode == "denoise":
            event = self.build_denoise_snapshot(state, req)
            return [event] if event is not None else []
        return [ServingError(state.rid, f"Unsupported stream_mode: {state.stream_mode}")]

    def build_block_append_delta(self, state: ServingRequestState, req) -> ServingDelta | None:
        token_ids = self.stable_generated_token_ids(req)
        text = decode_token_ids_robust(self.tokenizer, token_ids)
        if len(token_ids) <= state.emitted_token_count and len(text) <= state.emitted_text_len:
            return None

        delta = ServingDelta(
            rid=state.rid,
            token_offset=state.emitted_token_count,
            text=text[state.emitted_text_len :],
            token_ids=token_ids[state.emitted_token_count :],
            nfe=int(getattr(req, "nfe", 0) or 0),
            finished=req.is_finished,
        )
        state.emitted_token_count = len(token_ids)
        state.emitted_text_len = len(text)
        return delta

    def build_denoise_snapshot(self, state: ServingRequestState, req) -> ServingBufferSnapshot | None:
        snapshot = self.current_buffer_snapshot(req)
        if snapshot is None:
            return None

        absolute_start, absolute_end, token_ids = snapshot
        prefix_len = self.prompt_len(req)
        return ServingBufferSnapshot(
            rid=state.rid,
            token_offset=max(0, absolute_start - prefix_len),
            absolute_start=absolute_start,
            absolute_end=absolute_end,
            text=decode_token_ids_robust(self.tokenizer, token_ids),
            token_ids=token_ids,
            nfe=int(getattr(req, "nfe", 0) or 0),
            finished=req.is_finished,
        )

    def stable_generated_token_ids(self, req) -> list[int]:
        if req.is_finished:
            token_ids = list(getattr(req, "truncated_response", []) or [])
            return self.trim_at_first_mask_token(token_ids, req)

        if not getattr(req, "is_multi_block", False):
            return []

        buffer = getattr(req, "dllm_block_buffer", None)
        if buffer is None:
            return []

        prefix_len = self.prompt_len(req)
        stable_abs_end = max(prefix_len, min(buffer.first_running_block.start, len(req.token_ids)))
        token_ids = list(req.token_ids[prefix_len:stable_abs_end])
        return self.trim_at_first_mask_token(token_ids, req)

    def trim_at_first_mask_token(self, token_ids: list[int], req) -> list[int]:
        mask_token_id = self.mask_token_id(req)
        if mask_token_id is None or mask_token_id not in token_ids:
            return token_ids
        return token_ids[: token_ids.index(mask_token_id)]

    def drop_mask_tokens(self, token_ids: list[int], req) -> list[int]:
        mask_token_id = self.mask_token_id(req)
        if mask_token_id is None:
            return token_ids
        return [token_id for token_id in token_ids if token_id != mask_token_id]

    def mask_token_id(self, req) -> int | None:
        req_mask = getattr(req, "mask_token_id", None)
        if req_mask is not None:
            return int(req_mask)
        tokenizer_mask = getattr(self.tokenizer, "mask_token_id", None)
        return int(tokenizer_mask) if tokenizer_mask is not None else None

    def current_buffer_snapshot(self, req) -> tuple[int, int, list[int]] | None:
        prefix_len = self.prompt_len(req)
        if getattr(req, "is_multi_block", False):
            buffer = getattr(req, "dllm_block_buffer", None)
            if buffer is None:
                return None
            absolute_start = prefix_len
            absolute_end = min(len(req.token_ids), buffer.last_running_block.end)
            if absolute_end <= absolute_start:
                return None
            return absolute_start, absolute_end, list(req.token_ids[absolute_start:absolute_end])

        token_ids = list(getattr(req, "truncated_response", []) or [])
        if not token_ids:
            return None
        return prefix_len, prefix_len + len(token_ids), token_ids

    def prompt_len(self, req) -> int:
        return int(getattr(req, "prefix_len", getattr(req, "num_prompt_tokens", 0)) or 0)

    def build_serving_reply(self, rid: str, req) -> ServingReply:
        token_ids = self.drop_mask_tokens(list(getattr(req, "truncated_response", []) or []), req)
        full_token_ids = self.drop_mask_tokens(list(getattr(req, "full_response", []) or token_ids), req)
        eos = getattr(self.tokenizer, "eos_token", None) or ""

        raw_text = decode_token_ids_robust(self.tokenizer, token_ids)
        text = raw_text.split(eos)[0] if eos else raw_text
        full_text = decode_token_ids_robust(self.tokenizer, full_token_ids)

        return ServingReply(
            rid=rid,
            text=text,
            token_ids=token_ids,
            nfe=int(getattr(req, "nfe", 0) or 0),
            finish_reason=getattr(req, "completion_reason", None),
            full_text=full_text,
            full_token_ids=full_token_ids,
        )

    def render_chat_prompt_for_serving(self, messages: list[dict[str, str]]) -> str:
        tokenizer = self.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                logger.warning("Tokenizer chat template failed; using plain chat fallback", exc_info=True)
        return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages) + "\nassistant:"
