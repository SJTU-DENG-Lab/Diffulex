"""Run with: streamlit run examples/streamlit_block_append_chat.py -- --base-url http://localhost:8000."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

DEFAULT_STREAM_MODE = "denoise"
DEFAULT_MAX_TOKENS = 512
DEFAULT_MAX_NFE = 512
DEFAULT_TEMPERATURE = 0.0
MASK_TOKEN_TEXT = "<|MASK|>"
DEFAULT_MASK_SYMBOL = "▒"
DISPLAY_STOP_TOKENS = ("<|im_end|>",)


@dataclass
class StreamUpdate:
    text: str
    replace: bool = False
    event: str | None = None
    nfe: int | None = None
    token_offset: int = 0
    token_count: int = 0


@dataclass
class DenoiseSegment:
    token_offset: int
    token_count: int
    text: str

    @property
    def token_end(self) -> int:
        return self.token_offset + self.token_count


class DenoiseDraft:
    def __init__(self) -> None:
        self.segments: list[DenoiseSegment] = []

    def apply_snapshot(self, update: StreamUpdate) -> str:
        if update.token_count <= 0:
            return self.text

        snapshot_end = update.token_offset + update.token_count
        self.segments = [
            segment
            for segment in self.segments
            if segment.token_end <= update.token_offset or segment.token_offset >= snapshot_end
        ]
        self.segments.append(DenoiseSegment(update.token_offset, update.token_count, update.text))
        self.segments.sort(key=lambda segment: segment.token_offset)
        return self.text

    @property
    def text(self) -> str:
        return "".join(segment.text for segment in self.segments)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streamlit chat client for Diffulex streaming")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Diffulex HTTP serving base URL")
    parser.add_argument("--model", default=None, help="Optional model name passed to /v1/chat/completions")
    parser.add_argument("--timeout", type=float, default=300.0, help="HTTP request timeout in seconds")
    args, _ = parser.parse_known_args()
    return args


def iter_sse_data(response) -> Iterator[str]:
    for raw_line in response:
        line = raw_line.decode("utf-8").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break
        yield data


def truncate_display_special_tokens(text: str) -> str:
    stop_positions = [text.find(token) for token in DISPLAY_STOP_TOKENS if token in text]
    if not stop_positions:
        return text
    return text[: min(stop_positions)]


def render_mask_tokens(text: str, mask_symbol: str) -> str:
    return text.replace(MASK_TOKEN_TEXT, mask_symbol or DEFAULT_MASK_SYMBOL)


def render_assistant_text(text: str, mask_symbol: str) -> str:
    return render_mask_tokens(truncate_display_special_tokens(text), mask_symbol)


def stream_chat_completion(
    *,
    base_url: str,
    messages: list[dict[str, str]],
    model: str | None,
    max_tokens: int,
    temperature: float,
    max_nfe: int | None,
    max_repetition_run: int | None,
    ignore_eos: bool,
    stream_mode: str,
    timeout: float,
) -> Iterator[StreamUpdate]:
    payload: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_mode": stream_mode,
        "ignore_eos": ignore_eos,
    }
    if model:
        payload["model"] = model
    if max_nfe is not None:
        payload["max_nfe"] = max_nfe
    if max_repetition_run is not None:
        payload["max_repetition_run"] = max_repetition_run

    request = Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            for data in iter_sse_data(response):
                event = json.loads(data)
                if "error" in event:
                    message = event["error"].get("message", event["error"])
                    raise RuntimeError(str(message))
                if stream_mode == "denoise":
                    event_type = event.get("event")
                    if event_type in {"buffer_snapshot", "final"}:
                        nfe = event.get("nfe")
                        token_ids = event.get("token_ids") or []
                        yield StreamUpdate(
                            text=str(event.get("text", "")),
                            replace=True,
                            event=event_type,
                            nfe=int(nfe) if nfe is not None else None,
                            token_offset=int(event.get("token_offset", 0) or 0),
                            token_count=len(token_ids),
                        )
                    continue
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield StreamUpdate(text=content)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to connect to Diffulex server: {exc.reason}") from exc


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="Diffulex Chat", page_icon=None)
    st.title("Diffulex Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Server")
        base_url = st.text_input("Base URL", value=args.base_url)
        model = st.text_input("Model", value=args.model or "")
        timeout = st.number_input("Timeout seconds", min_value=1.0, value=args.timeout, step=10.0)

        st.header("Sampling")
        max_tokens = st.number_input("Max tokens", min_value=1, max_value=8192, value=DEFAULT_MAX_TOKENS, step=16)
        temperature = st.number_input("Temperature", min_value=0.0, max_value=5.0, value=DEFAULT_TEMPERATURE, step=0.1)
        max_nfe_value = st.number_input("Max NFE, 0 means unset", min_value=0, value=DEFAULT_MAX_NFE, step=1)
        max_repetition_value = st.number_input("Max repetition run, 0 means unset", min_value=0, value=0, step=1)
        ignore_eos = st.checkbox("Ignore EOS", value=False)
        stream_mode_options = ["denoise", "block_append"]
        stream_mode = st.selectbox(
            "Stream mode",
            options=stream_mode_options,
            index=stream_mode_options.index(DEFAULT_STREAM_MODE),
        )
        mask_symbol = st.text_input("Mask symbol", value=DEFAULT_MASK_SYMBOL, max_chars=4)

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            content = message["content"]
            if message["role"] == "assistant":
                content = render_assistant_text(content, mask_symbol)
            st.markdown(content)

    prompt = st.chat_input("Type a message")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        raw_collected = ""
        display_collected = ""
        denoise_draft = DenoiseDraft()
        try:
            for update in stream_chat_completion(
                base_url=base_url,
                messages=st.session_state.messages,
                model=model or None,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                max_nfe=int(max_nfe_value) if max_nfe_value else None,
                max_repetition_run=int(max_repetition_value) if max_repetition_value else None,
                ignore_eos=ignore_eos,
                stream_mode=stream_mode,
                timeout=float(timeout),
            ):
                if update.replace:
                    if update.event == "buffer_snapshot":
                        raw_collected = denoise_draft.apply_snapshot(update)
                    else:
                        raw_collected = update.text
                    display_collected = render_assistant_text(raw_collected, mask_symbol)
                    display_text = display_collected
                    if update.event == "buffer_snapshot" and update.nfe is not None:
                        display_text = f"`nfe={update.nfe}`\n\n{display_text}"
                    placeholder.markdown(display_text or " ")
                else:
                    raw_collected += update.text
                    display_collected = render_assistant_text(raw_collected, mask_symbol)
                    placeholder.markdown(display_collected or " ")
        except RuntimeError as exc:
            raw_collected = f"Request failed: {exc}"
            display_collected = raw_collected
            placeholder.error(display_collected)

    st.session_state.messages.append({"role": "assistant", "content": raw_collected})


if __name__ == "__main__":
    main()
