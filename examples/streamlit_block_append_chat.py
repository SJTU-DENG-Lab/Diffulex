"""Run with: streamlit run examples/streamlit_block_append_chat.py -- --base-url http://localhost:8000."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

DEFAULT_STREAM_MODE = "denoise"
DEFAULT_MAX_TOKENS = 8192
MAX_TOKENS_LIMIT = 16384
DEFAULT_MAX_NFE = 1024
DEFAULT_MAX_REPETITION_RUN = 0
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SMOOTH_RENDER = True
DEFAULT_RENDER_INTERVAL_MS = 80
MASK_TOKEN_TEXTS = ("<|MASK|>", "<|mask|>", "<mask>")
DEFAULT_MASK_SYMBOL = "▨"
DISPLAY_STOP_TOKENS = ("<|im_end|>", "<eos>", "<turn|>", "<|turn>", "<|endoftext|>")
DISPLAY_DROP_TOKENS = (
    "<pad>",
    "<bos>",
    "<|channel>thought",
    "<|channel>analysis",
    "<|channel>final",
    "<channel|>",
)
SPECIAL_TOKEN_IDS = {0, 1, 4, 50, 106}

RAW_CHAT_TEMPLATE = "Raw chat"
DGEMMA_MATH_TEMPLATE = "GSM8K/MATH - DiffusionGemma eval"
DGEMMA_MBPP_TEMPLATE = "MBPP - DiffusionGemma eval"
DGEMMA_HUMANEVAL_TEMPLATE = "HumanEval - DiffusionGemma eval"
DMAX_GSM8K_TEMPLATE = "GSM8K - DMax/LLaDA2 eval"
DMAX_MATH500_TEMPLATE = "MATH500 - DMax/LLaDA2 4-shot eval"
DMAX_MBPP_TEMPLATE = "MBPP - DMax/LLaDA2 eval"
DMAX_HUMANEVAL_TEMPLATE = "HumanEval - DMax/LLaDA2 eval"
PROMPT_TEMPLATE_OPTIONS = [
    DGEMMA_MATH_TEMPLATE,
    DGEMMA_MBPP_TEMPLATE,
    DGEMMA_HUMANEVAL_TEMPLATE,
    DMAX_GSM8K_TEMPLATE,
    DMAX_MATH500_TEMPLATE,
    DMAX_MBPP_TEMPLATE,
    DMAX_HUMANEVAL_TEMPLATE,
    RAW_CHAT_TEMPLATE,
]


@dataclass
class StreamUpdate:
    text: str
    replace: bool = False
    event: str | None = None
    nfe: int | None = None
    token_offset: int = 0
    token_count: int = 0
    token_ids: list[int] = field(default_factory=list)
    engine_decode_time_s: float | None = None
    engine_decode_tokens: int | None = None
    engine_decode_tps: float | None = None


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


class SmoothMarkdownRenderer:
    def __init__(
        self,
        *,
        placeholder,
        enabled: bool,
        render_interval_s: float,
    ) -> None:
        self.placeholder = placeholder
        self.enabled = enabled
        self.render_interval_s = max(0.02, float(render_interval_s))
        self.displayed_text = ""
        self.displayed_prefix = ""
        self.last_render_s = 0.0

    def render(self, target_text: str, *, prefix: str = "", force: bool = False) -> bool:
        now = time.perf_counter()
        if (
            self.enabled
            and not force
            and (self.displayed_text or self.displayed_prefix)
            and now - self.last_render_s < self.render_interval_s
        ):
            return False

        if target_text == self.displayed_text and prefix == self.displayed_prefix and not force:
            return False

        self.displayed_text = target_text
        self.displayed_prefix = prefix
        self.last_render_s = now
        rendered = target_text or " "
        if prefix:
            rendered = f"{prefix}\n\n{rendered}"
        self.placeholder.markdown(rendered)
        return True

    def finish(self, target_text: str, *, prefix: str = "") -> None:
        self.render(target_text, prefix=prefix, force=True)


def dmax_role_prompt(user_content: str) -> str:
    return (
        "<role>SYSTEM</role>detailed thinking off<|role_end|>"
        f"<role>HUMAN</role>{user_content}<|role_end|>"
        "<role>ASSISTANT</role>"
    )


def diffusion_gemma_math_prompt(question: str) -> str:
    return (
        "<bos><|turn>user\n"
        f"{question.strip()}\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
        "<turn|>\n"
        "<|turn>model\n"
        "<|channel>thought\n"
        "<channel|>\n"
    )


def diffusion_gemma_user_prompt(user_content: str) -> str:
    return (
        "<bos><|turn>user\n"
        f"{user_content.strip()}"
        "<turn|>\n"
        "<|turn>model\n"
        "<|channel>thought\n"
        "<channel|>\n"
    )


def diffusion_gemma_mbpp_prompt(problem: str) -> str:
    user_content = (
        "You are an expert Python programmer, and here is your task:\n"
        f"{problem.strip()}\n\n"
        "Please enclose your code within delimiters as follows:\n"
        "```python\n# YOUR CODE HERE\n```"
    )
    return diffusion_gemma_user_prompt(user_content)


def diffusion_gemma_humaneval_prompt(problem: str) -> str:
    clean_problem = problem.strip().removeprefix("You need to complete this code:\n").strip()
    user_content = (
        "Write a solution to the following problem and make sure that it passes the tests:\n"
        f"```python\n{clean_problem}\n```\n\n"
        "Please enclose your code within delimiters as follows:\n"
        "```python\n# YOUR CODE HERE\n```"
    )
    return diffusion_gemma_user_prompt(user_content)


def dmax_gsm8k_prompt(question: str) -> str:
    return dmax_role_prompt(f"{question.strip()}\nLet's think step by step\n")


def math_4shot_examples() -> str:
    try:
        from diffulex_bench.tasks.common.lightning_math_prompts import MATH_4SHOT_EXAMPLES
    except Exception:
        return ""
    return MATH_4SHOT_EXAMPLES


def dmax_math500_prompt(question: str) -> str:
    return dmax_role_prompt(f"{math_4shot_examples()}{question.strip()}\nLet's think step by step\n")


def dmax_mbpp_prompt(problem: str) -> str:
    user_content = (
        "You are an expert Python programmer, and here is your task:\n"
        f"{problem.strip()}\n\n"
        "Please enclose your code within delimiters as follows:\n"
        "```python\n# YOUR CODE HERE\n```\n\n"
    )
    return dmax_role_prompt(user_content)


def dmax_humaneval_prompt(problem: str) -> str:
    clean_problem = problem.strip().removeprefix("You need to complete this code:\n").strip()
    user_content = (
        "Write a solution to the following problem and make sure that it passes the tests:\n"
        f"```python\n{clean_problem}\n```\n\n"
        "Please enclose your code within delimiters as follows:\n"
        "```python\n# YOUR CODE HERE\n```\n\n"
    )
    return dmax_role_prompt(user_content)


def build_prompt_from_template(template: str, user_text: str) -> str:
    if template == DGEMMA_MATH_TEMPLATE:
        return diffusion_gemma_math_prompt(user_text)
    if template == DGEMMA_MBPP_TEMPLATE:
        return diffusion_gemma_mbpp_prompt(user_text)
    if template == DGEMMA_HUMANEVAL_TEMPLATE:
        return diffusion_gemma_humaneval_prompt(user_text)
    if template == DMAX_GSM8K_TEMPLATE:
        return dmax_gsm8k_prompt(user_text)
    if template == DMAX_MATH500_TEMPLATE:
        return dmax_math500_prompt(user_text)
    if template == DMAX_MBPP_TEMPLATE:
        return dmax_mbpp_prompt(user_text)
    if template == DMAX_HUMANEVAL_TEMPLATE:
        return dmax_humaneval_prompt(user_text)
    return user_text


def template_uses_raw_generate(template: str) -> bool:
    return template != RAW_CHAT_TEMPLATE


def chat_input_placeholder(template: str) -> str:
    if template == DGEMMA_MATH_TEMPLATE:
        return "Paste a GSM8K/MATH problem"
    if template in {DMAX_GSM8K_TEMPLATE, DMAX_MATH500_TEMPLATE}:
        return "Paste a math problem"
    if template in {DGEMMA_MBPP_TEMPLATE, DGEMMA_HUMANEVAL_TEMPLATE, DMAX_MBPP_TEMPLATE, DMAX_HUMANEVAL_TEMPLATE}:
        return "Paste a coding problem"
    return "Type a message"


def default_template_for_model(model: str | None) -> str:
    model_name = (model or "").lower()
    if "llada" in model_name:
        return DMAX_GSM8K_TEMPLATE
    if "diffusion_gemma" in model_name or "gemma" in model_name:
        return DGEMMA_MATH_TEMPLATE
    return PROMPT_TEMPLATE_OPTIONS[0]


def default_max_tokens_for_model(model: str | None) -> int:
    model_name = (model or "").lower()
    if "llada" in model_name:
        return 4096
    return DEFAULT_MAX_TOKENS


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


def drop_display_special_tokens(text: str) -> str:
    rendered = text
    for token in DISPLAY_DROP_TOKENS:
        rendered = rendered.replace(token, "")
    return rendered


def drop_mask_tokens(text: str) -> str:
    rendered = text
    for token in MASK_TOKEN_TEXTS:
        rendered = rendered.replace(token, "")
    return rendered


def render_mask_tokens(text: str, mask_symbol: str) -> str:
    rendered = text
    for token in MASK_TOKEN_TEXTS:
        rendered = rendered.replace(token, mask_symbol or DEFAULT_MASK_SYMBOL)
    return rendered


def clean_assistant_text(text: str) -> str:
    return drop_display_special_tokens(truncate_display_special_tokens(text)).strip()


def render_assistant_text(text: str, mask_symbol: str) -> str:
    return render_mask_tokens(clean_assistant_text(text), mask_symbol)


def clean_final_assistant_text(text: str) -> str:
    return drop_mask_tokens(clean_assistant_text(text)).strip()


def estimate_visible_tokens(text: str) -> int:
    cleaned = drop_mask_tokens(clean_assistant_text(text)).strip()
    if not cleaned:
        return 0
    return max(1, len(cleaned.split()))


def count_visible_token_ids(token_ids: list[int], fallback_text: str) -> int:
    fallback_count = estimate_visible_tokens(fallback_text)
    if fallback_count == 0:
        return 0
    count = sum(1 for token_id in token_ids if int(token_id) not in SPECIAL_TOKEN_IDS)
    if count > 0:
        return count
    return fallback_count


def format_latency(seconds: float | None) -> str:
    if seconds is None:
        return "pending"
    return f"{seconds:.2f}s"


def format_tps(tokens: int, seconds: float | None) -> str:
    if seconds is None or seconds <= 0 or tokens <= 0:
        return "pending"
    return f"{tokens / seconds:.2f} tok/s"


def format_tps_value(value: float | None) -> str:
    if value is None or value <= 0:
        return "pending"
    return f"{value:.2f} tok/s"


def render_metrics(
    *,
    placeholder,
    ttft_s: float | None,
    client_decode_time_s: float | None,
    token_count: int,
    nfe: int | None,
    engine_decode_tps: float | None,
) -> None:
    metrics = [
        f"TTFT: {format_latency(ttft_s)}",
        f"Decode TPS (client): {format_tps(token_count, client_decode_time_s)}",
        f"Decode TPS (engine): {format_tps_value(engine_decode_tps)}",
        f"Tokens: {token_count}",
    ]
    if nfe is not None:
        metrics.append(f"NFE: {nfe}")
    placeholder.caption(" | ".join(metrics))


def event_engine_decode_time(event: dict[str, Any]) -> float | None:
    value = event.get("engine_decode_time_s")
    return float(value) if value is not None else None


def event_engine_decode_tokens(event: dict[str, Any]) -> int | None:
    value = event.get("engine_decode_tokens")
    return int(value) if value is not None else None


def event_engine_decode_tps(event: dict[str, Any]) -> float | None:
    value = event.get("engine_decode_tps")
    return float(value) if value is not None else None


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
                            token_ids=[int(token_id) for token_id in token_ids],
                            engine_decode_time_s=event_engine_decode_time(event),
                            engine_decode_tokens=event_engine_decode_tokens(event),
                            engine_decode_tps=event_engine_decode_tps(event),
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


def stream_prompt_completion(
    *,
    base_url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    max_nfe: int | None,
    max_repetition_run: int | None,
    ignore_eos: bool,
    stream_mode: str,
    timeout: float,
) -> Iterator[StreamUpdate]:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_mode": stream_mode,
        "ignore_eos": ignore_eos,
    }
    if max_nfe is not None:
        payload["max_nfe"] = max_nfe
    if max_repetition_run is not None:
        payload["max_repetition_run"] = max_repetition_run

    request = Request(
        f"{base_url.rstrip('/')}/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            for data in iter_sse_data(response):
                event = json.loads(data)
                if event.get("event") == "error":
                    raise RuntimeError(str(event.get("message", "server error")))
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
                            token_ids=[int(token_id) for token_id in token_ids],
                            engine_decode_time_s=event_engine_decode_time(event),
                            engine_decode_tokens=event_engine_decode_tokens(event),
                            engine_decode_tps=event_engine_decode_tps(event),
                        )
                    continue
                if event.get("event") == "append":
                    text = str(event.get("text", ""))
                    if text:
                        nfe = event.get("nfe")
                        yield StreamUpdate(
                            text=text,
                            event="append",
                            nfe=int(nfe) if nfe is not None else None,
                            token_ids=[int(token_id) for token_id in event.get("token_ids") or []],
                            engine_decode_time_s=event_engine_decode_time(event),
                            engine_decode_tokens=event_engine_decode_tokens(event),
                            engine_decode_tps=event_engine_decode_tps(event),
                        )
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

        st.header("Prompt")
        prompt_template = st.selectbox(
            "Task template",
            options=PROMPT_TEMPLATE_OPTIONS,
            index=PROMPT_TEMPLATE_OPTIONS.index(default_template_for_model(model)),
            help="Eval templates send the current problem as a raw prompt through /generate.",
        )
        use_raw_generate = template_uses_raw_generate(prompt_template)
        show_prompt_preview = st.checkbox("Show rendered prompt", value=False)

        st.header("Sampling")
        max_tokens = st.number_input(
            "Max tokens",
            min_value=1,
            max_value=MAX_TOKENS_LIMIT,
            value=default_max_tokens_for_model(model),
            step=16,
        )
        temperature = st.number_input("Temperature", min_value=0.0, max_value=5.0, value=DEFAULT_TEMPERATURE, step=0.1)
        max_nfe_value = st.number_input("Max NFE, 0 means unset", min_value=0, value=DEFAULT_MAX_NFE, step=1)
        max_repetition_value = st.number_input(
            "Max repetition run, 0 means unset",
            min_value=0,
            value=DEFAULT_MAX_REPETITION_RUN,
            step=1,
            key="max_repetition_run_unset_default",
        )
        ignore_eos = st.checkbox("Ignore EOS", value=False)
        send_history = st.checkbox(
            "Send chat history",
            value=False,
            disabled=use_raw_generate,
            help="Disable this for DiffusionGemma demos to avoid feeding generated special tokens back into the next turn.",
        )
        if use_raw_generate:
            st.caption("Template modes send only the current rendered prompt through /generate.")
        stream_mode_options = ["denoise", "block_append"]
        stream_mode = st.selectbox(
            "Stream mode",
            options=stream_mode_options,
            index=stream_mode_options.index(DEFAULT_STREAM_MODE),
        )
        mask_symbol = st.text_input("Mask symbol", value=DEFAULT_MASK_SYMBOL, max_chars=4)
        smooth_render = st.checkbox(
            "Smooth frontend rendering",
            value=DEFAULT_SMOOTH_RENDER,
            help="Throttle Streamlit updates and render only the latest server snapshot.",
        )
        render_interval_ms = st.slider(
            "Render interval ms",
            min_value=80,
            max_value=1000,
            value=DEFAULT_RENDER_INTERVAL_MS,
            step=10,
            disabled=not smooth_render,
            key="render_interval_ms_default_80",
        )

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            content = message["content"]
            if message["role"] == "assistant":
                content = render_assistant_text(content, mask_symbol)
            st.markdown(content)

    prompt = st.chat_input(chat_input_placeholder(prompt_template))
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    rendered_prompt = build_prompt_from_template(prompt_template, prompt)
    if show_prompt_preview and use_raw_generate:
        with st.expander("Rendered prompt", expanded=False):
            st.code(rendered_prompt)
    request_messages = None
    if not use_raw_generate:
        request_messages = (
            st.session_state.messages
            if send_history
            else [{"role": "user", "content": prompt}]
        )

    with st.chat_message("assistant"):
        metrics_placeholder = st.empty()
        placeholder = st.empty()
        text_renderer = SmoothMarkdownRenderer(
            placeholder=placeholder,
            enabled=bool(smooth_render),
            render_interval_s=float(render_interval_ms) / 1000.0,
        )
        raw_collected = ""
        display_collected = ""
        display_prefix = ""
        denoise_draft = DenoiseDraft()
        request_start = time.perf_counter()
        first_update_time: float | None = None
        latest_update_time: float | None = None
        latest_nfe: int | None = None
        latest_engine_decode_tps: float | None = None
        last_metrics_render_s = 0.0
        final_token_count = 0
        streamed_token_count = 0
        render_metrics(
            placeholder=metrics_placeholder,
            ttft_s=None,
            client_decode_time_s=None,
            token_count=0,
            nfe=None,
            engine_decode_tps=None,
        )
        try:
            if use_raw_generate:
                updates = stream_prompt_completion(
                    base_url=base_url,
                    prompt=rendered_prompt,
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                    max_nfe=int(max_nfe_value) if max_nfe_value else None,
                    max_repetition_run=int(max_repetition_value) if max_repetition_value else None,
                    ignore_eos=ignore_eos,
                    stream_mode=stream_mode,
                    timeout=float(timeout),
                )
            else:
                assert request_messages is not None
                updates = stream_chat_completion(
                    base_url=base_url,
                    messages=request_messages,
                    model=model or None,
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                    max_nfe=int(max_nfe_value) if max_nfe_value else None,
                    max_repetition_run=int(max_repetition_value) if max_repetition_value else None,
                    ignore_eos=ignore_eos,
                    stream_mode=stream_mode,
                    timeout=float(timeout),
                )
            for update in updates:
                now = time.perf_counter()
                if first_update_time is None:
                    first_update_time = now
                latest_update_time = now
                if update.nfe is not None:
                    latest_nfe = update.nfe
                if update.engine_decode_tps is not None:
                    latest_engine_decode_tps = update.engine_decode_tps
                if update.event == "final":
                    final_token_count = count_visible_token_ids(update.token_ids, update.text)
                if update.replace:
                    if update.event == "buffer_snapshot":
                        raw_collected = denoise_draft.apply_snapshot(update)
                    else:
                        raw_collected = update.text
                    display_collected = render_assistant_text(raw_collected, mask_symbol)
                    if update.event == "buffer_snapshot":
                        final_token_count = estimate_visible_tokens(raw_collected)
                    display_prefix = (
                        f"`nfe={update.nfe}`"
                        if update.event == "buffer_snapshot" and update.nfe is not None
                        else ""
                    )
                    did_render_text = text_renderer.render(display_collected, prefix=display_prefix)
                else:
                    raw_collected += update.text
                    display_collected = render_assistant_text(raw_collected, mask_symbol)
                    streamed_token_count += count_visible_token_ids(update.token_ids, update.text)
                    final_token_count = streamed_token_count or estimate_visible_tokens(raw_collected)
                    display_prefix = ""
                    did_render_text = text_renderer.render(display_collected, prefix=display_prefix)
                ttft_s = first_update_time - request_start if first_update_time is not None else None
                client_decode_time_s = (
                    latest_update_time - first_update_time
                    if (
                        first_update_time is not None
                        and latest_update_time is not None
                        and latest_update_time > first_update_time
                    )
                    else latest_update_time - request_start
                    if latest_update_time is not None
                    else None
                )
                visible_token_count = final_token_count
                if (
                    not smooth_render
                    or did_render_text
                    or now - last_metrics_render_s >= float(render_interval_ms) / 1000.0
                ):
                    render_metrics(
                        placeholder=metrics_placeholder,
                        ttft_s=ttft_s,
                        client_decode_time_s=client_decode_time_s,
                        token_count=visible_token_count,
                        nfe=latest_nfe,
                        engine_decode_tps=latest_engine_decode_tps,
                    )
                    last_metrics_render_s = now
        except RuntimeError as exc:
            raw_collected = f"Request failed: {exc}"
            display_collected = raw_collected
            placeholder.error(display_collected)
        else:
            if display_collected:
                text_renderer.finish(
                    display_collected,
                    prefix=display_prefix,
                )
                now = time.perf_counter()
                ttft_s = first_update_time - request_start if first_update_time is not None else None
                client_decode_time_s = (
                    latest_update_time - first_update_time
                    if (
                        first_update_time is not None
                        and latest_update_time is not None
                        and latest_update_time > first_update_time
                    )
                    else latest_update_time - request_start
                    if latest_update_time is not None
                    else None
                )
                render_metrics(
                    placeholder=metrics_placeholder,
                    ttft_s=ttft_s,
                    client_decode_time_s=client_decode_time_s,
                    token_count=final_token_count,
                    nfe=latest_nfe,
                    engine_decode_tps=latest_engine_decode_tps,
                )

    if not raw_collected.startswith("Request failed:"):
        raw_collected = clean_final_assistant_text(raw_collected)
    st.session_state.messages.append({"role": "assistant", "content": raw_collected})


if __name__ == "__main__":
    main()
