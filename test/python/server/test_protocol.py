from __future__ import annotations

from diffulex.sampling_params import SamplingParams
from diffulex.server.protocol import (
    ChatInput,
    PromptInput,
    ServingDelta,
    ServingGenerate,
    ServingReply,
    serving_command_from_dict,
    serving_command_to_dict,
    serving_event_from_dict,
    serving_event_to_dict,
)


def test_serving_generate_protocol_round_trips_chat_input():
    command = ServingGenerate(
        rid="rid-1",
        input=ChatInput([{"role": "user", "content": "hi"}]),
        sampling_params=SamplingParams(max_tokens=7, temperature=0.0, max_nfe=16),
        stream=True,
        stream_mode="block_append",
        user="debug-user",
        created_time=123.0,
    )

    decoded = serving_command_from_dict(serving_command_to_dict(command))

    assert isinstance(decoded, ServingGenerate)
    assert decoded.rid == "rid-1"
    assert isinstance(decoded.input, ChatInput)
    assert decoded.input.messages == [{"role": "user", "content": "hi"}]
    assert decoded.sampling_params.max_tokens == 7
    assert decoded.sampling_params.max_nfe == 16
    assert decoded.stream is True
    assert decoded.user == "debug-user"


def test_serving_generate_protocol_round_trips_prompt_ids():
    command = ServingGenerate(
        rid="rid-2",
        input=PromptInput([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=3),
    )

    decoded = serving_command_from_dict(serving_command_to_dict(command))

    assert isinstance(decoded, ServingGenerate)
    assert isinstance(decoded.input, PromptInput)
    assert decoded.input.prompt == [1, 2, 3]
    assert decoded.stream_mode == "denoise"
    assert decoded.sampling_params.max_nfe == 512


def test_serving_event_protocol_round_trips_reply_and_delta():
    reply = ServingReply(rid="rid-1", text="ok", token_ids=[1, 2], nfe=3, finish_reason="stop")
    delta = ServingDelta(rid="rid-1", token_offset=0, text="o", token_ids=[1], nfe=1)

    decoded_reply = serving_event_from_dict(serving_event_to_dict(reply))
    decoded_delta = serving_event_from_dict(serving_event_to_dict(delta))

    assert decoded_reply == reply
    assert decoded_delta == delta
