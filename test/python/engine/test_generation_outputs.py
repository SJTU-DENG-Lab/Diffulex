from types import SimpleNamespace

from diffulex.utils.output import GenerationOutputs


def test_generation_outputs_handles_empty_prefill_suffix() -> None:
    outputs = GenerationOutputs(1)
    req = SimpleNamespace(
        req_id=0,
        is_prefilling=True,
        new_tokens=0,
        running_sequence=[],
        block_size=4,
        dllm_block_buffer=SimpleNamespace(dllm_blocks=[]),
        truncated_response=[],
        full_response=[],
        is_truncated=False,
        max_new_tokens_reached=False,
        max_model_len_reached=False,
        max_nfe_reached=False,
        max_repetition_run_reached=False,
        eos_token_generated=False,
        completion_reason=None,
    )

    outputs.record_step([req], step_time=1.0)

    assert outputs.prefill_throughput == 0
    assert outputs.postfix()["tps"] == "0.00tok/s"
    assert outputs.postfix()["ptps"] == "0.00tok/s"


def test_generation_outputs_decode_throughput_uses_batch_time() -> None:
    outputs = GenerationOutputs(2)
    shared_buffer = SimpleNamespace(dllm_blocks=[])
    reqs = [
        SimpleNamespace(
            req_id=0,
            is_prefilling=False,
            new_tokens=1,
            running_sequence=[1, 2, 3, 4],
            block_size=4,
            dllm_block_buffer=shared_buffer,
            truncated_response=[],
            full_response=[],
            is_truncated=False,
            max_new_tokens_reached=False,
            max_model_len_reached=False,
            max_nfe_reached=False,
            max_repetition_run_reached=False,
            eos_token_generated=False,
            completion_reason=None,
        ),
        SimpleNamespace(
            req_id=1,
            is_prefilling=False,
            new_tokens=1,
            running_sequence=[5, 6, 7, 8],
            block_size=4,
            dllm_block_buffer=shared_buffer,
            truncated_response=[],
            full_response=[],
            is_truncated=False,
            max_new_tokens_reached=False,
            max_model_len_reached=False,
            max_nfe_reached=False,
            max_repetition_run_reached=False,
            eos_token_generated=False,
            completion_reason=None,
        ),
    ]

    outputs.record_step(reqs, step_time=1.0)

    assert outputs.decode_throughput == 2.0
    assert outputs.tpf == 1.0
    assert outputs.throughput == 2.0
    assert outputs.total_time == 1.0
    assert outputs.postfix()["tpf"] == "1.00tok/step"
    assert outputs.postfix()["dtps"] == "2.00tok/s"
    assert outputs.postfix()["tps"] == "2.00tok/s"


def test_generation_outputs_prefill_throughput_uses_batch_time() -> None:
    outputs = GenerationOutputs(2)
    shared_buffer = SimpleNamespace(dllm_blocks=[])
    reqs = [
        SimpleNamespace(
            req_id=0,
            is_prefilling=True,
            new_tokens=0,
            running_sequence=[1, 2, 3, 4],
            block_size=4,
            dllm_block_buffer=shared_buffer,
            truncated_response=[],
            full_response=[],
            is_truncated=False,
            max_new_tokens_reached=False,
            max_model_len_reached=False,
            max_nfe_reached=False,
            max_repetition_run_reached=False,
            eos_token_generated=False,
            completion_reason=None,
        ),
        SimpleNamespace(
            req_id=1,
            is_prefilling=True,
            new_tokens=0,
            running_sequence=[5, 6, 7, 8],
            block_size=4,
            dllm_block_buffer=shared_buffer,
            truncated_response=[],
            full_response=[],
            is_truncated=False,
            max_new_tokens_reached=False,
            max_model_len_reached=False,
            max_nfe_reached=False,
            max_repetition_run_reached=False,
            eos_token_generated=False,
            completion_reason=None,
        ),
    ]

    outputs.record_step(reqs, step_time=2.0)

    assert outputs.prefill_throughput == 4.0
    assert outputs.total_time == 2.0
    assert outputs.postfix()["ptps"] == "4.00tok/s"
    assert outputs.postfix()["tps"] == "0.00tok/s"


def test_generation_outputs_latency_metrics_are_per_request_means() -> None:
    outputs = GenerationOutputs(2)
    shared_buffer = SimpleNamespace(dllm_blocks=[])

    req0_prefill = SimpleNamespace(
        req_id=0,
        is_prefilling=True,
        new_tokens=0,
        running_sequence=[1, 2],
        block_size=2,
        dllm_block_buffer=shared_buffer,
        truncated_response=[],
        full_response=[],
        is_truncated=False,
        max_new_tokens_reached=False,
        max_model_len_reached=False,
        max_nfe_reached=False,
        max_repetition_run_reached=False,
        eos_token_generated=False,
        completion_reason=None,
    )
    req1_prefill = SimpleNamespace(
        req_id=1,
        is_prefilling=True,
        new_tokens=0,
        running_sequence=[3, 4],
        block_size=2,
        dllm_block_buffer=shared_buffer,
        truncated_response=[],
        full_response=[],
        is_truncated=False,
        max_new_tokens_reached=False,
        max_model_len_reached=False,
        max_nfe_reached=False,
        max_repetition_run_reached=False,
        eos_token_generated=False,
        completion_reason=None,
    )
    outputs.record_step([req0_prefill, req1_prefill], step_time=0.5)

    req0_decode = SimpleNamespace(
        req_id=0,
        is_prefilling=False,
        new_tokens=2,
        running_sequence=[1, 2],
        block_size=2,
        dllm_block_buffer=shared_buffer,
        truncated_response=[10, 11],
        full_response=[10, 11],
        is_truncated=False,
        max_new_tokens_reached=False,
        max_model_len_reached=False,
        max_nfe_reached=False,
        max_repetition_run_reached=False,
        eos_token_generated=False,
        completion_reason=None,
    )
    req1_decode = SimpleNamespace(
        req_id=1,
        is_prefilling=False,
        new_tokens=4,
        running_sequence=[3, 4],
        block_size=2,
        dllm_block_buffer=shared_buffer,
        truncated_response=[12, 13, 14, 15],
        full_response=[12, 13, 14, 15],
        is_truncated=False,
        max_new_tokens_reached=False,
        max_model_len_reached=False,
        max_nfe_reached=False,
        max_repetition_run_reached=False,
        eos_token_generated=False,
        completion_reason=None,
    )
    outputs.record_step([req0_decode, req1_decode], step_time=1.0)

    assert outputs.tpf == 1.5
    assert outputs.ttft == 1.5
    assert outputs.throughput == 4.0
    assert outputs.postfix()["ttft"] == "1.50s"
    assert outputs.postfix()["dtps"] == "6.00tok/s"
    assert outputs.postfix()["tps"] == "4.00tok/s"


def test_generation_outputs_benchmark_format_uses_nfe() -> None:
    outputs = GenerationOutputs(1)
    shared_buffer = SimpleNamespace(dllm_blocks=[])
    req = SimpleNamespace(
        req_id=0,
        is_prefilling=False,
        new_tokens=1,
        running_sequence=[1, 2, 3, 4],
        block_size=4,
        dllm_block_buffer=shared_buffer,
        truncated_response=[42],
        full_response=[42],
        is_truncated=True,
        max_new_tokens_reached=False,
        max_model_len_reached=False,
        max_nfe_reached=True,
        max_repetition_run_reached=False,
        eos_token_generated=False,
        completion_reason="max_nfe_reached",
    )

    outputs.record_step([req], step_time=1.0)
    formatted = outputs.to_benchmark_format()

    assert formatted == [
        {
            "text": "",
            "full_text": "",
            "token_ids": [42],
            "nfe": 1,
        }
    ]
