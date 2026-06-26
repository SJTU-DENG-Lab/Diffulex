# Troubleshooting

Start with the narrowest failing command and verify the runtime layer before
debugging Diffulex-specific behavior. Most failures fall into environment,
configuration, model loading, scheduler capacity, or serving lifecycle issues.

## Import Errors

Confirm that Diffulex is installed in the active Python environment:

```bash
python -c "from diffulex import Diffulex, SamplingParams; print('ok')"
```

If this fails, check that the shell is using the expected virtual environment
and that the package was installed from the repository root.

Also check lightweight kernel imports:

```bash
python -c "import diffulex_kernel; print('ok')"
```

This import should not eagerly load all optional kernels.

## CUDA Availability

Confirm that PyTorch can see the expected GPUs:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

If CUDA is unavailable, fix the PyTorch/CUDA installation before changing
Diffulex settings. If the device count is lower than expected, inspect
`CUDA_VISIBLE_DEVICES` and cluster scheduler allocation.

## Model Loading

Check that model and tokenizer paths are local directories. Diffulex validates
the model directory and then loads Hugging Face config and tokenizer metadata.

For LoRA runs:

| Setting | What to check |
| --- | --- |
| `use_lora` / `--use-lora` | Enable adapter loading only when a LoRA checkpoint should be used. |
| `lora_path` / `--lora-path` | Point to the adapter checkpoint directory when LoRA is enabled. |
| Adapter and base model | Confirm the adapter was trained for the selected base model family. |

If startup fails after model loading begins, retry with `tensor_parallel_size=1`
and `data_parallel_size=1` to separate model compatibility from distributed
topology problems.

## Configuration Errors

Validation errors usually name the invalid field. Common examples:

| Field or condition | Constraint |
| --- | --- |
| `block_size`, `page_size` | `block_size` must be less than or equal to `page_size`, and both values must be supported for the selected model family. |
| `decoding_strategy="dmax"` | Requires `sampling_mode="edit"` and a compatible model name. |
| Parallel world size | Must not exceed the number of visible CUDA devices. |
| `max_num_batched_tokens`, `max_model_len` | `max_num_batched_tokens` must be at least the effective `max_model_len`. |

Fix the first validation error before looking at later symptoms.

## Serving

Use a small `max_num_reqs`, `max_num_batched_tokens`, and `max_model_len` while
validating a new serving command.

If the HTTP server starts but the client cannot connect, check the host, port,
and client base URL. If the server exits during startup, inspect backend worker
logs and reduce engine limits.

## Benchmarking

Use `--dataset-limit` when testing a new config. If lm-eval cannot find a task,
check `--dataset` and `--include-path`. If code tasks fail before scoring, check
whether `--confirm-run-unsafe-code` is required.

## Performance Problems

First confirm correctness with eager mode and small batches. Then measure one
change at a time:

| Change | What it measures |
| --- | --- |
| Remove `--enforce-eager` | Measures optimized execution after eager-mode correctness is established. |
| Enable CUDA graph paths | Measures launch-overhead reduction. |
| Increase request and token limits | Measures scheduler and memory behavior under larger serving load. |
| Increase tensor or data parallelism | Measures multi-GPU scaling after single-device behavior is stable. |

Avoid changing model family, strategy, thresholds, and optimization flags in the
same experiment.
