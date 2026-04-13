# Async HTTP Serving Plan

This note captures the current plan for adding an sglang-style HTTP serving path to Diffulex.

## Reference Model

`/data1/jyj/mini-sglang` separates online serving into these roles:

- FastAPI frontend that owns request ids, SSE streams, and OpenAI-compatible routes.
- Tokenizer worker that converts text/chat messages to token ids.
- Scheduler worker that runs a long-lived scheduling loop and owns the engine.
- Detokenizer worker that turns generated token ids into streamable text chunks.

The important point is that async serving is mostly achieved by decoupling frontend I/O from a long-lived backend
scheduler loop. The scheduler itself is still a synchronous control loop around model execution, with non-blocking
message receive and optional CUDA stream overlap.

## Diffulex Starting Point

Diffulex currently exposes `DiffulexTPWorker.generate()` as an offline-style batch API:

1. Add all prompts to the scheduler.
2. Repeatedly call `step()` until all requests finish.
3. Accumulate `GenerationOutputs`.
4. Decode final text.

The old `mixin/async_engine` helpers wrapped synchronous methods in thread executors. That was useful for not blocking a
caller, but it was not an online serving architecture. The serving path should instead expose command/reply owner-loop
helpers that keep scheduler and model-runner mutations serialized.

## Proposed Direction

Implement the serving path in phases instead of copying Mini-SGLang's full multi-process/ZMQ design immediately.

### Phase 1: In-Process Async HTTP Server

Add a FastAPI server and one background engine loop in the same process:

- HTTP handlers allocate request ids and enqueue requests.
- A single background engine loop owns one `DiffulexTPWorker`.
- All engine mutations happen in that loop: `add_request`, `step`, scheduler postprocess, and output collection.
- Per-request futures or queues return results to the HTTP handlers.

Initial endpoints:

- `POST /generate`
- `POST /v1/chat/completions`
- `GET /v1/models`

The first implementation supports both non-streaming responses and SSE streams. This validates request admission,
batching, and engine ownership while keeping the scheduler/model runner synchronous and single-owned.

### Phase 2: Streaming

The current SSE streaming path keeps the same owner-loop model as non-streaming serving.

For diffusion/block decoding, streaming should not assume one stable token per step like autoregressive models. The
serving API exposes `stream=true` plus `stream_mode`:

- `block_append`: append-only stream. Emit tokens only after they leave the active block buffer and become
  stable/in-cache. This is the OpenAI-compatible mode for `/v1/chat/completions`.
- `denoise`: update stream. Emit the current active buffer snapshot each step with `token_offset`, `absolute_start`,
  `absolute_end`, `token_ids`, and decoded text so the frontend can replace that window and show the denoising effect.
- Both modes always send a final full result and `[DONE]`.

### Phase 3: Process Separation

Only after the in-process path is correct, consider Mini-SGLang-style process separation:

- API process
- tokenizer/detokenizer process
- scheduler/engine process
- optional DP shard processes

This can use ZMQ or the existing multiprocessing/RPC patterns, but should not be the first step because Diffulex already
has TP/DP worker process lifecycle to preserve.

## Implementation Sketch

Potential file layout:

- `diffulex/server/api_server.py`
- `diffulex/server/args.py`
- `diffulex/server/launch.py`
- `diffulex/server/engine_loop.py`
- `diffulex/mixin/async_engine/engine/serving_worker.py`

Potential command:

```bash
python -m diffulex.server.launch --model /path/to/model --port 8000
```

## Phase 1 Acceptance Criteria

- Concurrent HTTP requests do not call the same engine concurrently.
- New requests can be admitted while other requests are running.
- Non-streaming `/generate` returns correct text/token ids.
- Non-streaming `/v1/chat/completions` returns an OpenAI-like JSON payload.
- Shutdown calls `engine.exit()` and does not leave TP child processes alive.

## Open Questions

- How stable are partial outputs for multi-BD diffusion decoding at each step?
- Should chat templating live in the HTTP server or a tokenizer helper?
- Should the first server support DP, or initially require `data_parallel_size=1`?
- What request cancellation semantics should free KV cache safely?
