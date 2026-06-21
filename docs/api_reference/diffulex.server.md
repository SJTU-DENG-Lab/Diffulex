# diffulex.server

`diffulex.server` is the online serving layer. It turns HTTP requests into
engine commands, forwards them to the backend worker, and converts backend
events back into plain JSON or server-sent events.

The package has two serving paths:

| Path | When it is used | Main modules |
| --- | --- | --- |
| HTTP + ZMQ backend | The normal `diffulex.server` CLI path. The FastAPI process and the engine process communicate through ZMQ queues. | `api_server`, `frontend`, `backend_worker`, `launch`, `protocol`, `zmq_queue` |
| In-process async loop | Useful for tests or embedders that want to own a single Python process. Engine calls still run through a one-worker executor. | `engine_loop` |

The HTTP surface exposes `POST /generate`, `GET /v1/models`, and
`POST /v1/chat/completions`. Streaming responses are sent as SSE frames and can
use either block append events or denoise buffer snapshots.

## diffulex.server.api_server

`api_server` builds the FastAPI application. It owns request validation,
OpenAI-compatible chat response shaping, SSE formatting, and translation from
HTTP payloads to `ServingGenerate` commands.

| Symbol | How to use it | What it does |
| --- | --- | --- |
| `GenerateRequest` | Use it as the request model for `/generate`. | Accepts a string prompt or token IDs, plus sampling and streaming fields. |
| `ChatMessage` | Use it inside `ChatCompletionRequest.messages`. | Stores one chat message with `role` and `content`. |
| `ChatCompletionRequest` | Use it as the request model for `/v1/chat/completions`. | Accepts chat messages and the same sampling/streaming controls as `/generate`. |
| `sampling_params_from_request` | Pass a generate or chat request model. | Builds `SamplingParams` from request fields such as `temperature`, `max_tokens`, `max_nfe`, and `ignore_eos`. |
| `create_app` | Pass a `FrontendManager`. | Returns the FastAPI app and wires startup/shutdown hooks to the frontend. |
| `chat_delta_chunk` | Pass a `ServingDelta`, request, and model id. | Builds an OpenAI-style streaming chat delta chunk. |
| `chat_finish_chunk` | Pass a final `ServingReply`. | Builds the final OpenAI-style streaming chat chunk and usage shell. |
| `denoise_chat_event` | Pass a `ServingDelta`, `ServingBufferSnapshot`, or `ServingReply`. | Wraps denoise events in chat-completion metadata while preserving Diffulex-specific fields. |

`stream_mode="block_append"` is closest to normal token streaming: only appended
text deltas are emitted. `stream_mode="denoise"` keeps the diffusion decoding
state visible by sending buffer snapshots as the model edits a block.

## diffulex.server.args

`args` defines the CLI schema used by `diffulex.server`. `ServerArgs`
keeps web-server options and engine options in one dataclass, then exposes
`engine_kwargs()` for constructing `DiffulexEngine`.

| Symbol | How to use it | What it does |
| --- | --- | --- |
| `parse_device_ids` | Pass a comma-separated string such as `0,1,2,3`. | Converts it to logical CUDA device IDs; empty input becomes an empty list. |
| `ServerArgs` | Construct directly in tests, or receive it from `parse_args`. | Stores host/port, ZMQ addresses, model identity, parallelism, cache, attention, MoE, threshold, and LoRA options. |
| `ServerArgs.engine_kwargs` | Call before creating the engine. | Returns only the engine-facing subset and fills threshold defaults when a CLI value is omitted. |
| `build_arg_parser` | Use when extending the server CLI. | Creates the `argparse.ArgumentParser` and declares allowed values for fields such as `sampling_mode`, `attn_impl`, and MoE backends. |
| `parse_args` | Pass an optional argv sequence. | Parses CLI flags and returns `ServerArgs`. |

When adding a new serving flag, add it to `ServerArgs`, `build_arg_parser`, and
`parse_args`, then decide whether it belongs in `engine_kwargs()`. Web-only
fields such as `host` and `port` should stay out of the engine kwargs.

## diffulex.server.backend_worker

`backend_worker` runs the synchronous engine process used by the CLI server. It
receives serialized `ServingCommand` objects, advances the engine, and pushes
serialized `ServingEvent` objects back to the frontend.

| Symbol | How to use it | What it does |
| --- | --- | --- |
| `default_engine_factory` | Leave as the default unless tests inject a fake engine. | Imports built-in strategies and constructs `DiffulexEngine`. |
| `SyncBackendWorker` | Create through `from_zmq` for the normal server path. | Owns one engine instance and one blocking receive/send loop. |
| `SyncBackendWorker.from_zmq` | Pass model, engine kwargs, command address, and event address. | Builds the worker with ZMQ pull/push queues and protocol serializers. |
| `SyncBackendWorker.run_forever` | Call inside the backend process. | Initializes the engine, serves commands until shutdown, and closes resources. |
| `run_sync_backend_worker` | Use as the multiprocessing target. | Constructs the worker, reports startup errors through `ready_queue`, and enters `run_forever()`. |

The backend treats `ServingShutdown` as a control command and forwards all other
commands to `DiffulexEngine.run_serving_tick`.

## diffulex.server.engine_loop

`engine_loop` is an in-process alternative to the ZMQ worker. It is useful when
an application wants async admission and streaming, but does not want to start a
separate backend process.

| Symbol | How to use it | What it does |
| --- | --- | --- |
| `QueuedCommand` | Internal queue item. | Wraps a `ServingCommand` before the loop admits it. |
| `default_engine_factory` | Default constructor hook. | Imports strategies and creates `DiffulexEngine`. |
| `EngineLoop` | Construct with a model path and engine kwargs, then `await start()`. | Owns one engine in a one-worker executor and serializes all engine mutations. |
| `EngineLoop.generate` | Pass prompt text or token IDs and `SamplingParams`. | Queues a non-streaming request and awaits the final `ServingReply`. |
| `EngineLoop.generate_stream` | Pass prompt, sampling params, and optional disconnect callback. | Yields `ServingDelta`, `ServingBufferSnapshot`, `ServingReply`, or `ServingError` events. |
| `EngineLoop.render_chat_prompt` | Pass chat messages. | Delegates chat-template rendering to the loaded engine. |
| `EngineLoop.stop` | Call during shutdown. | Stops the loop, fails pending waiters, exits the engine, and closes the executor. |

All engine calls run through `call_engine`, so FastAPI-style async request
handling does not mutate `DiffulexEngine` concurrently.

## diffulex.server.frontend

`frontend` is the async bridge between HTTP handlers and the backend process. It
tracks per-request event queues and aborts backend work when the client
disconnects before completion.

| Symbol | How to use it | What it does |
| --- | --- | --- |
| `ClientDisconnected` | Catch it at the HTTP layer. | Signals that the client went away while the request was still active. |
| `FrontendReqState` | Internal request state. | Stores buffered events and an `asyncio.Event` used to wake waiters. |
| `FrontendManager` | Create directly with queues or via `from_zmq`. | Sends commands, listens for backend events, and maps events back to request IDs. |
| `FrontendManager.from_zmq` | Pass model id and ZMQ addresses. | Creates async push/pull queues using the protocol serializers. |
| `FrontendManager.generate` | Pass a `ServingGenerate`. | Waits until a final reply or error arrives. |
| `FrontendManager.generate_stream` | Pass a `ServingGenerate`. | Yields backend events as they arrive and aborts incomplete requests on exit. |
| `FrontendManager.abort_request` | Pass a request ID. | Sends a `ServingAbort` command to the backend. |

The frontend creates request IDs with the `diffulex-` prefix and keeps request
state only while the request is active.

## diffulex.server.launch

`launch` is the CLI entry point. It parses `ServerArgs`, resolves IPC
addresses, starts the backend process, builds the FastAPI app, and runs Uvicorn.

| Symbol | How to use it | What it does |
| --- | --- | --- |
| `default_ipc_addr` | Pass parsed server args and a name such as `commands`. | Creates an `ipc://` address under the system temporary directory. |
| `resolve_zmq_addrs` | Pass `ServerArgs`. | Uses explicit ZMQ addresses when provided, otherwise creates default IPC addresses. |
| `start_backend` | Pass args and resolved ZMQ addresses. | Starts the synchronous backend in a spawned process and waits for readiness. |
| `main` | Used by the CLI module. | Runs the complete HTTP server lifecycle. |

The default address scheme is local IPC. Use `--zmq-command-addr` and
`--zmq-event-addr` only when the frontend and backend need explicit custom
transport addresses.

## diffulex.server.protocol

`protocol` defines the typed messages that cross the frontend/backend boundary.
The dataclasses are intentionally simple because they are serialized to dicts
and packed with msgpack before being sent over ZMQ.

| Symbol | How to use it | What it does |
| --- | --- | --- |
| `PromptInput` | Store a string prompt or token ID list. | Represents `/generate` input. |
| `ChatInput` | Store a list of `{role, content}` messages. | Represents chat-completion input before template rendering. |
| `ServingGenerate` | Send from frontend to backend. | Starts a generation request with sampling params, streaming mode, user, and timestamp metadata. |
| `ServingAbort` | Send from frontend to backend. | Requests cancellation for one request ID. |
| `ServingShutdown` | Send during frontend shutdown. | Asks the backend worker to stop. |
| `ServingReply` | Send from backend to frontend. | Represents the final generated text, token IDs, NFE count, and finish reason. |
| `ServingDelta` | Send during `block_append` streaming. | Represents newly appended text and token IDs with an offset. |
| `ServingBufferSnapshot` | Send during `denoise` streaming. | Represents the current editable buffer span and text after a denoising update. |
| `ServingError` | Send when backend work fails. | Carries an error message for one request ID. |
| `serving_command_to_dict` / `serving_command_from_dict` | Use at queue boundaries. | Serialize and restore frontend-to-backend commands. |
| `serving_event_to_dict` / `serving_event_from_dict` | Use at queue boundaries. | Serialize and restore backend-to-frontend events. |

Each command and event exposes `request_id` through the underlying `rid`, which
keeps the HTTP, frontend, backend, and engine layers aligned on one identifier.

## diffulex.server.zmq_queue

`zmq_queue` wraps pyzmq PUSH/PULL sockets with msgpack serialization. The
module has synchronous classes for the backend worker and async classes for the
FastAPI frontend.

| Symbol | How to use it | What it does |
| --- | --- | --- |
| `ZmqPushQueue` | Use in synchronous code that sends objects. | Encodes an object to a dict, packs it with msgpack, and sends it through a PUSH socket. |
| `ZmqPullQueue` | Use in synchronous code that receives objects. | Receives a msgpack payload from a PULL socket and decodes it back to a typed object. |
| `ZmqAsyncPushQueue` | Use in async code that sends objects. | Async PUSH equivalent of `ZmqPushQueue`. |
| `ZmqAsyncPullQueue` | Use in async code that receives objects. | Async PULL equivalent of `ZmqPullQueue`. |

All queue classes accept a `create` flag. When `create=True`, the socket binds
the address; when `create=False`, it connects to an address owned by another
process.
