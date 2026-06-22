# diffulex.server

`diffulex.server` launches an HTTP service backed by a Diffulex engine process.
Use it when an application, UI, or test client needs request/response access
instead of in-process Python generation.

The entry point is:

```bash
python -m diffulex.server --help
```

## Minimal Command

```bash
python -m diffulex.server \
  --model /path/to/LLaDA2.0-mini \
  --model-name llada2_mini \
  --decoding-strategy multi_bd \
  --sampling-mode naive \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-reqs 1 \
  --block-size 32 \
  --buffer-size 1 \
  --page-size 32 \
  --host 0.0.0.0 \
  --port 8000
```

The server starts a frontend process and a synchronous backend worker. The
frontend exposes HTTP routes; the backend owns the Diffulex engine and model
execution. ZMQ addresses are generated automatically unless explicitly provided.

## Network Arguments

| Flag | How to set it | What it does |
| --- | --- | --- |
| `--host` | Use an interface such as `127.0.0.1` for local access or `0.0.0.0` to listen on all interfaces. The default is `0.0.0.0`. | Sets the HTTP bind host. |
| `--port` | Use an available TCP port. The default is `8000`. | Sets the HTTP bind port. |
| `--log-level` | Use a uvicorn log level such as `info`, `warning`, or `debug`. The default is `info`. | Controls server log verbosity. |
| `--zmq-command-addr` | Leave empty for automatic setup, or provide a ZMQ address. | Sets the optional frontend-to-backend command channel. |
| `--zmq-event-addr` | Leave empty for automatic setup, or provide a ZMQ address. | Sets the optional backend-to-frontend event channel. |

Most local runs should leave the ZMQ addresses unset.

## Model and Strategy Arguments

| Flag | How to set it | What it does |
| --- | --- | --- |
| `--model` | Point to the local base-model checkpoint directory. This flag is required. | Loads model weights for the engine backend. |
| `--model-name` | Use a registered model key. The default is `dream`. | Selects model adapter and sampler defaults. |
| `--decoding-strategy` | Use `d2f`, `multi_bd`, `dmax`, or `diffusion_gemma` where supported by the selected model/config. | Chooses the strategy-specific request, scheduler, cache, runner, and attention metadata path. |
| `--sampling-mode` | Use `naive` for the standard sampler or `edit` for compatible edit-sampling models. The default is `naive`. | Selects sampler behavior. |

## Parallelism and Device Arguments

| Flag | How to set it | What it does |
| --- | --- | --- |
| `--tensor-parallel-size` | Use `1` to `8` ranks. The server default is `1`. | Splits one model replica across multiple GPUs. |
| `--data-parallel-size` | Use `1` to `1024` groups. The default is `1`. | Runs independent worker groups for serving throughput. |
| `--master-addr` | Use the host address for distributed initialization. The default is `localhost`. | Tells distributed workers where to rendezvous. |
| `--master-port` | Use an available port from `1` to `65535`. The default is `2333`. | Sets the distributed rendezvous port. |
| `--distributed-timeout-seconds` | Use a positive timeout in seconds. The default is `600`. | Bounds how long distributed setup may wait. |
| `--device-ids` | Provide comma-separated logical CUDA IDs, or leave empty. | Limits the server to selected PyTorch-visible devices. |

## Capacity and Layout Arguments

| Flag | How to set it | What it does |
| --- | --- | --- |
| `--block-size` | Use `4`, `8`, `16`, or `32` for most models; DiffusionGemma uses `256`. The default is `32`. | Sets the token span of one diffusion block. |
| `--buffer-size` | Use a positive block count. The default is `4`. | Controls how many diffusion blocks can remain active for one request. |
| `--page-size` | Use `4`, `8`, `16`, or `32` for most models; DiffusionGemma uses `256`. Keep it at least as large as `--block-size`. | Sets the KV cache page size. |
| `--max-model-len` | Use a positive sequence length. The default is `2048`, and the HF config may clamp it. | Sets the requested prompt-plus-output length limit. |
| `--max-num-batched-tokens` | Use a positive token budget. The default is `4096`; it must cover the effective model length. | Limits scheduler batch size by token count. |
| `--max-num-reqs` | Use a positive request count. The default is `128`. | Caps active requests tracked by the server. |
| `--gpu-memory-utilization` | Use a fraction such as `0.9`. | Guides GPU memory planning. |
| `--kv-cache-layout` | Use `unified` for the default layout or `distinct` for strategy experiments. | Chooses KV cache storage layout. |

## Runtime Toggles

| Flag | How to set it | What it does |
| --- | --- | --- |
| `--disable-full-static-runner` | Add the flag when isolating full-static runner issues. | Disables the supported full-static CUDA Graph runner path. |
| `--disable-torch-compile` | Add the flag while debugging compile-related behavior. | Disables `torch.compile` where it would otherwise be used. |
| `--torch-compile-mode` | Defaults to `reduce-overhead`; use another PyTorch compile mode only when profiling calls for it. | Passes compile mode through to PyTorch. |
| `--enforce-eager` | Add during correctness debugging. Leave it off for optimized throughput checks. | Forces eager execution and bypasses graph-style optimizations. |
| `--attn-impl` | Use `triton_grouped` for normal serving and performance reports. `triton` and `naive` are compatibility/debug fallbacks. The default is `triton_grouped`. | Selects the server attention backend. |
| `--disable-prefix-caching` | Add only when debugging cache behavior or comparing without reuse. | Turns off compatible prefix cache reuse. |

Use `--enforce-eager` while debugging model or scheduler behavior. Remove it
when measuring optimized throughput.

## LoRA Arguments

| Flag | How to set it | What it does |
| --- | --- | --- |
| `--use-lora` | Add the flag when serving with an adapter. | Enables LoRA adapter loading. |
| `--lora-path` | Point to the adapter checkpoint directory. Required with `--use-lora`. | Loads the adapter weights. |
| `--pre-merge-lora` | Add when the adapter should be merged into the base model at load time. | Avoids per-forward adapter compute when the model path supports merging. |

## Threshold Arguments

| Flag | How to set it | What it does |
| --- | --- | --- |
| `--add-block-threshold` | Omit it to use `0.1`, or pass a float for block-add tuning. | Controls when another decoding block can be added. |
| `--semi-complete-threshold` | Omit it to use `0.9`, or pass a float for block advancement tuning. | Controls when semi-complete block state can advance. |
| `--accept-threshold` | Use a confidence value from `0` to `1`. The default is `0.9`. | Accepts mask-to-token updates once confidence is high enough. |
| `--remask-threshold` | Use a confidence value from `0` to `1`. The default is `0.4`. | Remasks filled tokens that fall below the confidence threshold. |
| `--token-stability-threshold` | Use a stability ratio from `0` to `1`. The default is `0.0`. | Controls edit-block progress for DMax-style decoding. |

## Validation Tips

Start with conservative limits such as low `--max-num-reqs`,
`--max-num-batched-tokens`, and `--max-model-len`. Once the server starts and a
small request succeeds, increase limits for the target workload.
