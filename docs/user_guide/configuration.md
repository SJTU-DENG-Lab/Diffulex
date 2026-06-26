# Configuration

Diffulex uses the same core engine fields across Python inference, HTTP
serving, and benchmark execution. Each entry point exposes those fields in a
slightly different form, but the validation rules and runtime effects are the
same.

## Engine Arguments

Engine arguments control model loading, decoding strategy, parallelism, memory
limits, KV cache layout, LoRA behavior, and runtime optimizations.

The config validator enforces relationships between these values. For example,
`block_size` must be less than or equal to `page_size`, and both must be one of
the supported page/block sizes for the selected model family.

## Engine Parameter Reference

| Key | How to set it | What it does |
| --- | --- | --- |
| `model` / `model_path` | Point to the local model checkpoint directory. This is required. | Loads the base model. Python config uses `model`; benchmark YAML uses `engine.model_path`. |
| `tokenizer_path` | Point to a tokenizer directory, or leave it `null` to reuse the model path. | Lets benchmark flows use a tokenizer stored separately from the weights. |
| `model_name` | Choose one of the registered model keys: `dream`, `sdar`, `sdar_moe`, `fast_dllm_v2`, `llada`, `llada2`, `llada2_moe`, `llada2_mini`, `llada2dot1_mini`, `llada2_mini_dmax`, or `diffusion_gemma`. The default is `dream`. | Selects the model adapter and the sampler defaults that go with it. |
| `decoding_strategy` | Use `d2f`, `multi_bd`, `dmax`, or `diffusion_gemma`. The default is `d2f`. | Chooses the request type, scheduler, KV cache manager, model runner, and attention metadata path. |
| `sampling_mode` | Use `naive` for the standard sampler or `edit` for edit-style decoding. The default is `naive`. | Selects sampler behavior. `edit` is only valid for compatible LLaDA2-family models. |
| `mask_token_id` | Use the tokenizer's mask token ID. The default is `151666`, and tokenizer metadata can override it. | Tells diffusion samplers which token represents a masked position. |
| `tensor_parallel_size` | Use `1` to `8` tensor-parallel ranks. Core config defaults to `2`; CLI examples usually start at `1`. | Splits one model replica across multiple GPUs. |
| `data_parallel_size` | Use `1` to `1024` data-parallel groups. The default is `1`. | Runs independent serving or evaluation groups for higher throughput. |
| `gpu_memory_utilization` | Use a fractional target such as `0.9`. | Guides engine memory planning so it does not reserve the entire GPU. |
| `max_model_len` | Use a positive sequence length. The default is `2048`, and the HF model config may clamp it lower. | Sets the requested maximum prompt-plus-output length. |
| `max_num_batched_tokens` | Use a positive token budget. The default is `4096`, and it must be at least the effective `max_model_len`. | Limits how many tokens the scheduler can place in one batch. |
| `max_num_reqs` | Use a positive request count. The default is `128`. | Caps the number of active requests the engine can track. |
| `block_size` | Use `4`, `8`, `16`, or `32` for most models; `diffusion_gemma` uses `256`. The default is `32`. | Sets the token span of one diffusion block. |
| `buffer_size` | Use a positive block count. The default is `4`; `diffusion_gemma` is forced to `1`. | Controls how many diffusion blocks can be active for one request. |
| `page_size` | Use `4`, `8`, `16`, or `32` for most models; `diffusion_gemma` uses `256`. Keep it greater than or equal to `block_size`. | Sets the KV cache page size. |
| `kv_cache_layout` | Use `unified` unless a strategy or experiment needs `distinct`. | Chooses how KV cache storage is organized internally. |
| `attn_impl` | Use `triton_grouped` for normal serving and benchmark runs. `triton` and `naive` are compatibility/debug fallbacks and are not recommended for performance reporting. The default is `triton_grouped`. | Selects the attention backend. |
| `enable_prefix_caching` | Leave it `True` for compatible strategies. `d2f` forces it off during normalization. | Reuses compatible prefix KV cache state across requests. |
| `enforce_eager` | Set `True` while debugging; leave `False` for optimized runs. | Disables graph-style optimized execution paths. |
| `enable_full_static_runner` | Leave `True` for supported optimized multi-block paths. | Enables full-static CUDA graph runner paths. |
| `enable_torch_compile` | Leave `True` when the model path supports compile; turn it off for debugging. | Enables `torch.compile` where Diffulex can use it safely. |
| `torch_compile_mode` | Defaults to `reduce-overhead`; use another PyTorch compile mode only when profiling justifies it. | Passes the compile mode through to PyTorch. |
| `enable_vllm_layers` | Leave `True` unless isolating a layer implementation issue. | Uses optional vLLM-backed common layers. |

## MoE and Token Merge Parameters

| Key | How to set it | What it does |
| --- | --- | --- |
| `token_merge_mode` | Use `dmax_topk` or `iter_smooth_topk`. The default is `dmax_topk`. | Chooses how token-merge metadata is built. |
| `token_merge_top_k` | Use a positive integer. The default is `1`. | Keeps this many candidate tokens in token-merge metadata. |
| `token_merge_renormalize` | Leave `True` unless an experiment needs raw probabilities. | Renormalizes token-merge probabilities after candidate filtering. |
| `token_merge_weight` | Use a non-negative float. The default is `1.0`. | Weights the token-merge interpolation. |
| `moe_gemm_impl` | Use `triton`, `vllm`, `vllm_modular`, or `naive`. The default is `triton`. | Selects the MoE GEMM implementation. |

## DMax and Edit Sampling Parameters

| Key | How to set it | What it does |
| --- | --- | --- |
| `max_post_edit_steps` | Use a positive integer. The default is `16`. | Caps the number of edit steps per block in DMax/edit-sampling runs. |
| `penalty_lambda` | Use a non-negative float. The default is `0.0`. | Penalty weight for token-to-token edit selection. |
| `dmax_sampler_fast_path` | Leave `True` unless debugging. The default is `True`. | Enables the fast argmax path for DMax confidence computation. |
| `dmax_force_prefill_active` | Leave `False` unless profiling a specific DMax prefill pattern. | Forces DMax active blocks to remain on the prefill execution path. |
| `enable_vectorized_sampler` | Leave `False` unless extending the vectorized sampler. | Enables the vectorized greedy sampler for batched mask-filling. |
| `enable_vectorized_sampler_compile` | Leave `False` unless paired with `enable_vectorized_sampler`. | Applies `torch.compile` to the vectorized sampler path. |

## Fast-dLLM-v2 Parameters

| Key | How to set it | What it does |
| --- | --- | --- |
| `fdv2_use_block_cache` | Leave `True` for the standard dual-cache path. The default is `True`. | When enabled, already-refined sub-blocks within the buffer reuse cached KV — only the active sub-block is recomputed. Set `False` to re-compute the full buffer every sub-block step. |

## DiffusionGemma Sampler Controls

DiffusionGemma uses a canvas-denoising sampler with its own hyperparameters.
These are only active when `model_name=diffusion_gemma`:

| Key | How to set it | What it does |
| --- | --- | --- |
| `diffusion_gemma_max_denoising_steps` | Use a positive integer. The default is `48`. | Caps the number of denoising steps per block. |
| `diffusion_gemma_stability_threshold` | Use a positive integer. The default is `2`. | Number of consecutive argmax matches required for convergence. |
| `diffusion_gemma_confidence_threshold` | Use a float from `0` to `1`. The default is `0.1`. | Mean entropy threshold for accepting convergence. |
| `diffusion_gemma_t_min` | Use a float. The default is `0.0`. | Minimum temperature in the denoising schedule. |
| `diffusion_gemma_t_max` | Use a float. The default is `1.0`. | Maximum (initial) temperature in the denoising schedule. |
| `diffusion_gemma_entropy_bound` | Use a float from `0` to `1`. The default is `1.0`. | Entropy threshold for stochastic re-masking during denoising. |

## MoE Dispatching

| Key | How to set it | What it does |
| --- | --- | --- |
| `expert_parallel_size` | Use `1` for standard MoE; increase only when extending the runtime. The default is `1`. | Number of expert-parallel groups for MoE models. |
| `moe_dispatcher_backend` | Use `standard`, `naive`, or `deepep`. The default is `standard`. | Selects the MoE token dispatcher. |
| `deepep_mode` | Use `auto` unless debugging DeepEP-specific behavior. | Controls the DeepEP dispatch mode when `moe_dispatcher_backend=deepep`. |
| `deepep_num_max_dispatch_tokens_per_rank` | Use a positive integer. The default is `256`. | Caps tokens per rank in DeepEP dispatch. |

## Distributed and Deployment

| Key | How to set it | What it does |
| --- | --- | --- |
| `master_addr` | Use an IP or hostname. The default is `"localhost"`. | PyTorch distributed master address. |
| `master_port` | Use an integer. The default is `2333`. | PyTorch distributed master port. |
| `distributed_backend` | Use `nccl` unless debugging with `gloo`. | PyTorch distributed communication backend. |
| `distributed_timeout_seconds` | Use a positive integer. The default is `600`. | Timeout for distributed operations. |
| `device_start` | Use `0` unless mapping specific GPUs. | Starting CUDA device index for local ranks. |

## Runtime Tuning

| Key | How to set it | What it does |
| --- | --- | --- |
| `skip_warmup` | Leave `False` unless iterating on config changes. The default is `False`. | Skips model and CUDA graph warmup at engine startup. |
| `enable_cudagraph_torch_compile` | Leave `False` unless profiling shows a measurable gain. | Applies `torch.compile` to CUDA graph captured regions. |
| `enable_prefill_cudagraph` | Leave `False` unless prefill latency is the bottleneck. | Enables CUDA graph capture for prefill steps. |
| `prefill_cudagraph_max_len` | Use `0` (auto-detect) or a positive token count. | Maximum sequence length for prefill CUDA graph capture. |
| `auto_max_nfe_warmup_steps` | Use a positive integer. The default is `8`. | Number of warmup steps when auto-computing max NFE. |
| `auto_max_nfe_tpf_floor` | Use a positive float. The default is `1.0`. | Minimum TPF used when auto-computing max NFE. |
| `num_pages` | Use `-1` for auto-sizing, or a specific page count. | Overrides automatic KV cache page pool sizing. |
| `k_cache_hdim_split_factor_x` | Leave at the default `8` unless tuning the KV cache layout. | Splits the K cache head dimension for memory layout optimization. |

## Profiler

| Key | How to set it | What it does |
| --- | --- | --- |
| `profiler_config` | Use `null` unless collecting a profiling trace. | PyTorch profiler configuration dict. Set to a dict with `activities`, `schedule`, etc. to enable profiling. |

## Strategy Defaults

Some settings are normalized based on the selected strategy:

| Strategy | Normalized behavior |
| --- | --- |
| `d2f` | Forces `multi_block_prefix_full=True` and disables prefix caching. |
| `multi_bd` | Enables block-causal Multi-Block Diffusion and forces `multi_block_prefix_full=False`. |
| `dmax` | Forces `multi_block_prefix_full=False` and requires edit sampling. |
| `diffusion_gemma` | Uses DiffusionGemma request/sampler/runtime defaults. |

Model-specific defaults may also apply. `diffusion_gemma` uses block and page
size `256`, uses `buffer_size=1`, and enables DiffusionGemma sampler controls.

## Sampling Parameters

Sampling parameters are passed through `diffulex.SamplingParams` for Python
inference and through matching CLI/config fields for benchmark and server paths.
They are request-level settings rather than engine construction settings.

| Key | How to set it | What it does |
| --- | --- | --- |
| `temperature` | Use `0.0` for deterministic evaluation, or a higher float when sampling is desired. | Controls generation randomness. |
| `max_tokens` | Use a positive output-token limit. | Caps generated tokens for each request. |
| `max_nfe` | Use a positive integer, or leave it unset. | Caps forward evaluations when the strategy supports that limit. |
| `ignore_eos` | Leave `False` for normal generation; set `True` only when a task should continue after EOS. | Controls whether EOS ends generation. |
| `max_repetition_run` | Use a positive integer, or leave it unset. | Stops generation after a long repeated-token run. |

## Decoding Thresholds

Diffulex groups decoding thresholds in `DecodingThresholds`:

| Key | How to set it | What it does |
| --- | --- | --- |
| `add_block_threshold` | Start from the default `0.1`; tune as a float for block-add behavior. | Controls when a strategy may add another decoding block. |
| `semi_complete_threshold` | Start from the default `0.9`; tune as a float for block advancement. | Controls when semi-complete block state can move forward. |
| `accept_threshold` | Use a confidence value from `0` to `1`. The default is `0.9`. | Accepts mask-to-token updates once confidence is high enough. |
| `edit_threshold` | Use a confidence value from `0` to `1`. The default is `0.0`. | Accepts token-to-token edits in edit-style decoding. |
| `remask_threshold` | Use a confidence value from `0` to `1`. The default is `0.4`. | Remasks filled tokens that fall below the confidence threshold. |
| `token_stability_threshold` | Use a stability ratio from `0` to `1`. The default is `0.0`. | Requires enough token stability before DMax-style edit blocks advance. |

The flat CLI flags are folded into the threshold object during config
construction. Keep thresholds in the config file when comparing strategies so
command lines stay readable.

## LoRA

| Key | How to set it | What it does |
| --- | --- | --- |
| `use_lora` | Set `True` when an adapter should be loaded. | Enables LoRA adapter loading. |
| `lora_path` | Point to the adapter checkpoint directory. Required when `use_lora=True`. | Provides the adapter weights. |
| `pre_merge_lora` | Set `True` when the adapter should be merged into the base model at load time. | Avoids per-forward adapter compute when the model and adapter support merging. |

When `use_lora=True`, `lora_path` must be provided. If `pre_merge_lora=True`,
Diffulex attempts to merge adapter weights into the base model before inference
when the model path and runtime support it.

## Runtime Optimizations

| Key | How to set it | What it does |
| --- | --- | --- |
| `enforce_eager` | Set `True` while debugging; keep `False` for optimized runs. | Bypasses graph-style execution paths. |
| `enable_full_static_runner` | Leave `True` for supported multi-block optimized paths. | Enables the full-static runner where available. |
| `enable_torch_compile` | Leave `True` when the model supports compile; disable it to isolate compile issues. | Enables `torch.compile` where supported. |
| `torch_compile_mode` | Defaults to `reduce-overhead`. Change it only for a measured profiling reason. | Passes the compile mode to PyTorch. |
| `enable_vllm_layers` | Leave `True` unless comparing layer implementations. | Uses optional vLLM-backed common layers. |

Use eager mode while debugging. Enable CUDA graph and compile paths for
throughput measurements after the model and strategy are already validated.

## Benchmark YAML Structure

Benchmark YAML files use nested `engine` and `eval` sections:

```yaml
engine:
  model_path: /path/to/model
  tokenizer_path: null
  model_name: dream
  decoding_strategy: multi_bd
  tensor_parallel_size: 1
  data_parallel_size: 1
eval:
  dataset_name: gsm8k_diffulex
  dataset_limit: 10
  temperature: 0.0
  max_tokens: 512
```

`engine` fields are forwarded to the Diffulex engine after compatibility
normalization. `eval` fields control lm-eval task selection, sampling limits,
and output behavior.

## Evaluation Parameter Reference

| Key | How to set it | What it does |
| --- | --- | --- |
| `dataset_name` | Use an lm-eval task name. The default is `gsm8k_diffulex`. | Selects the benchmark task. |
| `dataset_split` | Use the dataset split name expected by the task. The default is `test`. | Selects the dataset split passed to lm-eval. |
| `dataset_limit` | Use a positive integer for smoke tests, or `null` for the full task. | Limits how many examples are evaluated. |
| `include_path` | Point to a directory of task YAMLs, leave it `null` for bundled tasks, or use an empty string to disable the bundled include path. | Controls where lm-eval looks for task definitions. |
| `dataset_data_files` | Point to a local data file, or leave it `null`. | Overrides `dataset_kwargs.data_files` in the task YAML. |
| `temperature` | Use a sampling temperature. The default is `0.0` for deterministic evaluation. | Sets generation randomness during benchmark requests. |
| `max_tokens` | Use a positive output-token limit. The config class defaults to `256`; the sample YAML uses `512`. | Caps generated tokens per example. |
| `max_nfe` | Use a positive integer, or leave it `null`. | Caps forward evaluations when a strategy supports that limit. |
| `max_repetition_run` | Use a positive integer, or leave it `null`. | Stops generation after a long repeated-token run. |
| `ignore_eos` | Leave `False` unless a task needs generation to continue past EOS. | Controls whether EOS terminates generation. |
| `output_dir` | Point to an output directory. The default is `benchmark_results`. | Sets where benchmark artifacts are written. |
| `use_run_subdirectory` | Leave `True` for normal runs. | Writes each run under a timestamped task directory. |
| `save_results` | Leave `True` unless only logs are needed. | Saves lm-eval results and sample outputs. |
| `confirm_run_unsafe_code` | Leave `True` only for tasks where executing generated code is expected and acceptable. | Allows lm-eval code-execution tasks to run. |

## Environment Variables

Diffulex uses normal Python, PyTorch, CUDA, and distributed runtime environment
variables. Set variables such as CUDA visibility and library paths before
starting the Python process.

When `CUDA_VISIBLE_DEVICES` is set, `device_ids` should refer to PyTorch logical
device IDs, not physical GPU IDs.
