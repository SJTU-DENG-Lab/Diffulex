# DiffusionGemma and Optional vLLM Layer Backend Notes

This note records the current implementation state and the checks to rerun when
moving this branch to another machine.

## Current Goals

The current branch is doing two related pieces of work:

1. Add `diffusion_gemma` as a model/sampler path that runs through the existing
   `multi_bd` backend with 256-token Gemma blocks.
2. Make common non-attention layers optionally use vLLM layer implementations
   when the runtime has a complete vLLM install, while preserving Diffulex's
   public layer APIs and checkpoint layout.

The vLLM layer work is intentionally conservative. We do not directly replace
Diffulex linear, embedding, lm-head, MoE, attention, or loader semantics with
vLLM modules yet. Those modules carry model-parallel, quantization, parameter,
and loader conventions that would affect every model.

## DiffusionGemma Path

Main files:

- `diffulex/model/diffusion_gemma.py`
- `diffulex/sampler/diffusion_gemma.py`
- `diffulex/model/config/diffusion_gemma/`
- `diffulex_bench/configs/diffusion_gemma_gsm8k.yml`
- `diffulex_bench/tasks/gsm8k/gsm8k_diffusion_gemma.yaml`
- `diffulex_bench/tasks/gsm8k/diffusion_gemma_utils.py`

Important runtime assumptions:

- `model_name="diffusion_gemma"`
- `decoding_strategy="multi_bd"`
- `block_size=256`
- `page_size=256`
- `buffer_size=1`
- `multi_block_prefix_full=false`

`Config.__post_init__` forces the Gemma block/page/buffer values for
`model_name="diffusion_gemma"`. The checkpoint generation config is also read
for DiffusionGemma sampler values such as max denoising steps, t range,
stability threshold, confidence threshold, and entropy bound.

The default local bench config uses:

- `tensor_parallel_size: 1`
- `max_model_len: 768`
- `max_num_reqs: 4`
- `max_num_batched_tokens: 3072`
- `max_tokens: 256`

Do not blindly copy vLLM serve's `--max-model-len 262144` into this offline
bench config. Diffulex cache/page budgeting is different, and a huge model len
can waste memory or break scheduling before it helps the GSM8K path.

## Optional vLLM Layer Backend

Main files:

- `diffulex/layer/vllm_backend.py`
- `diffulex/layer/activation.py`
- `diffulex/layer/layernorm.py`
- `diffulex/layer/rotary_embedding.py`
- `diffulex/config.py`
- `diffulex/model/auto_model.py`
- `diffulex_bench/arg_parser.py`
- `diffulex_bench/config.py`
- `diffulex_bench/main.py`

The config switch is:

```yaml
engine:
  enable_vllm_layers: true
```

or on the bench CLI:

```bash
--enable-vllm-layers
--no-enable-vllm-layers
```

`AutoModelForDiffusionLM.from_config()` calls
`set_vllm_layers_enabled(config.enable_vllm_layers)` before model
construction. Common layer constructors then try to instantiate vLLM layer
backends. If import or construction fails, they fall back to Diffulex native
implementations.

There is no environment variable switch for this anymore. The only remaining
layer debug env var is `DIFFULEX_REFERENCE_RMSNORM`, which forces the RMSNorm
reference path for debugging.

Currently covered optional backends:

- `SiluAndMul`
- `GeluAndMul(approximate="tanh")`
- `RMSNorm`, including `has_weight=False`
- RoPE through a small adapter around vLLM `get_rope`

Fallback behavior:

- Activation fallback uses Diffulex `torch.compile` wrappers.
- RMSNorm fallback uses Diffulex `torch.compile` wrappers.
- RoPE fallback is now in-place, matching vLLM's operational style.
- Partial RoPE only modifies the rotary slice and leaves the pass-through tail
  unchanged.
- Gemma4 proportional RoPE has a Diffulex fallback and optional vLLM backend.

Important caveat: on the current source machine, direct vLLM layer import is
not complete in the active Python environment. Adding `/data/jyj/vllm` to
`PYTHONPATH` starts importing vLLM but fails on missing dependencies such as
`cbor2`. Therefore the measured numbers below are for Diffulex fallback
implementations, not vLLM CustomOps.

## Verification Commands

Run the layer precision tests:

```bash
PYTHONPATH=. pytest -q test/python/layer/test_common_layer_precision.py
```

Expected result on the source machine:

```text
11 passed
```

Run model compatibility tests:

```bash
PYTHONPATH=. pytest -q \
  test/python/model/test_llada_model.py \
  test/python/model/test_sdar_moe.py \
  test/python/model/test_llada2_moe.py \
  test/python/engine/test_diffusion_gemma_block.py
```

Expected result on the source machine:

```text
33 passed
```

These tests check structure, loader aliases, tensor shapes, sampler behavior,
and layer-level numeric correctness for the common operators. They are not a
full end-to-end model quality test.

## Layer Precision Coverage

`test/python/layer/test_common_layer_precision.py` uses dummy tensors to cover:

- `SiluAndMul` against PyTorch reference
- `GeluAndMul` against PyTorch reference
- `RMSNorm` against reference, with and without residual
- `RMSNorm(has_weight=False)`
- full RoPE, 2D and 3D input
- partial RoPE, 2D and 3D input
- partial RoPE tail preservation
- Gemma4 proportional RoPE
- RoPE in-place behavior

This test caught and fixed a real bug in `RMSNorm.add_rms_forward`: for fp32
inputs, the returned residual could alias the tensor later normalized in-place.
The fix explicitly clones the residual output.

## Microbench Method

The quick microbench used dummy tensors and compared Diffulex current fallback
paths against plain PyTorch reference formulas. It is not a CI test and has no
thresholds.

Source-machine observation:

```text
== cpu ==
SiluAndMul current              0.0511 ms
SiluAndMul reference            0.0799 ms
GeluAndMul current              0.1257 ms
GeluAndMul reference            1.2434 ms
RMSNorm current add             0.1310 ms
RMSNorm reference add           1.8473 ms
RoPE full current inplace       0.4633 ms
RoPE partial current inplace    0.4270 ms

== cuda:0 ==
SiluAndMul current              0.1622 ms
SiluAndMul reference            0.2675 ms
GeluAndMul current              0.1681 ms
GeluAndMul reference            0.2635 ms
RMSNorm current add             0.0888 ms
RMSNorm reference add           0.2293 ms
RoPE full current inplace       0.2630 ms
RoPE partial current inplace    0.1763 ms
```

Again, this does not prove vLLM CustomOps are faster. It only shows the current
Diffulex compiled/in-place fallback is faster than naive PyTorch references on
that machine.

## DiffusionGemma Bench Command

Use the bundled config for a small GSM8K run:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m diffulex_bench.main \
  --config diffulex_bench/configs/diffusion_gemma_gsm8k.yml \
  --dataset-limit 10 \
  --engine-arg tensor_parallel_size=1
```

The checkpoint path currently expected by the config is:

```text
/data/ckpts/google/diffusiongemma-26B-A4B-it
```

Adjust `model_path` in the yaml or pass the equivalent override on the new
machine.

If the GPU is already occupied, do not use the run to judge performance. The
source-machine runs were mostly fallback/eager for DiffusionGemma graph capture,
because self-conditioning is not yet wired into static CUDA graph buffers.

## What Still Needs Work

- Run the optional vLLM backend on a machine with a complete vLLM install and
  compiled custom ops.
- Add a vLLM-vs-Diffulex numeric comparison test when vLLM imports cleanly.
- Add a vLLM-vs-Diffulex microbench once vLLM CustomOps are actually used.
- Decide whether linear/embedding/lm-head should stay Diffulex-native or move
  toward vLLM parameter/loader conventions.
- Refactor model weight loading toward vLLM's model-owned `load_weights`
  iterator style. This should be a separate change because it affects all
  models and LoRA/packed mapping behavior.
