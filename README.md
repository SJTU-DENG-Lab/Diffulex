<img src="./assets/imgs/diffulex_logo_new.png" alt="Diffulex" />

<div align="center">

# Diffulex

[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/NSa9WH4EKu)

</div>

Diffulex is a Paged-Attention-based inference framework for block-wise diffusion language models. It provides a unified engine for KV cache management, block scheduling, prefix reuse, MoE execution, CUDA graph replay, and model-specific diffusion samplers, while keeping decoding strategies configurable from a small set of runtime options.

Diffulex is also the runtime engine behind the **Multi-Block Diffusion Language Models (MBD-LMs)** line of work. In that context, **Multi-Block Diffusion (MultiBD)** means decoding a bounded running-set of consecutive diffusion blocks concurrently, instead of finishing one block at a time as in Single-Block Diffusion (SingleBD). In Diffulex, the runtime option for this method is `decoding_strategy=multi_bd`.

For reproducing the MBD-LMs experiments, use the Diffulex [`mbd-lms`](https://github.com/SJTU-DENG-Lab/Diffulex/tree/mbd-lms) branch. For engine development, open-source contributions, or exploring new decoding algorithms and turning them into runnable systems, use the [`main`](https://github.com/SJTU-DENG-Lab/Diffulex/tree/main) branch. `main` contains ongoing runtime and model-specific optimizations, so its behavior and performance profile may differ from the experiment reproduction branch.

The current codebase supports several dLLM model families and multiple decoding strategies, including D2F-style decoding, block-causal Multi-Block Diffusion, DMax token-merge decoding, and DiffusionGemma's entropy-bound canvas decoder.

## Supported Models

The following `model_name` values are registered by the engine today:

| Model family | `model_name` values | Typical strategy |
|---|---|---|
| Dream / DiffuCoder style dense dLLMs | `dream`, `diffucoder` | `d2f` |
| Dream reasoner / Stable-DiffCoder | `dream_reasoner`, `stable_diffcoder` | `multi_bd` |
| LLaDA style dense dLLMs | `llada` | `d2f` |
| Fast-dLLM-v2 | `fast_dllm_v2` | `multi_bd` |
| SDAR | `sdar` | `multi_bd` |
| SDAR-MoE | `sdar_moe` | `multi_bd` |
| LLaDA2 family | `llada2`, `llada2_mini`, `llada2_moe`, `llada2dot1_mini` | `multi_bd` or `dmax` |
| DiffusionGemma | `diffusion_gemma` | `diffusion_gemma` |

Runnable benchmark configs live under `diffulex_bench/configs/`. The most actively used configs currently cover LLaDA2-mini, DiffusionGemma, SDAR, SDAR-MoE, Fast-dLLM-v2, Dream, LLaDA, and DiffuCoder-style GSM8K runs.

## Sampling Modes

`sampling_mode` currently accepts:

| Sampling mode | Status | Notes |
|---|---|---|
| `naive` | Supported | Default confidence/threshold sampler for dense and MoE dLLMs. Also used by DiffusionGemma, whose sampler is model-specific internally. |
| `edit` | Supported for LLaDA2 family | Enables edit/remask refinement after initial fill. Required by `decoding_strategy=dmax`. `llada2dot1_mini` requires this mode. |

Additional sampler controls:

| Option | Values | Notes |
|---|---|---|
| `token_merge_mode` | `dmax_topk`, `iter_smooth_topk` | Used by DMax/token-merge decoding. |
| `enable_vectorized_sampler` | `true`, `false` | Enables the vectorized LLaDA2 sampler path when available. |
| DiffusionGemma sampler controls | `diffusion_gemma_*` options | Configure denoising steps, stability threshold, temperature range, confidence threshold, and entropy bound. |

## Decoding Strategies

Set `decoding_strategy` in the engine config:

| Strategy | Status | Main use | Important behavior |
|---|---|---|---|
| `d2f` | Supported | D2F / full-prefix block decoding | Forces `multi_block_prefix_full=true` and disables prefix caching. |
| `multi_bd` | Supported | Multi-Block Diffusion (MultiBD) | Uses block-causal multi-block decoding, keeps a bounded active block set, and supports prefix caching. This is the default high-throughput path for LLaDA2/SDAR-style models. |
| `dmax` | Supported | LLaDA2 token-merge decoding | Requires `sampling_mode=edit`; supports `dmax_topk` and `iter_smooth_topk` token merge modes. |
| `diffusion_gemma` | Supported | DiffusionGemma block/canvas decoding | Automatically selected for `model_name=diffusion_gemma`; uses 256-token canvas blocks and DiffusionGemma entropy-bound sampling. |

## Strategy Compatibility

Recommended combinations:

| Model family | Recommended config |
|---|---|
| Dream / DiffuCoder / LLaDA | `decoding_strategy=d2f`, `sampling_mode=naive` |
| Dream reasoner / Stable-DiffCoder | `decoding_strategy=multi_bd`, `sampling_mode=naive` |
| Fast-dLLM-v2 / SDAR / SDAR-MoE | `decoding_strategy=multi_bd`, `sampling_mode=naive` |
| LLaDA2 / LLaDA2-mini / LLaDA2-MoE | `decoding_strategy=multi_bd`, `sampling_mode=naive` |
| LLaDA2 DMax experiments | `decoding_strategy=dmax`, `sampling_mode=edit`, `token_merge_mode=dmax_topk` |
| LLaDA2.1-mini | `decoding_strategy=multi_bd`, `sampling_mode=edit` |
| DiffusionGemma | `model_name=diffusion_gemma`; the engine forces `decoding_strategy=diffusion_gemma` |

## Roadmap

Current planned work is focused on two areas:

| Area | Goal |
|---|---|
| Dual Cache | Implement a Dual Cache mechanism for strategies that need separate cache views or cache lifecycles. |
| Performance | Further optimize runtime overhead, sampler kernels, CUDA graph coverage, and model-specific hot paths. |

## Installation

Install from source:

```bash
uv pip install -e .
```

Install the tested vLLM build separately when using vLLM-backed layers, attention kernels, or MoE kernels:

```bash
uv pip install vllm==0.23.0
```

Diffulex currently validates its vLLM-backed paths against `vllm==0.23.0`. Use
other vLLM versions only when you are intentionally testing compatibility.

## Quick Start

Run the current LLaDA2-mini GSM8K preset:

```bash
CUDA_VISIBLE_DEVICES=0 script/run_llada2_mini_gsm8k.sh
```

Run the vLLM DiffusionGemma GSM8K preset:

```bash
CUDA_VISIBLE_DEVICES=0 \
CONFIG_PATH=examples/engine_lm_eval/configs/vllm_diffusion_gemma_gsm8k_full.yml \
script/run_vllm_diffusion_gemma_gsm8k.sh
```

Run through the benchmark entrypoint with a config:

```bash
python -m diffulex_bench.main --config diffulex_bench/configs/llada2_mini_gsm8k.yml
```

Useful config examples:

| Config | Purpose |
|---|---|
| `diffulex_bench/configs/llada2_mini_gsm8k.yml` | LLaDA2-mini with `multi_bd` and `naive` sampling. |
| `diffulex_bench/configs/llada2_mini_dmax_gsm8k.yml` | LLaDA2-mini with `dmax` and `edit` sampling. |
| `diffulex_bench/configs/diffusion_gemma_gsm8k.yml` | DiffusionGemma with the dedicated `diffusion_gemma` strategy. |
| `diffulex_bench/configs/sdar_chat_gsm8k.yml` | SDAR with `multi_bd`. |
| `diffulex_bench/configs/sdar_moe_chat_gsm8k.yml` | SDAR-MoE with `multi_bd`. |
| `diffulex_bench/configs/fast_dllm_v2_gsm8k.yml` | Fast-dLLM-v2 with `multi_bd`. |

More runtime and serving notes are in `docs/cookbook/`.

## Tested Devices

Diffulex has been tested on NVIDIA H200, H100, A100, RTX 4090, and RTX 3090 GPUs. Kernel availability and performance depend on the selected model, tensor parallel size, MoE backend, and vLLM/Triton build.

## Project Notes

Diffulex targets block-wise and cache-aware diffusion decoding. It does not aim to be a generic full-attention dLLM serving stack for every original checkpoint; support is added model by model through the model, sampler, and strategy registries.

## Join the Discussion

Welcome to join our Discord community for discussions, support, and collaboration.

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20Us-blue?logo=discord&style=for-the-badge)](https://discord.gg/NSa9WH4EKu)

## Acknowledgments

We would like to thank [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), [vLLM](https://github.com/vllm-project/vllm), [mini-sglang](https://github.com/sgl-project/mini-sglang), [SGLang](https://github.com/sgl-project/sglang), and [dInfer](https://github.com/inclusionAI/dInfer), whose designs informed parts of Diffulex's early backend, paged attention, serving architecture, and dLLM inference optimizations. Diffulex is developed by the DENG Lab at Shanghai Jiao Tong University.
