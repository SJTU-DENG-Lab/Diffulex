# Home

<figure style="text-align:center">
  <img src="_static/img/logo-v2.png" alt="Diffulex" style="width: 60%; max-width: 720px;" />
</figure>

<p style="text-align:center">
<strong>Diffusion Language Model Serving Engine</strong>
</p>

<p style="text-align:center">
<a href="https://github.com/SJTU-DENG-Lab/Diffulex">
  <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Diffulex-181717?logo=github" style="margin: 0 4px;" />
</a>
<a href="https://discord.gg/NSa9WH4EKu">
  <img alt="Discord" src="https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white" style="margin: 0 4px;" />
</a>
</p>

Diffulex is a Diffusion Language Model Serving Engine built on
PagedAttention-style runtime primitives. It provides a unified engine for KV
cache management, block scheduling, prefix reuse, MoE execution, CUDA graph
replay, and model-specific diffusion samplers.

Diffulex is also the runtime engine behind the **Multi-Block Diffusion Language
Models (MBD-LMs)** line of work. Native Block Diffusion LMs perform
**Single-Block Diffusion (SingleBD)**: each forward pass refines one noisy block
conditioned on a clean cached prefix. This preserves KV caching but leaves
blocks sequential, creating a store bubble where the GPU runs a forward that
produces no new output. **Multi-Block Diffusion (MultiBD)** removes this
bottleneck by maintaining a bounded running-set of consecutive blocks, enabling
decode-store overlap and inter-block parallelism. MBD-LMs are BD-LMs
post-trained with **Multi-block Teacher Forcing (MultiTF)** so the model can
handle practical MultiBD running-set states — and Diffulex executes them with an
optimized **Block Buffer** runtime that preserves static input shapes for CUDA
Graph replay. In the engine, MultiBD is exposed as `decoding_strategy=multi_bd`.

For reproducing the MBD-LMs experiments, use the Diffulex
[`mbd-lms`](https://github.com/SJTU-DENG-Lab/Diffulex/tree/mbd-lms) branch
(CUDA 12). For engine development, open-source contributions, or exploring new
decoding algorithms and turning them into runnable systems, use the
[`main`](https://github.com/SJTU-DENG-Lab/Diffulex/tree/main) branch. `main`
contains ongoing runtime and model-specific optimizations, so its behavior and
performance profile may differ from the experiment reproduction branch.
**The `main` branch requires CUDA 13.**

## Where to Start

| Goal | Start here |
| --- | --- |
| Understand MultiBD in the engine | [Multi-Block Diffusion](user_guide/features/multi_block_decoding.md) |
| Install Diffulex and run one command | [Quickstart](get_started/quickstart.md) |
| Set up Python, CUDA, and vLLM dependencies | [Installation](get_started/installation.md) |
| Run GSM8K or other lm-eval benchmarks | [Benchmark](user_guide/benchmark.md) |
| Start the HTTP server | [Server](user_guide/server.md) |
| Tune engine or YAML parameters | [Configuration](user_guide/configuration.md) |
| Use Diffulex as a research backend | [Research Engine](developer_guide/research_engine.md) |
| Add a model, strategy, or kernel | [Developer Guide](developer_guide/index.md) |

## Current Scope

Diffulex focuses on cache-aware block-wise dLLM decoding. The main supported
runtime pieces are:

- PagedAttention-style KV cache management for diffusion decoding.
- Strategy-specific schedulers and request state.
- Prefix caching for block-causal Multi-Block Diffusion.
- Tensor and data parallel inference paths.
- Optional vLLM-backed common layers and MoE kernels.
- Benchmark and HTTP serving entry points.

For new algorithms, Diffulex `main` is intended to be a research backend rather
than only a benchmark runner. Its Block Buffer, paged KV cache, scheduler,
sampler, and Triton kernel boundaries are designed so block-level generation
ideas can be implemented as strategy components. See
[Research Engine](developer_guide/research_engine.md) for the implementation
map.

## Model Families

| Model family | `model_name` | Typical strategy | Status |
| --- | --- | --- | --- |
| Dream / D2F-Dream | `dream` | `d2f` | Supported |
| DiffuCoder / D2F-DiffuCoder | `diffucoder` | `d2f` | Supported |
| Dream reasoner | `dream_reasoner` | `multi_bd` | Supported |
| Stable-DiffCoder | `stable_diffcoder` | `multi_bd` | Supported |
| LLaDA / D2F-LLaDA | `llada` | `d2f` | Supported |
| Fast-dLLM-v2 | `fast_dllm_v2` | `multi_bd` or `fast_dllm_v2` | Supported |
| SDAR | `sdar` | `multi_bd` | Supported |
| SDAR-MoE | `sdar_moe` | `multi_bd` | Supported |
| LLaDA2 family | `llada2`, `llada2_mini`, `llada2_moe`, `llada2dot1_mini`, `llada2_mini_dmax` | `multi_bd`, `dmax`, or `fast_dllm_v2` | Supported |
| DiffusionGemma | `diffusion_gemma` | `diffusion_gemma` | Supported |

Use [Models](user_guide/models.md) for compatibility details before mixing
model names, strategies, and sampling modes.

:::{toctree}
:hidden:
:maxdepth: 1
:caption: GETTING STARTED

get_started/quickstart
get_started/installation
get_started/index
:::

:::{toctree}
:hidden:
:maxdepth: 1
:caption: TUTORIALS

tutorials/model_loading_configuration
tutorials/benchmark_workflow
tutorials/http_serving_streamlit
tutorials/adding_new_model_family
:::

:::{toctree}
:hidden:
:maxdepth: 2
:caption: USER GUIDE

user_guide/configuration
user_guide/models
user_guide/benchmark
user_guide/server
user_guide/features/index
user_guide/troubleshooting
user_guide/index
:::

:::{toctree}
:hidden:
:maxdepth: 2
:caption: DEVELOPER GUIDE

developer_guide/the_design
developer_guide/research_engine
developer_guide/extending_the_engine
developer_guide/testing
developer_guide/developer_troubleshooting
developer_guide/profiling
developer_guide/index
:::

:::{toctree}
:hidden:
:maxdepth: 2
:caption: API REFERENCE

api_reference/diffulex
api_reference/diffulex_bench
api_reference/diffulex_kernel
:::

:::{toctree}
:hidden:
:maxdepth: 1
:caption: CLI REFERENCE

cli_reference/diffulex.server
cli_reference/diffulex.bench
:::
