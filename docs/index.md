# Home

<figure style="text-align:center">
  <img src="_static/img/logo-v2.png" alt="Diffulex" style="width: 60%; max-width: 720px;" />
</figure>

<p style="text-align:center">
<strong>Block-wise diffusion LLM inference engine</strong>
</p>

<p style="text-align:center">
<a href="https://github.com/SJTU-DENG-Lab/Diffulex">
  <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Diffulex-181717?logo=github" style="margin: 0 4px;" />
</a>
<a href="https://discord.gg/NSa9WH4EKu">
  <img alt="Discord" src="https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white" style="margin: 0 4px;" />
</a>
</p>

Diffulex is a PagedAttention-based inference framework for block-wise diffusion
language models. It provides a unified engine for KV cache management, block
scheduling, prefix reuse, MoE execution, CUDA graph replay, and model-specific
diffusion samplers.

For reproducing the MBD LMS experiments, use the `mbd-lms` branch. The current
main branch contains ongoing runtime and model-specific optimizations, so its
behavior and performance profile may differ from the experiment reproduction
branch.

## Where to Start

| Goal | Start here |
| --- | --- |
| Install Diffulex and run one command | [Quickstart](get_started/quickstart.md) |
| Set up Python, CUDA, and vLLM dependencies | [Installation](get_started/installation.md) |
| Run GSM8K or other lm-eval benchmarks | [Benchmark](user_guide/benchmark.md) |
| Start the HTTP server | [Server](user_guide/server.md) |
| Tune engine or YAML parameters | [Configuration](user_guide/configuration.md) |
| Add a model, strategy, or kernel | [Developer Guide](developer_guide/index.md) |

## Current Scope

Diffulex focuses on cache-aware block-wise dLLM decoding. The main supported
runtime pieces are:

- PagedAttention-style KV cache management for diffusion decoding.
- Strategy-specific schedulers and request state.
- Prefix caching for block-causal multi-block decoding.
- Tensor and data parallel inference paths.
- Optional vLLM-backed common layers and MoE kernels.
- Benchmark and HTTP serving entry points.

## Model Families

| Model family | `model_name` | Typical strategy | Status |
| --- | --- | --- | --- |
| Dream / D2F-Dream | `dream` | `d2f` | Supported |
| DiffuCoder / D2F-DiffuCoder | `diffucoder` | `d2f` | Supported |
| Dream reasoner | `dream_reasoner` | `multi_bd` | Supported |
| Stable-DiffCoder | `stable_diffcoder` | `multi_bd` | Supported |
| LLaDA / D2F-LLaDA | `llada` | `d2f` | Supported |
| Fast-dLLM-v2 | `fast_dllm_v2` | `multi_bd` | Supported |
| SDAR | `sdar` | `multi_bd` | Supported |
| SDAR-MoE | `sdar_moe` | `multi_bd` | Supported |
| LLaDA2 family | `llada2`, `llada2_mini`, `llada2_moe`, `llada2dot1_mini` | `multi_bd` or `dmax` | Supported |
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
user_guide/streamlit
user_guide/features/index
user_guide/troubleshooting
user_guide/index
:::

:::{toctree}
:hidden:
:maxdepth: 1
:caption: COOKBOOK

cookbook/index
:::

:::{toctree}
:hidden:
:maxdepth: 2
:caption: DEVELOPER GUIDE

developer_guide/the_design
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
