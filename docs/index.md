# Home

<figure style="text-align:center">
  <img src="_static/img/logo-v2.png" alt="Diffulex" style="width: 60%; max-width: 720px;" />
</figure>

<p style="text-align:center">
<strong>Flexible Diffusion LLM Inference Engine</strong>
</p>

<p style="text-align:center">
<a href="https://github.com/SJTU-DENG-Lab/Diffulex">
  <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Diffulex-181717?logo=github" style="margin: 0 4px;" />
</a>

<a href="https://github.com/SJTU-DENG-Lab/Diffulex/stargazers">
  <img alt="GitHub stars" src="https://img.shields.io/github/stars/SJTU-DENG-Lab/Diffulex?style=flat&logo=github&label=Stars" style="margin: 0 4px;" />
</a>

<a href="https://github.com/SJTU-DENG-Lab/Diffulex/forks">
  <img alt="GitHub forks" src="https://img.shields.io/github/forks/SJTU-DENG-Lab/Diffulex?style=flat&logo=github&label=Fork" style="margin: 0 4px;" />
</a>

<a href="https://discord.gg/NSa9WH4EKu">
  <img alt="Discord" src="https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white" style="margin: 0 4px;" />
</a>
</p>

**Diffulex** is a Paged Attention-based block-wise dLLM accelerated decoding
inference framework that is easy to develop and extend.

The design maximizes hiding the complexity of underlying KV Cache management,
parallel strategy scheduling, and memory optimization. By providing a clean
and unified API interface along with flexible inference strategy configurations
(e.g., D2F, Block Diffusion, Fast-dLLM), Diffulex allows developers to focus on
model inference logic and business requirements while maintaining production-level
inference performance and resource utilization efficiency.

---

Where you start depends on what you need to do:

- Run a local model with Diffulex: start with the [Quickstart Guide](get_started/quickstart.md).
- Run benchmark or serving examples: read the [Quickstart Guide](get_started/quickstart.md), then the [Tutorials](tutorials/index.md).
- Adapt Diffulex to a new model, strategy, or kernel: start with the [Developer Guide](developer_guide/index.md).

For project development and community links, see:

- [GitHub repository](https://github.com/SJTU-DENG-Lab/Diffulex)
- [Release plan](https://github.com/SJTU-DENG-Lab/Diffulex/issues/14)
- [Discord community](https://discord.gg/NSa9WH4EKu)

Diffulex focuses on:

- PagedAttention-based KV cache management for block-wise dLLM decoding.
- Flexible decoding strategy deployment, including D2F, Multi-Block Diffusion, and DMax.
- Tensor and data parallel inference paths for multi-GPU serving.
- Mixture-of-Experts model serving with expert routing.

Diffulex currently supports the following model families:

- D2F-LLaDA
- D2F-Dream
- Fast-dLLM-v2
- SDAR
- SDAR-MoE

Models in progress include D2F-DiffuCoder, LLaDA2, LLaDA2.1, LLaDA2-DMax, and
Stable-DiffCoder.

Diffulex does not plan to support full-attention dLLM models, including the
original LLaDA, Dream, LLaDA1.5, and DiffuCoder variants.

:::{toctree}
:hidden:
:maxdepth: 1
:caption: GETTING STARTED

get_started/quickstart
get_started/installation
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
user_guide/features/index
user_guide/troubleshooting
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
