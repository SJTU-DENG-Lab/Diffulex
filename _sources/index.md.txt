# 👋 Welcome to Diffulex

[GitHub](https://github.com/SJTU-DENG-Lab/Diffulex)

Diffulex is a Paged Attention-based block-wise dLLM accelerated decoding inference framework that is easy to develop and extensible. The design maximizes hiding the complexity of underlying KV Cache management, parallel strategy scheduling, and memory optimization. By providing a clean and unified API interface along with flexible inference strategy configurations (e.g., D2F, Block Diffusion, Fast-dLLM), Diffulex allows developers to focus on model inference logic and business requirements while maintaining production-level inference performance and resource utilization efficiency.

## Supported Models

Currently supported models: D2F-LLaDA, D2F-Dream, Fast-dLLM-v2, SDAR, SDAR-MoE.

Models in progress: D2F-DiffuCoder, LLaDA2, LLaDA2.1, LLaDA2-DMax, Stable-DiffCoder.

Diffulex does not plan to support full-attention dLLM models, including the original LLaDA, Dream, LLaDA1.5, and DiffuCoder variants.

:::{toctree}
:maxdepth: 2
:caption: GET STARTED
get_started/index
cookbook/index
:::

:::{toctree}
:maxdepth: 1
:caption: TUTORIALS
tutorials/index
:::

:::{toctree}
:maxdepth: 1
:caption: PROGRAMMING GUIDES
:::

:::{toctree}
:maxdepth: 1
:caption: DEEP LEARNING OPERATORS
:::

:::{toctree}
:maxdepth: 1
:caption: COMPILER INTERNALS
:::

:::{toctree}
:maxdepth: 1
:caption: Privacy

privacy
:::
