<p align="center">
  <img src="./assets/imgs/diffulex_logo_new.png" alt="Diffulex" />
</p>

<div align="center">

# Diffulex

[![Documentation](https://img.shields.io/badge/Documentation-Read%20the%20Docs-0A66C2?logo=readthedocs&logoColor=white)](https://sjtu-deng-lab.github.io/Diffulex/)
[![GitHub](https://img.shields.io/badge/GitHub-Diffulex-181717?logo=github)](https://github.com/SJTU-DENG-Lab/Diffulex)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white)](https://discord.gg/NSa9WH4EKu)

</div>

Diffulex is a PagedAttention-based inference framework for block-wise diffusion
language models. It provides a unified runtime for KV cache management, block
scheduling, prefix reuse, MoE execution, CUDA graph replay, HTTP serving, and
model-specific diffusion samplers.

Diffulex is also the runtime engine behind the **Multi-Block Diffusion Language
Models (MBD-LMs)** line of work. In the engine, **Multi-Block Diffusion
(MultiBD)** is exposed as `decoding_strategy=multi_bd`, where a bounded
running-set of consecutive diffusion blocks is decoded concurrently instead of
finishing one block at a time.

## Start Here

The README is intentionally brief. Use the documentation for installation,
configuration, benchmarks, serving, and development notes:

| Goal | Go to |
|---|---|
| Read the full documentation | [Diffulex Documentation](https://sjtu-deng-lab.github.io/Diffulex/) |
| Install Python, CUDA, and the tested `vllm==0.23.0` dependency | [Installation](https://sjtu-deng-lab.github.io/Diffulex/get_started/installation.html) |
| Run the first LLaDA2-mini command | [Quickstart](https://sjtu-deng-lab.github.io/Diffulex/get_started/quickstart.html) |
| Check supported models and strategy combinations | [Models](https://sjtu-deng-lab.github.io/Diffulex/user_guide/models.html) |
| Tune runtime and YAML parameters | [Configuration](https://sjtu-deng-lab.github.io/Diffulex/user_guide/configuration.html) |
| Run GSM8K and other benchmark workflows | [Benchmark](https://sjtu-deng-lab.github.io/Diffulex/user_guide/benchmark.html) |
| Start HTTP serving or the Streamlit demo | [Server](https://sjtu-deng-lab.github.io/Diffulex/user_guide/server.html) / [Streamlit](https://sjtu-deng-lab.github.io/Diffulex/user_guide/streamlit.html) |
| Add a model, decoding strategy, or kernel | [Developer Guide](https://sjtu-deng-lab.github.io/Diffulex/developer_guide/index.html) |

## Branches

For reproducing the MBD-LMs experiments, use the Diffulex
[`mbd-lms`](https://github.com/SJTU-DENG-Lab/Diffulex/tree/mbd-lms) branch.

For engine development, open-source contributions, or exploring new decoding
algorithms and turning them into runnable systems, use the
[`main`](https://github.com/SJTU-DENG-Lab/Diffulex/tree/main) branch. `main`
contains ongoing runtime and model-specific optimizations, so its behavior and
performance profile may differ from the experiment reproduction branch.

## Current Scope

Diffulex focuses on cache-aware block-wise dLLM decoding. The current engine
supports D2F-style decoding, block-causal MultiBD, DMax token-merge decoding,
DiffusionGemma canvas decoding, vLLM-backed common layers/MoE kernels, benchmark
entry points, and HTTP serving.

Supported model families include Dream/DiffuCoder-style dense dLLMs, Dream
reasoner, Stable-DiffCoder, LLaDA, Fast-dLLM-v2, SDAR, SDAR-MoE, LLaDA2, and
DiffusionGemma. See the [Models documentation](https://sjtu-deng-lab.github.io/Diffulex/user_guide/models.html)
for the up-to-date compatibility matrix.

## Discussion

For questions, development discussion, and collaboration, join the Discord:

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20Us-blue?logo=discord&style=for-the-badge)](https://discord.gg/NSa9WH4EKu)

## Acknowledgments

We would like to thank [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm),
[vLLM](https://github.com/vllm-project/vllm),
[mini-sglang](https://github.com/sgl-project/mini-sglang),
[SGLang](https://github.com/sgl-project/sglang), and
[dInfer](https://github.com/inclusionAI/dInfer), whose designs informed parts
of Diffulex's early backend, paged attention, serving architecture, and dLLM
inference optimizations. Diffulex is developed by the DENG Lab at Shanghai Jiao
Tong University.
