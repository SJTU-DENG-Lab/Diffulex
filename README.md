<img src=./assets/imgs/diffulex_design.png />

<div align="center">

# Diffulex

[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/NSa9WH4EKu)

</div>

Diffulex is a Paged Attention-based block-wise dLLM accelerated decoding inference framework that is easy to develop and extensible. The design maximizes hiding the complexity of underlying KV Cache management, parallel strategy scheduling, and memory optimization. By providing a clean and unified API interface along with flexible inference strategy configurations (e.g., D2F, Block Diffusion, Fast-dLLM), Diffulex allows developers to focus on model inference logic and business requirements while maintaining production-level inference performance and resource utilization efficiency.

## Supported Models

Currently supported models: D2F-LLaDA, D2F-Dream, Fast-dLLM-v2, SDAR, SDAR-MoE.

Models in progress: D2F-DiffuCoder, LLaDA2, LLaDA2.1, LLaDA2-DMax, Stable-DiffCoder.

Diffulex does not plan to support full-attention dLLM models, including the original LLaDA, Dream, LLaDA1.5, and DiffuCoder variants.

## Latest News
- 12/22/2025 ✨: We are excited to announce that Diffulex, a Paged Attention-based dLLM accelerated decoding inference framework, is now open source and available to the public!

## Tested Devices
Although Diffulex aims to be portable across a range of Devices, it has been specifically tested and validated on the following devices: for NVIDIA GPUs, this includes the H200, A100, RTX 4090, RTX 3090.

## Installation
### Method 1: Install with Pip

The only way to get started is to install from source:

```bash
uv pip install -e .
```

Install vLLM manually:

```bash
uv pip install vllm==0.19.1
```

## Quick Start

For model-specific startup instructions and runnable configurations, see the [Cookbook](docs/cookbook.md).

## Upcoming Features

Check our [Diffulex v0.0.1 release plan](https://github.com/SJTU-DENG-Lab/Diffulex/issues/14) for upcoming features.

## Join the Discussion

Welcome to join our Discord community for discussions, support, and collaboration!

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20Us-blue?logo=discord&style=for-the-badge)](https://discord.gg/NSa9WH4EKu)

## Acknowledgments

We would like to express our gratitude to [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), which inspired the initial backend implementation of this project, and [vLLM](https://github.com/vllm-project/vllm), from which we draw core architectural concepts, particularly the Paged Attention mechanism. Our server design also references [mini-sglang](https://github.com/sgl-project/mini-sglang); although Diffulex has diverged significantly from these projects, we remain grateful for their contributions to the open-source inference community. The initial version of this project was mainly developed by [Yijie Jin](https://github.com/drewjin0827) with supervision from Prof. [Zhijie Deng](https://thudzj.github.io) at Shanghai Jiao Tong University.
