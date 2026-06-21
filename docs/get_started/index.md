---
orphan: true
---

# Getting Started

Start here if you want the shortest path to installing and running Diffulex.
This section is intentionally practical: it covers installation, a first local
generation run, serving, and a minimal benchmark command.

## Recommended Path

- [Quickstart](quickstart.md)
- [Installation](installation.md)

Read [Installation](installation.md) first if the Python environment or CUDA
runtime is not ready. Read [Quickstart](quickstart.md) first if Diffulex is
already installed and you want a working command path.

## What You Need

Before running the examples, prepare:

- a Python environment with Diffulex installed from this repository;
- CUDA-visible NVIDIA GPUs;
- a local model checkpoint directory;
- a tokenizer path, if it differs from the model path;
- an optional LoRA checkpoint for D2F examples.

Start with low request and token limits until the model loads and one request
completes.

For deeper workflow explanations, see the [Tutorials](../tutorials/index.md).
