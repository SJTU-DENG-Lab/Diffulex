# Getting Started

Start here for a practical path from environment setup to a working Diffulex
run.

- [Quickstart](quickstart.md): install, run a limited benchmark, call the
  Python API, and start the HTTP server.
- [Installation](installation.md): prepare Python, CUDA, vLLM, checkpoint paths,
  and documentation build dependencies.

Before running the examples, prepare:

- a Python 3.11+ environment;
- CUDA-visible NVIDIA GPUs;
- local model checkpoint directories;
- vLLM if the selected config uses vLLM-backed layers or baselines.

Use `DATASET_LIMIT` or `--dataset-limit` for the first run. Increase limits only
after model loading and a small generation are successful.
