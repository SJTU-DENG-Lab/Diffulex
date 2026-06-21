# Installation

Diffulex is currently installed from source. Select the setup path that matches
your environment.

## Supported Environments

- Primary runtime target: NVIDIA GPUs.
- Tested NVIDIA devices include H200, H100, A100, RTX 4090, and RTX 3090.
- Python 3.11 or newer is required.
- Model checkpoints are loaded from local directories.

Diffulex depends on PyTorch, Transformers, vLLM, and CUDA-compatible GPU
runtime libraries. If PyTorch or vLLM cannot detect your accelerator, fix that
environment first before debugging Diffulex.

## Set Up Using Python

Create a dedicated Python environment, then install Diffulex in editable mode.

`````{tabs}

````{tab} NVIDIA CUDA

This is the recommended path for running Diffulex.

```bash
uv venv --python 3.11 --seed
source .venv/bin/activate
uv pip install -e .
uv pip install vllm==0.19.1 --torch-backend=auto
```

`uv` can select a matching PyTorch backend with `--torch-backend=auto`. If your
cluster requires a specific CUDA wheel, replace `auto` with the matching backend
for your driver and CUDA stack.

```bash
uv pip install vllm==0.19.1 --torch-backend=cu126
```

````

````{tab} Other Environment

Currently, only CUDA Linux environments are fully supported.

````

`````

## Verify the Install

Check that the package imports:

```bash
python -c "from diffulex import Diffulex, SamplingParams; print('ok')"
```

Check that PyTorch can see CUDA devices:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

The import is intentionally lightweight. Model weights are loaded only when a
`Diffulex` engine is constructed.

## Build the Documentation

Diffulex documentation is built with Sphinx:

```bash
cd docs
../.venv/bin/python -m sphinx -b html . _build/html
```

The generated site is written to `docs/_build/html`.
