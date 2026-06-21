# Installation

Diffulex is installed from source. The supported production path today is
Linux with NVIDIA CUDA GPUs and local Hugging Face-style checkpoint
directories.

## Environment Requirements

| Component | Requirement |
| --- | --- |
| OS | Linux. Other platforms are not a supported runtime target. |
| Python | Python 3.11 or newer. |
| GPU | NVIDIA CUDA GPU visible to PyTorch. H200, H100, A100, RTX 4090, and RTX 3090 have been used in development. |
| Checkpoints | Local checkpoint directories. Diffulex examples do not download model weights at runtime. |
| vLLM | Recommended for optimized layer/MoE backends and required by some benchmark presets. |

If PyTorch or vLLM cannot see the GPU, fix that environment first. Diffulex
will not be able to recover from an invalid CUDA/PyTorch installation.

## Create the Environment

From the repository root:

```bash
uv venv --python 3.11 --seed
source .venv/bin/activate
uv pip install -e .
```

Install vLLM in the same environment if you plan to use the optimized vLLM
layer backends, MoE kernels, or the vLLM baseline scripts:

```bash
uv pip install vllm
```

On clusters with strict CUDA wheel requirements, install the PyTorch/vLLM build
that matches the driver and CUDA runtime provided by the cluster. Keep all of
Diffulex, PyTorch, and vLLM in the same Python environment unless you are only
using the external vLLM baseline launcher.

## Verify the Environment

Check that the package imports:

```bash
python -c "from diffulex import Diffulex, SamplingParams; print('diffulex ok')"
```

Check CUDA visibility:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

If vLLM-backed paths will be used, check vLLM separately:

```bash
python -c "import vllm; print('vllm ok')"
```

## Model Paths

Most configs in this repository use paths from the development cluster, for
example `/data/ckpts/inclusionAI/LLaDA2.0-mini`. Replace those paths with the
local checkpoint directory in your environment.

For one-off runs, prefer command-line overrides:

```bash
MODEL_PATH=/path/to/LLaDA2.0-mini \
DATASET_LIMIT=10 \
CUDA_VISIBLE_DEVICES=0 \
script/run_llada2_mini_gsm8k.sh
```

For repeated runs, edit or copy the YAML config under `diffulex_bench/configs/`.

## Optional vLLM Baseline Environment

The DiffusionGemma vLLM baseline launcher can use a separate editable vLLM
environment. By default it looks under `/data/jyj/vllm-env`; override that when
your vLLM checkout lives elsewhere:

```bash
VLLM_ENV_DIR=/path/to/vllm-env \
CUDA_VISIBLE_DEVICES=0 \
script/run_vllm_diffusion_gemma_gsm8k.sh
```

The vLLM install used by that script must support
`DiffusionGemmaForBlockDiffusion`.

## Build the Documentation

Install documentation dependencies, then build the Sphinx site:

```bash
uv pip install -r docs/requirements.txt
python -m sphinx -b html docs docs/_build/html
```

The generated site is written to `docs/_build/html`.
