# Models

Diffulex is built for diffusion language model inference. Model support is tied
to three pieces of configuration: the model family name, the decoding strategy,
and the sampler behavior.

## Supported Models

Diffulex currently documents support for these model families:

- D2F-LLaDA
- D2F-Dream
- Fast-dLLM-v2
- SDAR
- SDAR-MoE

Models in progress include D2F-DiffuCoder, LLaDA2, LLaDA2.1, LLaDA2-DMax, and
Stable-DiffCoder.

Diffulex does not plan to support full-attention dLLM models, including the
original LLaDA, Dream, LLaDA1.5, and DiffuCoder variants.

## Model Names

The CLI and config layer use stable `model_name` strings. Benchmark choices are
listed in `diffulex_bench.arg_parser.MODEL_NAME_CHOICES`, including:

| `model_name` | Typical use |
| --- | --- |
| `dream` | Dream-family model paths. |
| `sdar` | SDAR dense model paths. |
| `sdar_moe` | SDAR Mixture-of-Experts paths. |
| `fast_dllm_v2` | Fast-dLLM-v2 paths. |
| `llada` | LLaDA-family D2F paths. |
| `llada2` | LLaDA2 edit-sampling paths. |
| `llada2_moe` | LLaDA2 MoE edit-sampling paths. |
| `llada2_mini` | LLaDA2 mini paths. |
| `llada2dot1_mini` | LLaDA2.1 mini paths. |
| `llada2_mini_dmax` | LLaDA2 mini paths intended for DMax-style decoding. |
| `diffusion_gemma` | DiffusionGemma paths with larger block and page sizes. |

Use the model name that matches the implementation registered under
`diffulex/model/`.

## Strategy Compatibility

Not every model is valid with every decoding strategy. The config validator
enforces the combinations that would otherwise produce invalid runtime state.

| Strategy | Use it for |
| --- | --- |
| `d2f` | D2F-style LLaDA and Dream paths. |
| `multi_bd` | Multi-block diffusion models such as Fast-dLLM-v2 and SDAR-family models. |
| `dmax` | Supported edit-sampling LLaDA2-family models only. |

`diffusion_gemma` is normalized to `multi_bd` and uses larger block and page
sizes.

## Loading Requirements

Model and tokenizer paths should point to local directories. The engine loads
Hugging Face config and tokenizer metadata during startup, then builds the
registered model implementation.

Before running a full benchmark, verify that:

| Requirement | What to verify |
| --- | --- |
| Model path | The model checkpoint directory exists locally. |
| Tokenizer | The tokenizer can be loaded, either from the model path or a separate tokenizer path. |
| `model_name` | The selected model name is registered under `diffulex/model/`. |
| Sampler | The selected sampler is registered when the model needs custom sampling behavior. |
| Strategy import | The selected strategy is imported by `diffulex.strategy`. |

## Adding Support

To add a model family, implement and register the model, add a sampler if
needed, add config validation only where required, and run a small offline
generation before adding benchmark or server examples.
