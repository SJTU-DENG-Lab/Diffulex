# diffulex.sampler

`diffulex.sampler` contains token update logic for diffusion language models.
Importing the package loads built-in samplers so their `AutoSampler.register(...)`
calls run.

| Module | Role |
| --- | --- |
| `diffulex.sampler.auto_sampler` | Registry and factory for sampler implementations. |
| `diffulex.sampler.base` | Base sampler classes and sample output utilities. |
| `diffulex.sampler.diffusion_gemma` | DiffusionGemma sampler registration. |
| `diffulex.sampler.dream` | Dream sampler registration. |
| `diffulex.sampler.fast_dllm_v2` | Fast-dLLM-v2 sampler registration. |
| `diffulex.sampler.llada` | LLaDA sampler registration. |
| `diffulex.sampler.llada2` | LLaDA2, edit-sampling, and DMax sampler logic. |
| `diffulex.sampler.sdar` | SDAR and SDAR-MoE sampler registration. |

## diffulex.sampler.auto_sampler

This module maps `Config.model_name` values to sampler factories. It supports
both simple class registration and factories that need the full Diffulex config.

| Symbol | Purpose |
| --- | --- |
| `AutoSampler` | Registry used by model runners to construct the configured sampler. |

New model families should register a sampler only when the token update rule
differs from existing sampler behavior.

## diffulex.sampler.base

This package provides shared sampler contracts and output structures.

| Symbol | Purpose |
| --- | --- |
| `SamplerShiftLogits` / `DllmSamplerShiftBase` | Base classes for samplers that shift logits before sampling. |
| `SamplerNoShiftLogits` / `DllmSamplerNoShiftBase` | Base classes for samplers that operate without shifted logits. |
| `SampleOutputBase` | Base sample-output dataclass. |
| `merge_sample_outputs` | Merges sample outputs from parallel paths. |

Use these base classes before adding a new model-family sampler from scratch.

## diffulex.sampler.diffusion_gemma

This module registers the DiffusionGemma sampler.

| Symbol | Purpose |
| --- | --- |
| `DiffusionGemmaSampler` | Sampler registered for `model_name="diffusion_gemma"` with full-config construction. |

## diffulex.sampler.dream

This module registers the Dream sampler.

| Symbol | Purpose |
| --- | --- |
| `DreamSampler` | Shift-logits sampler registered for `model_name="dream"`. |

## diffulex.sampler.fast_dllm_v2

This module registers the Fast-dLLM-v2 sampler.

| Symbol | Purpose |
| --- | --- |
| `FastdLLMV2Sampler` | Shift-logits sampler registered for `model_name="fast_dllm_v2"`. |

## diffulex.sampler.llada

This module registers the LLaDA sampler.

| Symbol | Purpose |
| --- | --- |
| `LLaDASampler` | No-shift sampler registered for `model_name="llada"`. |

## diffulex.sampler.llada2

This module contains the most feature-rich sampler path: accepted-token helpers,
edit sampling, DMax token merging, and factory registration for multiple LLaDA2
model names.

| Symbol | Purpose |
| --- | --- |
| `LLaDA2Sampler` | Base LLaDA2 sampler with accepted-token handling. |
| `LLaDA2dot1Sampler` | Edit-sampling variant. |
| `LLaDA2DMaxSampler` | DMax/token-merge sampler variant. |
| `build_llada2_sampler` | Factory registered for LLaDA2 model names. |

Use this module as the reference for edit-sampling and token-merge sampler
behavior.

## diffulex.sampler.sdar

This module registers the SDAR sampler for dense and MoE SDAR variants.

| Symbol | Purpose |
| --- | --- |
| `SDARSampler` | No-shift sampler registered for `sdar` and `sdar_moe`. |
