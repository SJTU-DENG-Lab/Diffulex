# diffulex.model

`diffulex.model` contains model-family implementations and the model registry
used by the engine. Importing the package loads built-in model modules so their
`AutoModelForDiffusionLM.register(...)` decorators run.

| Module | Role |
| --- | --- |
| `diffulex.model.auto_model` | Registry and factory for model implementations. |
| `diffulex.model.config` | Hugging Face `PretrainedConfig` adapters for supported model families. |
| `diffulex.model.diffucoder` | Reserved model module for DiffuCoder-family work. |
| `diffulex.model.diffusion_gemma` | DiffusionGemma model implementation. |
| `diffulex.model.dream` | Dream model implementation. |
| `diffulex.model.fast_dllm_v2` | Fast-dLLM-v2 model implementation. |
| `diffulex.model.llada` | LLaDA model implementation. |
| `diffulex.model.llada2` | LLaDA2 dense and MoE model implementation. |
| `diffulex.model.sdar` | SDAR dense model implementation. |
| `diffulex.model.sdar_moe` | SDAR-MoE model implementation. |

## diffulex.model.auto_model

This module maps `Config.model_name` values to model factories. It also supports
model factories that need either Hugging Face config only or the full Diffulex
config.

| Symbol | Purpose |
| --- | --- |
| `AutoModelForDiffusionLM` | Registry used by model runners to construct the selected model family. |
| `AutoModelLM` | Compatibility alias for `AutoModelForDiffusionLM`. |

New model families should register here through the decorator exposed by
`AutoModelForDiffusionLM.register(...)`.

## diffulex.model.config

This package contains model-family configuration adapters that integrate custom
checkpoint configs with Hugging Face `AutoConfig`.

| Included configs | Purpose |
| --- | --- |
| DiffusionGemma config | Registers DiffusionGemma text and wrapper config classes. |
| Dream config | Registers Dream checkpoint config. |
| Fast-dLLM-v2 config | Registers Fast-dLLM-v2 checkpoint config. |
| LLaDA config | Provides LLaDA config enums and `LLaDAConfig`. |
| SDAR config | Registers SDAR checkpoint config. |

Use this package when a checkpoint requires a custom `PretrainedConfig` before
model construction can proceed.

## diffulex.model.diffucoder

This module is currently reserved for DiffuCoder-family implementation work.
Keep user-facing examples pointed at documented supported model families until
the module contains a registered model implementation.

There are no public runtime symbols in this module yet. When DiffuCoder support
is added, this page should be updated with the registered model class, config
requirements, sampler compatibility, and checkpoint-loading notes.

## diffulex.model.diffusion_gemma

This module implements DiffusionGemma-specific layers, routing, attention, and
the registered diffusion language model wrapper.

| Symbol | Purpose |
| --- | --- |
| `DiffusionGemmaForDiffusionLM` | Registered model class for `model_name="diffusion_gemma"`. |
| `DiffusionGemmaMoE` | MoE block used by DiffusionGemma layers. |
| `DiffusionGemmaAttention` | Attention implementation wired to Diffulex attention metadata. |

DiffusionGemma uses larger block/page sizes and strategy defaults during config
normalization.

## diffulex.model.dream

This module implements the Dream-family transformer stack and registers it with
the model registry.

| Symbol | Purpose |
| --- | --- |
| `DreamForDiffusionLM` | Registered model class for `model_name="dream"`. |
| `DreamModel` | Decoder stack. |
| `DreamAttention` / `DreamMLP` / `DreamDecoderLayer` | Core layer building blocks. |

Use this module as a reference for a dense model family with custom attention
and MLP layers.

## diffulex.model.fast_dllm_v2

This module implements the Fast-dLLM-v2 model family.

| Symbol | Purpose |
| --- | --- |
| `FastdLLMV2ForDiffusionLM` | Registered model class for `model_name="fast_dllm_v2"`. |
| `FastdLLMV2Model` | Decoder stack. |
| `FastdLLMV2Attention` / `FastdLLMV2MLP` / `FastdLLMV2DecoderLayer` | Core layer building blocks. |

Fast-dLLM-v2 is typically used with multi-block decoding.

## diffulex.model.llada

This module implements the LLaDA model family used by D2F-style paths.

| Symbol | Purpose |
| --- | --- |
| `LLaDAForDiffusionLM` | Registered model class for `model_name="llada"`. |
| `LLaDAModel` | Decoder stack. |
| `LLaDAAttention` / `LLaDAMLP` / `LLaDABlock` | Core layer building blocks. |

Use this module as the closest dense-model reference for LLaDA-like checkpoints.

## diffulex.model.llada2

This module implements LLaDA2 dense and MoE variants. It includes custom QKV
projection behavior, dense and MoE MLP construction, runtime config helpers,
and registry entries for multiple LLaDA2 model names.

| Symbol | Purpose |
| --- | --- |
| `LLaDA2ForDiffusionLM` | Registered model class for `llada2`, `llada2_moe`, `llada2_mini`, and `llada2dot1_mini`. |
| `LLaDA2Model` | Decoder stack. |
| `LLaDA2QKVParallelLinear` | QKV projection layer specific to LLaDA2. |
| `build_llada2_mlp` | Builds dense or MoE MLP blocks based on config and layer index. |
| `build_llada2_runtime_config` | Converts full Diffulex config into LLaDA2 runtime settings. |

This module is the main model-side reference for edit-sampling and MoE-capable
LLaDA2 paths.

## diffulex.model.sdar

This module implements the dense SDAR model family.

| Symbol | Purpose |
| --- | --- |
| `SDARForDiffusionLM` | Registered model class for `model_name="sdar"`. |
| `SDARModel` | Decoder stack. |
| `SDARAttention` / `SDARMLP` / `SDARDecoderLayer` | Core layer building blocks. |

SDAR is typically paired with multi-block decoding.

## diffulex.model.sdar_moe

This module extends SDAR with MoE decoder blocks and registers the SDAR-MoE
model family.

| Symbol | Purpose |
| --- | --- |
| `SDARMoEForDiffusionLM` | Registered model class for `model_name="sdar_moe"`. |
| `SDARMoEModel` | MoE decoder stack. |
| `SDARMoEDecoderLayer` | SDAR decoder layer with MoE feed-forward behavior. |

Use this module as a reference when adapting a dense family into an MoE variant.
