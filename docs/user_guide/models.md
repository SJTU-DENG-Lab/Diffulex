# Models

Diffulex model support is defined by three choices:

- `model_name`: selects the registered model implementation and sampler factory;
- `decoding_strategy`: selects request state, scheduler, KV cache manager, and
  model runner;
- `sampling_mode`: selects standard or edit-style sampler behavior.

The config validator normalizes some model/strategy combinations and rejects
known invalid combinations.

## Supported Model Families

| Model family | `model_name` values | Typical strategy | Notes |
| --- | --- | --- | --- |
| Dream / D2F-Dream | `dream` | `d2f` | D2F-style full-prefix block decoding. |
| DiffuCoder / D2F-DiffuCoder | `diffucoder` | `d2f` | Uses shifted sampler behavior. |
| Dream reasoner | `dream_reasoner` | `multi_bd` | Block-causal MultiBD path. |
| Stable-DiffCoder | `stable_diffcoder` | `multi_bd` | Block-causal MultiBD path. |
| LLaDA / D2F-LLaDA | `llada` | `d2f` | Use D2F LoRA-style configs when applicable. |
| Fast-dLLM-v2 | `fast_dllm_v2` | `multi_bd` | Multi-Block Diffusion path. |
| SDAR | `sdar` | `multi_bd` | Dense SDAR path. |
| SDAR-MoE | `sdar_moe` | `multi_bd` | MoE path; keep expert parallel at `1` unless extending the runtime. |
| LLaDA2 family | `llada2`, `llada2_mini`, `llada2_moe`, `llada2dot1_mini` | `multi_bd` or `dmax` | LLaDA2-mini GSM8K is the primary maintained benchmark path. |
| DiffusionGemma | `diffusion_gemma` | `diffusion_gemma` | Native 256-token canvas/block decoder. |

The original full-attention inference implementations from some upstream dLLM
projects are not the target runtime. Diffulex adds support model by model
through block-wise adapters, samplers, and strategy registrations.

## Strategy Compatibility

| Strategy | Use it for | Important behavior |
| --- | --- | --- |
| `d2f` | D2F-style LLaDA, Dream, and DiffuCoder paths | Forces full-prefix multi-block behavior and disables prefix caching. |
| `multi_bd` | LLaDA2, SDAR, Fast-dLLM-v2, stable DiffuCoder/Dream reasoner paths | Implements Multi-Block Diffusion with block-causal visibility and prefix caching. |
| `dmax` | Supported LLaDA2 edit-sampling experiments | Requires `sampling_mode="edit"`. |
| `diffusion_gemma` | DiffusionGemma | Uses DiffusionGemma request, sampler, block/page size, and canvas defaults. |

## Sampling Modes

| Sampling mode | Use it for |
| --- | --- |
| `naive` | Standard confidence-based diffusion sampling. This is the default for most supported models. |
| `edit` | LLaDA2-family edit/remask sampling. Required by DMax-style decoding. |

## Model Path Requirements

Model and tokenizer paths should point to local directories. During startup,
Diffulex loads tokenizer metadata, Hugging Face config, and then the registered
Diffulex model implementation.

Before a full benchmark, verify:

| Requirement | Check |
| --- | --- |
| Checkpoint path | The model directory exists and contains the expected config and weights. |
| Tokenizer | The tokenizer loads from `tokenizer_path` or the model path. |
| `model_name` | The model name is listed in the table above and registered under `diffulex/model/`. |
| Strategy | The strategy is compatible with the model family. |
| Mask token | `mask_token_id` matches the tokenizer for the selected model. |
| Page/block size | `block_size <= page_size`; DiffusionGemma uses `256/256`. |

## Maintained Benchmark Configs

Common starting points live under `diffulex_bench/configs/`:

| Config | Purpose |
| --- | --- |
| `llada2_mini_gsm8k.yml` | LLaDA2-mini, `multi_bd`, GSM8K. |
| `llada2_mini_dmax_gsm8k.yml` | LLaDA2-mini DMax/edit sampling. |
| `diffusion_gemma_gsm8k.yml` | DiffusionGemma native Diffulex benchmark. |
| `diffucoder_instruct_gsm8k.yml` | DiffuCoder D2F-style benchmark. |
| `dream_base_gsm8k.yml` | Dream D2F-style benchmark. |
| `fast_dllm_v2_gsm8k.yml` | Fast-dLLM-v2 multi-block benchmark. |
| `sdar_chat_gsm8k.yml` | SDAR dense benchmark. |
| `sdar_moe_chat_gsm8k.yml` | SDAR-MoE benchmark. |

Start with `--dataset-limit` before running a full dataset.
