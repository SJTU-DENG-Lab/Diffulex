# diffulex.layer

`diffulex.layer` contains reusable neural-network layers and backend adapters
used by model implementations. The package keeps tensor-parallel layout,
optional LoRA handling, rotary embeddings, activation fusion, and vLLM-backed
fallbacks outside individual model files.

| Module | Role |
| --- | --- |
| `diffulex.layer.activation` | Fused gated activations with native and optional vLLM-backed paths. |
| `diffulex.layer.embed_head` | Tensor-parallel vocabulary embeddings and LM heads. |
| `diffulex.layer.layernorm` | RMSNorm and fused add-RMSNorm wrappers. |
| `diffulex.layer.linear` | Replicated, column-parallel, row-parallel, QKV, and merged linear layers with LoRA support. |
| `diffulex.layer.rotary_embedding` | Rotary embedding construction and application helpers. |
| `diffulex.layer.vllm_backend` | Runtime toggles and lazy accessors for optional vLLM layer implementations. |

## diffulex.layer.activation

This module provides gated activation blocks used by MLP implementations. It
prefers vLLM fused operators when enabled and available, then falls back to
native PyTorch implementations.

| Symbol | Purpose |
| --- | --- |
| `SiluAndMul` | SiLU-gated activation block for SwiGLU-style MLPs. |
| `GeluAndMul` | GELU-tanh gated activation block. |

Use these modules in model code instead of open-coding chunk/split activation
logic.

## diffulex.layer.embed_head

This module handles vocabulary sharding for embeddings and output projection.
It gathers or reduces tensor-parallel outputs as needed so model code can share
the same layer abstractions across single-GPU and tensor-parallel execution.

| Symbol | Purpose |
| --- | --- |
| `VocabParallelEmbedding` | Sharded embedding table with tensor-parallel rank handling. |
| `ParallelLMHead` | Output head built on the same vocabulary-parallel layout. |

Use these layers when model vocab weights need to be partitioned across tensor
parallel ranks.

## diffulex.layer.layernorm

This module provides RMSNorm implementations with optional fused vLLM paths.
The wrapper keeps model code independent of the selected backend.

| Symbol | Purpose |
| --- | --- |
| `RMSNorm` | RMSNorm module with optional fused add+norm path. |

Use `RMSNorm` in model implementations when the checkpoint architecture expects
RMS normalization.

## diffulex.layer.linear

This module contains the common linear-layer variants used by model families.
It combines tensor-parallel splitting/gathering with optional LoRA weight
loading hooks.

| Symbol | Purpose |
| --- | --- |
| `LoRAMixin` | Adapter-loading behavior shared by linear variants. |
| `LinearBase` | Common base class for Diffulex linear layers. |
| `ReplicatedLinear` | Non-sharded linear layer. |
| `ColumnParallelLinear` | Column-sharded tensor-parallel linear layer. |
| `MergedColumnParallelLinear` | Column-parallel layer for merged projections. |
| `QKVParallelLinear` | Specialized QKV projection layer. |
| `RowParallelLinear` | Row-sharded tensor-parallel linear layer. |

Choose the layer variant that matches the checkpoint's weight layout and the
model's tensor-parallel split.

## diffulex.layer.rotary_embedding

This module builds and applies rotary position embeddings. It includes standard
rotary embeddings, partial rotary embeddings, Gemma-style proportional rotary
scaling, and adapters to vLLM rotary implementations.

| Symbol | Purpose |
| --- | --- |
| `RotaryEmbedding` | Standard rotary embedding module. |
| `PartialRotaryEmbedding` | Rotary embedding for models that rotate only part of the head dimension. |
| `Gemma4ProportionalRotaryEmbedding` | Gemma-style proportional rotary scaling. |
| `VllmRotaryEmbeddingAdapter` | Adapter around vLLM rotary implementations. |
| `get_rope` | Cached rotary embedding factory. |
| `get_gemma4_proportional_rope` | Cached Gemma proportional rotary factory. |

Use the factory helpers instead of constructing rotary modules manually when
model code should share cache behavior.

## diffulex.layer.vllm_backend

This module owns the runtime switch for optional vLLM-backed common layers. It
keeps imports lazy so environments without the relevant vLLM symbols can still
import Diffulex.

| Symbol | Purpose |
| --- | --- |
| `set_vllm_layers_enabled` | Enables or disables vLLM-backed layer paths. |
| `is_vllm_layers_enabled` | Reports whether vLLM-backed paths are active. |
| `clear_vllm_layer_caches` | Clears cached backend lookups. |
| `get_vllm_silu_and_mul_cls` / `get_vllm_gelu_and_mul_cls` | Lazy activation backend accessors. |
| `get_vllm_rmsnorm_cls` | Lazy RMSNorm backend accessor. |
| `get_vllm_rope_fn` | Lazy rotary backend accessor. |

Use these helpers inside layer wrappers, not directly from model code.
