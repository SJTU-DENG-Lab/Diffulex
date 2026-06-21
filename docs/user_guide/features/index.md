# Features

Diffulex features are grouped by the part of inference they affect: decoding
state, parallel topology, attention and cache behavior, runtime capture, adapter
loading, and MoE execution.

:::{toctree}
:maxdepth: 1

multi_block_decoding
decoding_strategies
parallelism
optimized_attention
prefix_caching
cuda_graph
lora_adapters
fused_moe
:::

Use the feature pages as tuning references. For a first run, keep defaults and
change only the model path, model name, and decoding strategy. Once generation
works, tune one feature area at a time.
