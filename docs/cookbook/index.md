# Cookbook

This section provides model-oriented recipes and entry-point references for Diffulex.

Use the model pages when you want recommended serving and benchmark configurations for a specific checkpoint. Use the entry-point pages when you want the generic command structure.

## D2F Recipes

These recipes use `d2f` decoding and include high-concurrency and low-concurrency speed presets where applicable.

:::{toctree}
:maxdepth: 1
models/llada_instruct
models/dream_base
:::

## MultiBD Recipes

These recipes use `multi_bd` decoding and focus on balanced serving and throughput-oriented presets.

:::{toctree}
:maxdepth: 1
models/fast_dllm_v2
models/sdar_chat
:::

## Entry Points

:::{toctree}
:maxdepth: 1
benchmark
server
streamlit
:::
