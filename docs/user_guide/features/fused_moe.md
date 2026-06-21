# Fused MoE

Fused MoE paths accelerate routing and expert execution for supported
Mixture-of-Experts models. Diffulex exposes MoE dispatcher and GEMM controls,
but the current config path still restricts all-to-all MoE execution.

## Dispatcher Backend

`moe_dispatcher_backend` can be set to `standard`, `naive`, or `deepep`. The
default is `standard`, and that is the only value currently accepted by config
validation.

Current validation requires `moe_dispatcher_backend == "standard"`.

## GEMM Implementation

`moe_gemm_impl` can be `triton`, `vllm`, `vllm_modular`, or `naive`. The default
is `triton`.

Use `naive` for debugging and optimized implementations for performance checks.

## DeepEP Controls

`deepep_mode` accepts `normal`, `low_latency`, or `auto`.

`deepep_num_max_dispatch_tokens_per_rank` must be positive and defaults to
`256`.

## Current Limitation

Although expert parallel values are present in config, current validation
requires:

| Key | Required value |
| --- | --- |
| `expert_parallel_size` | `1` |
| `moe_dispatcher_backend` | `standard` |

Treat other MoE A2A settings as experimental until this restriction is removed.
