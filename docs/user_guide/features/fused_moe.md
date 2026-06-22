# Fused MoE

Fused MoE paths accelerate routing and expert execution for supported
Mixture-of-Experts models. The public tuning surface focuses on the GEMM
implementation used inside the validated single-node MoE path.

## GEMM Implementation

`moe_gemm_impl` can be `triton`, `vllm`, `vllm_modular`, or `naive`. The default
is `triton`.

Use `naive` for debugging and optimized implementations for performance checks.
