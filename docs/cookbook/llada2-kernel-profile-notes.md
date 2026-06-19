# LLaDA2 Kernel Profile Notes

These notes summarize the LLaDA2-mini TP4 profiling run used while tuning the
current vLLM-backed kernels.

Run context:

- Profile root: `benchmark_results/profile_diffulex_llada2_gsm8k_limit2_tp4_modular`
- Model: `/data1/ckpts/inclusionAI/LLaDA2.0-mini`
- TP/DP/EP: `4/1/1`
- Attention: `triton_grouped`
- MoE: `vllm_modular`
- CUDA graph: prefill and full static decode enabled
- Hardware observed during the run: RTX 3090, PCIe topology without NVLink

LLaDA2-mini structure:

- 20 decoder layers
- hidden size 2048
- 16 query heads, 4 KV heads, head dim 128
- 256 global experts, TP4 gives 64 local experts per rank
- top-8 routed MoE
- layer 0 is dense, layers 1-19 are MoE

## Stage Breakdown

The table below uses GPU annotation ranges from the trace. These ranges are
useful for pipeline timing, but they should not be interpreted as exact kernel
ownership. CUDA graph replay and stream ordering can make a Python
`record_function` range inherit work or waiting that was launched elsewhere.

Percentages are per-rank fractions of visible stage time in this profile.

| rank | stage | total ms | count | avg ms | ratio |
|---:|---|---:|---:|---:|---:|
| 0 | full_static_decode | 2059.9 | 94 | 21.91 | 57.4% |
| 0 | full_static_prefill | 1090.4 | 6 | 181.73 | 30.4% |
| 0 | sampler | 258.0 | 96 | 2.69 | 7.2% |
| 0 | prepare_chunked_prefill | 91.6 | 96 | 0.95 | 2.6% |
| 0 | nccl:gather | 83.5 | 96 | 0.87 | 2.3% |
| 1 | full_static_decode | 1686.5 | 94 | 17.94 | 55.4% |
| 1 | full_static_prefill | 1003.4 | 6 | 167.24 | 32.9% |
| 1 | prepare_chunked_prefill | 277.7 | 96 | 2.89 | 9.1% |
| 1 | nccl:gather | 58.9 | 96 | 0.61 | 1.9% |
| 2 | full_static_decode | 1583.4 | 94 | 16.85 | 52.7% |
| 2 | full_static_prefill | 1003.6 | 6 | 167.27 | 33.4% |
| 2 | prepare_chunked_prefill | 351.2 | 96 | 3.66 | 11.7% |
| 2 | nccl:gather | 43.5 | 96 | 0.45 | 1.5% |
| 3 | full_static_decode | 1672.3 | 94 | 17.79 | 55.1% |
| 3 | full_static_prefill | 1008.1 | 6 | 168.02 | 33.2% |
| 3 | prepare_chunked_prefill | 271.3 | 96 | 2.83 | 8.9% |
| 3 | nccl:gather | 61.9 | 96 | 0.65 | 2.0% |

Notes:

- `prepare_chunked_prefill` is not a hundreds-of-ms single-step cost. It is
  roughly 1-4 ms per step in this profile.
- `sampler` only does real sampling work on rank 0 for this model path. Its
  median is lower than its average because there is a large outlier.
- `model_forward` as an outer annotation is not the true model cost when full
  static CUDA graph replay is enabled; use `full_static_prefill` and
  `full_static_decode` for the graph replay stage.

## Kernel Breakdown

The table below groups selected actual CUDA kernels and operators. Percentages
are fractions of selected kernel time, not end-to-end wall time.

| rank | kernel group | total ms | ratio |
|---:|---|---:|---:|
| 0 | NCCL bf16 all-reduce | 1647.2 | 71.4% |
| 0 | MoE fused kernel | 369.6 | 16.0% |
| 0 | NCCL gather/sendrecv | 166.5 | 7.2% |
| 0 | attention grouped | 39.1 | 1.7% |
| 0 | dense/small GEMM | 24.3 | 1.1% |
| 0 | router fused topk | 20.5 | 0.9% |
| 0 | RMSNorm + SiluAndMul | 17.1 | 0.7% |
| 1 | NCCL bf16 all-reduce | 1206.3 | 66.0% |
| 1 | MoE fused kernel | 388.7 | 21.3% |
| 1 | NCCL gather/sendrecv | 117.9 | 6.5% |
| 1 | attention grouped | 38.9 | 2.1% |
| 2 | NCCL bf16 all-reduce | 784.9 | 57.1% |
| 2 | MoE fused kernel | 385.2 | 28.0% |
| 2 | NCCL gather/sendrecv | 86.9 | 6.3% |
| 2 | attention grouped | 39.3 | 2.9% |
| 3 | NCCL bf16 all-reduce | 1212.0 | 66.3% |
| 3 | MoE fused kernel | 377.0 | 20.6% |
| 3 | NCCL gather/sendrecv | 123.9 | 6.8% |
| 3 | attention grouped | 38.9 | 2.1% |

Interpretation:

- Attention is not the dominant bottleneck in this workload after switching to
  `triton_grouped`. It is around 2% of selected kernel time.
- TP all-reduce dominates on this PCIe TP4 setup. LLaDA2-mini performs many
  small all-reduces per forward: embedding, attention output projection, routed
  MoE output, shared expert output, and dense MLP output.
- MoE remains the largest compute kernel after communication. The
  `vllm_modular` path is the right baseline for current comparisons.
- Small layers such as vLLM RMSNorm and SiluAndMul are already fast in absolute
  terms. Their remaining impact is mostly kernel fragmentation and launch
  count, not single-kernel latency.

## Optimization Order

Current priority order from this profile:

1. Reduce or hide TP all-reduce. Compare TP2 on `0,1` or `2,3` against TP4 on
   `0,1,2,3`; this separates topology cost from compute cost.
2. Keep MoE on vLLM modular and verify the global expert-id plus expert-map
   path. Avoid reverting to local-id masking for the modular backend.
3. Tensorize sampler decisions. The current sampler has CPU sync points such as
   `.item()`, `.tolist()`, and `nonzero(...).tolist()`.
4. Reduce prepare metadata copies and Python list construction. This is smaller
   than communication, but it is now visible at 1-4 ms per step.
5. Treat attention as a correctness-sensitive maintained kernel, not the next
   main performance target for this workload.

