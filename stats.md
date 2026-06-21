# Single A100 GSM8K Stats

Date: 2026-06-21 UTC

Hardware: 1x NVIDIA A100-SXM4-80GB

All runs use the full GSM8K test split with 1319 samples. We recommend using aggregate TPS as the primary throughput number: it is token/time weighted and reflects full-run throughput. Average TPS is still useful for typical per-request behavior, but it can overweight short samples.

The table below keeps only strict single-sample / single-active-request runs. vLLM uses the local patched server-side DLLM usage counters with CUDA-event step timing; its raw OpenAI client wall-clock TPS is shown separately.

| Benchmark | Engine / config | Samples | Visible tokens | Server tokens | Wall time (s) | E2E time (s) | Agg e2e TPS | Avg e2e TPS | Agg decode TPS | Avg decode TPS | Client wall e2e TPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaDA2-mini | Diffulex, max reqs 1 | 1319 | 415897 | - | 2318.98 | 2296.20 | 181.12 | 204.02 | 193.66 | 224.69 | - |
| LLaDA2-mini | SGLang, max running requests 1 | 1319 | 509575 | 512140 | 2871.16 | 2871.16 | 177.48 | 217.07 | 194.78 | 245.53 | - |
| DiffusionGemma | Diffulex, max reqs 1 | 1319 | 376617 | - | 807.00 | 803.42 | 468.77 | 481.35 | 797.48 | 825.48 | - |
| DiffusionGemma | vLLM, max seqs 1 | 1319 | 384177 | 385546 | 680.63 | 628.09 | 611.66 | 641.01 | 658.79 | 696.40 | 564.44 |
