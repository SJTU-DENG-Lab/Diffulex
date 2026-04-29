# Benchmark

Use the benchmark entry point when you want to run evaluation workloads such as GSM8K, HumanEval, or MBPP through `diffulex_bench`.

Generic form:

```bash
python -m diffulex_bench.main \
  --config path/to/config.yml \
  --log-file path/to/run.log \
  --log-level DEBUG
```

The config file provides the engine and evaluation settings. In the repository, the common pattern is:

- `--config diffulex_bench/configs/<name>.yml`
- optional overrides like `--model-path`, `--tokenizer-path`, `--model-name`, `--decoding-strategy`, `--tensor-parallel-size`, `--data-parallel-size`, `--dataset`, `--dataset-limit`, `--temperature`, `--max-tokens`, and `--output-dir`

## Model recipes

Use the model pages for recommended benchmark configurations:

- [LLaDA-8B-Instruct (D2F)](models/llada_instruct)
- [Dream-v0-Base-7B (D2F)](models/dream_base)
- [Fast_dLLM_v2_7B (MultiBD)](models/fast_dllm_v2)
- [SDAR-1.7B-Chat-b32 (MultiBD)](models/sdar_chat)
