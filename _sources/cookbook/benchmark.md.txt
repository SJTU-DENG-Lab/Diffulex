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

## Supported models

### D2F-LLaDA

```bash
python -m diffulex_bench.main \
  --config diffulex_bench/configs/llada_instruct_gsm8k.yml \
  --model-path /YOUR-CKPT-PATH/GSAI-ML/LLaDA-8B-Instruct \
  --tokenizer-path /YOUR-CKPT-PATH/GSAI-ML/LLaDA-8B-Instruct \
  --model-name llada \
  --decoding-strategy d2f \
  --use-lora \
  --lora-path /YOUR-CKPT-PATH/SJTU-DENG-Lab/D2F_LLaDA_Instruct_8B_Lora \
  --tensor-parallel-size 2 \
  --data-parallel-size 1 \
  --dataset gsm8k_diffulex \
  --dataset-limit 100 \
  --temperature 0.0 \
  --max-tokens 256
```

### D2F-Dream

```bash
python -m diffulex_bench.main \
  --config diffulex_bench/configs/dream_base_gsm8k.yml \
  --model-path /YOUR-CKPT-PATH/Dream-org/Dream-v0-Base-7B \
  --tokenizer-path /YOUR-CKPT-PATH/Dream-org/Dream-v0-Base-7B \
  --model-name dream \
  --decoding-strategy d2f \
  --use-lora \
  --lora-path /YOUR-CKPT-PATH/SJTU-DENG-Lab/D2F_Dream_Base_7B_Lora \
  --tensor-parallel-size 2 \
  --data-parallel-size 1 \
  --dataset gsm8k_diffulex \
  --dataset-limit 100 \
  --temperature 0.0 \
  --max-tokens 256
```

### Fast-dLLM-v2

```bash
python -m diffulex_bench.main \
  --config diffulex_bench/configs/fast_dllm_v2_gsm8k.yml \
  --model-path /YOUR-CKPT-PATH/Efficient-Large-Model/Fast_dLLM_v2_7B \
  --tokenizer-path /YOUR-CKPT-PATH/Efficient-Large-Model/Fast_dLLM_v2_7B \
  --model-name fast_dllm_v2 \
  --decoding-strategy multi_bd \
  --tensor-parallel-size 2 \
  --data-parallel-size 1 \
  --dataset gsm8k_diffulex \
  --dataset-limit 100 \
  --temperature 0.0 \
  --max-tokens 256
```

### SDAR

```bash
python -m diffulex_bench.main \
  --config diffulex_bench/configs/sdar_chat_gsm8k.yml \
  --model-path /YOUR-CKPT-PATH/JetLM/SDAR-1.7B-Chat-b32 \
  --model-name sdar \
  --decoding-strategy multi_bd \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --dataset gsm8k_diffulex \
  --temperature 0.0 \
  --max-tokens 256
```

### SDAR-MoE

Use the same benchmark entry point and a matching `sdar_moe` config. The repository already treats `sdar_moe` as a supported model family; keep the same benchmark structure as SDAR and set the model path to your SDAR-MoE checkpoint.
