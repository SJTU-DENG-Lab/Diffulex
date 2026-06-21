# Diffulex

## Run Experiments

Experiment configs live in:

```bash
diffulex_bench/configs/experiment/
```

Core hyperparameters are mirrored in those config files. Common values for all rows:

```text
max_model_len=4096, max_new_tokens=4096, max_nfe=1024
```

| Configuration | Variant | Task | Buffer | Block | tau_add | tau_semi | tau_stable | tau_M2T | tau_T2T |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaDA2-Mini-DMax | SingleBD Native | Math | 1 | 32 | - | - | - | 0.50 | - |
| LLaDA2-Mini-DMax | SingleBD Native | Code | 1 | 32 | - | - | - | 0.65 | - |
| LLaDA2-Mini-DMax | MultiBD training-free | Math | 2 | 32 | 0.10 | 0.90 | 0.50 | 0.50 | - |
| LLaDA2-Mini-DMax | MultiBD training-free | Code | 2 | 32 | 0.90 | 0.90 | 0.50 | 0.65 | - |
| MBD-LLaDA2-Mini-DMax | MBD | Math | 2 | 32 | 0.10 | 0.90 | 0.50 | 0.50 | - |
| MBD-LLaDA2-Mini-DMax | MBD | Code | 2 | 32 | 0.90 | 0.90 | 0.50 | 0.65 | - |
| LLaDA2-Mini | SingleBD Native | Math | 1 | 32 | - | - | - | 0.95 | - |
| LLaDA2-Mini | SingleBD Native | Code | 1 | 32 | - | - | - | 0.95 | - |
| LLaDA2-Mini | MultiBD training-free | Math | 2 | 32 | 0.10 | 0.90 | - | 0.95 | - |
| LLaDA2-Mini | MultiBD training-free | Code | 2 | 32 | 0.90 | 0.90 | - | 0.95 | - |
| MBD-LLaDA2-Mini | MBD | Math | 2 | 32 | 0.10 | 0.90 | - | 0.95 | - |
| MBD-LLaDA2-Mini | MBD | Code | 2 | 32 | 0.90 | 0.90 | - | 0.95 | - |
| SDAR-8B-Chat-b32 | SingleBD Native | Math | 1 | 32 | - | - | - | 0.95 | - |
| SDAR-8B-Chat-b32 | SingleBD Native | Code | 1 | 32 | - | - | - | 0.95 | - |
| SDAR-8B-Chat-b32 | MultiBD training-free | Math | 4 | 32 | 0.10 | 0.90 | - | 0.95 | - |
| SDAR-8B-Chat-b32 | MultiBD training-free | Code | 4 | 32 | 0.90 | 0.90 | - | 0.95 | - |
| MBD-SDAR-8B-Chat-b32 | MBD | Math | 4 | 32 | 0.10 | 0.90 | - | 0.95 | - |
| MBD-SDAR-8B-Chat-b32 | MBD | Code | 4 | 32 | 0.90 | 0.90 | - | 0.95 | - |
| SDAR-8B-Chat-b4 | SingleBD Native | Math | 1 | 4 | - | - | - | 0.95 | - |
| SDAR-8B-Chat-b4 | SingleBD Native | Code | 1 | 4 | - | - | - | 0.95 | - |
| SDAR-8B-Chat-b4 | MultiBD training-free | Math | 4 | 4 | 0.10 | 0.25 | - | 0.95 | - |
| SDAR-8B-Chat-b4 | MultiBD training-free | Code | 4 | 4 | 0.75 | 0.75 | - | 0.95 | - |
| MBD-SDAR-8B-Chat-b4 | MBD | Math | 4 | 4 | 0.10 | 0.25 | - | 0.95 | - |
| MBD-SDAR-8B-Chat-b4 | MBD | Code | 4 | 4 | 0.75 | 0.75 | - | 0.95 | - |
| LLaDA2-Mini-CAP | SingleBD Native | Math | 1 | 32 | - | - | - | 0.95 | - |
| LLaDA2-Mini-CAP | SingleBD Native | Code | 1 | 32 | - | - | - | 0.95 | - |
| LLaDA2-Mini-CAP | MultiBD training-free | Math | 2 | 32 | 0.10 | 0.90 | - | 0.95 | - |
| LLaDA2-Mini-CAP | MultiBD training-free | Code | 2 | 32 | 0.90 | 0.90 | - | 0.95 | - |
| LLaDA2.1-Mini | SingleBD Native | Math | 1 | 32 | - | - | - | 0.70 | 0.50 |
| LLaDA2.1-Mini | SingleBD Native | Code | 1 | 32 | - | - | - | 0.70 | 0.50 |
| LLaDA2.1-Mini | MultiBD training-free | Math | 2 | 32 | 0.10 | 0.90 | - | 0.70 | 0.50 |
| LLaDA2.1-Mini | MultiBD training-free | Code | 2 | 32 | 0.90 | 0.90 | - | 0.70 | 0.50 |

The only experiment entrypoint is:

```bash
./script/run_batch_experiments.sh
```

Preview the run plan without launching models:

```bash
DRY_RUN=1 ./script/run_batch_experiments.sh
```

Run all experiment configs:

```bash
./script/run_batch_experiments.sh
```

Run selected config files:

```bash
CONFIG_FILES=llada2_mini.yml ./script/run_batch_experiments.sh
CONFIG_FILES=llada2_mini.yml,sdar_8b_chat_b32.yml ./script/run_batch_experiments.sh
```

Filter selected experiments by name/group/task/model:

```bash
FILTER=multibd_math ./script/run_batch_experiments.sh
FILTER=llada2_mini DATASET_LIMIT=10 ./script/run_batch_experiments.sh
```

Override capacity or checkpoint paths:

```bash
MAX_NUM_REQS=256 ./script/run_batch_experiments.sh
LLADA2_MINI_MODEL=/data/ckpts/inclusionAI/LLaDA2.0-mini ./script/run_batch_experiments.sh
SDAR_B32_MODEL=/path/to/SDAR-8B-Chat-b32 ./script/run_batch_experiments.sh
```

If some configured checkpoints are unavailable and you want to run only the available ones:

```bash
SKIP_MISSING_MODELS=1 ./script/run_batch_experiments.sh
```

Outputs are written to:

```bash
benchmark_results/experiment/<run_id>/
logs/experiment/<run_id>/
```

Each run also writes resolved per-experiment benchmark YAMLs under:

```bash
benchmark_results/experiment/<run_id>/resolved_configs/
```
