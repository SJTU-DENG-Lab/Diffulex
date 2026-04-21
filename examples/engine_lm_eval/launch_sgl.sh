#!/bin/bash
MODEL=/root/data/ckpts/inclusionAI/LLaDA2.0-mini
# MODEL=/root/data/ckpts/JetLM/SDAR-8B-Chat-b32

python -m sglang.launch_server \
  --model-path $MODEL \
  --dllm-algorithm LowConfidence \
  --dllm-algorithm-config configs/low_confidence.yml \
  --trust-remote-code \
  --tp-size 2 \
  --cuda-graph-max-bs 16 \
  --max-running-requests 16 \
  --mem-fraction-static 0.7 \
  --host 0.0.0.0 \
  --port 29998