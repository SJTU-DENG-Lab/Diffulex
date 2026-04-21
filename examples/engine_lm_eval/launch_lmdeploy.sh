#!/bin/bash

lmdeploy serve api_server /root/data/ckpts/JetLM/SDAR-8B-Chat-b32 \
  --backend pytorch \
  --tp 2 \
  --max-batch-size 16 \
  --distributed-executor-backend mp \
  --max-concurrent-requests 16 \
  --cache-max-entry-count 0.7 \
  --dllm-block-length 32 \
  --dllm-unmasking-strategy low_confidence_dynamic \
  --dllm-denoising-steps 32 \
  --dllm-confidence-threshold 0.95 \
  --server-name 0.0.0.0 \
  --server-port 29998