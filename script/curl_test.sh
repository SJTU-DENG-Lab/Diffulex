#!/bin/bash
curl -N --no-buffer http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"sdar","messages":[{"role":"user","content":"What do you know about LLMs."}],"stream":true,"stream_mode":"denoise","max_tokens":64,"temperature":0.0,"max_nfe":100}'