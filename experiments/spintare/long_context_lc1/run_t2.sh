#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
# Running-mean baseline (T2) in the LC1 harness at 32K: the twelve LC1 models, bfloat16 weights, one
# GPU (RTX 5090 D in the reported run), one model at a time. Feeds the "Run" columns of the 32K
# output-KL tables. Aggregate afterwards with
#   python t2_runmean.py agg $KVREL_WORK/t2/res --logs $KVREL_WORK/t2/logs
#          --lc1-res $KVREL_WORK/lc1/res --lc1-dir <this directory>
# usage: bash run_t2.sh [gpu]
. "$(cd "$(dirname "$0")/../../.." && pwd)/scripts/env.sh"
set -u
HERE=$(cd "$(dirname "$0")" && pwd); GPU=${1:-0}
W=$KVREL_WORK/t2; mkdir -p "$W/res" "$W/logs"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU
while read -r short dir order; do
  out=$W/res/t2_$short.json
  [ -s "$out" ] && continue
  [ -d "$KVREL_MODELS/$dir" ] || { echo "$short: no $KVREL_MODELS/$dir, skipped"; continue; }
  timeout 14400 "$KVREL_PYTHON" "$HERE/t2_runmean.py" real "$out" "$KVREL_MODELS/$dir" "$short" \
    --lc1-dir "$HERE" --order "$order" --data "$KVREL_DATA" --L 32768 --dtype bfloat16 \
    > "$W/logs/t2_$short.log" 2>&1
  echo "$short rc=$?"
done <<'EOF'
q3_0p6b Qwen3-0.6B hf
q25_1p5b Qwen2.5-1.5B-Instruct hf
q25_7b Qwen2.5-7B-Instruct hf
r1q_7b DeepSeek-R1-Distill-Qwen-7B hf
q3_8b Qwen3-8B hf
llama3b Llama-3.2-3B-Instruct gguf
llama31_8b Meta-Llama-3.1-8B-Instruct gguf
mistral7b Mistral-7B-Instruct-v0.3 gguf
granite8b granite-3.1-8b-instruct gguf
olmo7b OLMo-2-1124-7B-Instruct hf
yi6b Yi-1.5-6B-Chat gguf
ds7b deepseek-llm-7b-chat_tokfix gguf
EOF
