#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# Exemption gate in simulation (C23): whole-model next-token KL with centred Q4_0 keys in every
# layer; ten models x eight 4096-token windows, float32 weights, exact attention, one GPU (RTX PRO
# 5000 in the registered run). Feeds the exemption-gate table. The quantizer is checked
# bit for bit against the ggml library of a llama.cpp build (KVREL_GGML_LIB, default
# $KVREL_LLAMA_CPP/build/bin/libggml-base.so). Aggregate: python c23_kl.py agg $KVREL_WORK/c23/res
# A model starts only while the card is free (< 2000 MiB in use); otherwise the queue stops, so a
# shared card is never taken from another user. One queue at a time (lock file).
# usage: bash run_c23.sh [gpu]
. "$(cd "$(dirname "$0")/../.." && pwd)/scripts/env.sh"
set -u
HERE=$(cd "$(dirname "$0")" && pwd); GPU=${1:-0}
W=$KVREL_WORK/c23; mkdir -p "$W/res" "$W/logs"
LIB=${KVREL_GGML_LIB:-$KVREL_LLAMA_CPP/build/bin/libggml-base.so}
exec 9> "$W/queue.lock"; flock -n 9 || { echo "another C23 queue holds $W/queue.lock"; exit 75; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for job in Qwen2.5-1.5B-Instruct:q25_1p5b Llama-3.2-3B-Instruct:llama3b \
           Meta-Llama-3.1-8B-Instruct:llama31_8b Mistral-7B-Instruct-v0.3:mistral7b Qwen3-8B:q3_8b \
           Qwen2.5-7B-Instruct:q25_7b granite-3.1-8b-instruct:granite8b Yi-1.5-6B-Chat:yi6b \
           OLMo-2-1124-7B-Instruct:olmo7b deepseek-llm-7b-chat_tokfix:ds7b; do
  IFS=: read -r m s <<< "$job"
  [ -s "$W/res/c23_$s.json" ] && continue
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  if [ -z "$used" ] || [ "$used" -gt 2000 ]; then
    echo "GPU $GPU busy (${used:-?} MiB), stop before $s"; exit 4
  fi
  timeout 5400 "$KVREL_PYTHON" "$HERE/c23_kl.py" real "$W/res/c23_$s.json" "$KVREL_MODELS/$m" "$s" \
    --lib "$LIB" > "$W/logs/c23_$s.log" 2>&1 < /dev/null
  echo "$s rc=$?"
done
