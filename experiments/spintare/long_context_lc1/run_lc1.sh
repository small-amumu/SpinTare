#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# LC1 (REGISTRATION_LC1.md + A1): static mean-centring of post-RoPE keys at 32K context, twelve models with bfloat16
# weights, then three small models again with float32 weights (control for the rotary tables, not aggregated).
# One RTX 5090 D; the card lock is held for the whole queue, so the queue starts when the timing
# run (OV1) has released it. One model at a time; a failed model is logged and the queue goes on.
# usage: setsid nohup bash run_lc1.sh > /dev/null 2>&1 < /dev/null &
. "$(cd "$(dirname "$0")/../.." && pwd)/scripts/env.sh"  # KVREL_* paths (see docs/setup.md)
set -u
W=${KVREL_WORK}/lc1; M=${KVREL_MODELS}; DATA=${KVREL_DATA}; GPU=0
mkdir -p "$W/res" "$W/logs"; LOG=$W/logs/lc1_all.log
exec 8>"$W/queue.lock"; flock -n 8 || exit 3
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
log "queue pid=$$ lc1_long $(md5sum $W/lc1_long.py | cut -c1-8) waiting for the card lock"
exec 9>"${KVREL_WORK}/gpu0.lock"; flock -w 14400 9 || { log "card lock not released in 4 h, exit"; exit 4; }
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU)
others(){ nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$uuid"; }
w=0; while [ "$(others)" != 0 ] && [ $w -lt 60 ]; do sleep 60; w=$((w+1)); done
[ "$(others)" != 0 ] && { log "card has a compute process of someone, exit"; exit 5; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU
log "start"
run(){  # short dir order dtype suffix
  local out=$W/res/lc1_$1$5.json
  [ -s "$out" ] && { log "$1$5 exists, skipped"; return; }
  [ -d "$M/$2" ] || { log "$1$5 rc=97 gpu (model directory missing)"; return; }
  timeout 14400 python3 "$W/lc1_long.py" real "$out" "$M/$2" "$1" --order "$3" --data "$DATA" --L 32768 \
    --dtype "$4" > "$W/logs/lc1_$1$5.log" 2>&1 < /dev/null 8>&- 9>&-
  local rc=$?; echo $rc > "$W/logs/lc1_$1$5.rc"
  log "$1$5 rc=$rc gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i $GPU)"
}
while read -r short dir order; do
  run "$short" "$dir" "$order" bfloat16 ""
done <<'EOF'
q3_0p6b Qwen3-0.6B hf
q25_1p5b Qwen2.5-1.5B-Instruct hf
llama3b Llama-3.2-3B-Instruct gguf
q25_7b Qwen2.5-7B-Instruct hf
r1q_7b DeepSeek-R1-Distill-Qwen-7B hf
llama31_8b Meta-Llama-3.1-8B-Instruct gguf
mistral7b Mistral-7B-Instruct-v0.3 gguf
q3_8b Qwen3-8B hf
granite8b granite-3.1-8b-instruct gguf
olmo7b OLMo-2-1124-7B-Instruct hf
yi6b Yi-1.5-6B-Chat gguf
ds7b deepseek-llm-7b-chat_tokfix gguf
EOF
while read -r short dir order; do
  run "$short" "$dir" "$order" float32 "_f32"
done <<'EOF'
q3_0p6b Qwen3-0.6B hf
q25_1p5b Qwen2.5-1.5B-Instruct hf
llama3b Llama-3.2-3B-Instruct gguf
EOF
log "QUEUE_DONE"
