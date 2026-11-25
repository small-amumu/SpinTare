#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
# SG2 E1-E2 (REGISTRATION_SG2.md) on one RTX 5090 D, exclusive card (gpu0.lock); mu6 tree with SG1 + SG2.
# llama-perplexity --kl-divergence against the same model with an F16 K / F16 V cache (no mu, -ub 2), K4V4, 2K,
# 32 chunks: kqpos with the gate unset (E2), gate=1 with -ub 2 (E1 decode) and with the default ubatch (E1 prefill).
# Compared with the SG1 logs by sg2_agg.py. Base logits in ${KVREL_WORK}/ov2/kl, deleted after each model.
# usage: setsid nohup bash sg2_acc.sh > /dev/null 2>&1 < /dev/null &
. "$(cd "$(dirname "$0")/../../.." && pwd)/scripts/env.sh"  # KVREL_* paths (see docs/setup.md)
set -u
W=${KVREL_WORK}/ov2; O1=${KVREL_WORK}/ov1; D=$O1/gguf; F=$O1/files
# B6: the SpinTare tree (scripts/build_llama_cpp.sh)
B6=${KVREL_LLAMA_CPP}/build/bin; GPU=0
R=$W/sg2_acc; K=$W/kl; mkdir -p "$R" "$K"; LOG=$W/sg2_acc.log; TXT=$W/files/g3_text.txt
exec 8>"$W/sg2_acc.lock"; flock -n 8 || exit 3
exec 9>"${KVREL_WORK}/gpu0.lock"; flock -w 14400 9 || { echo "$(date -u +%FT%TZ) GPU lock not free, exit" >> "$LOG"; exit 4; }
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU)
others(){ nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep "$uuid" \
  | awk -F', ' '{print $2":"$3}' | tr '\n' ' '; }
unset LLAMA_KV_DEBIAS_MU LLAMA_KV_DEBIAS_MU_PATH LLAMA_KV_MU_GATE
chk(){ [ -s "$1" ] && [ "$(md5sum "$1" | cut -c1-8)" = "$2" ] || { log "INPUT_MISMATCH $1 (want $2), exit"; exit 6; }; }
chk $D/qwen25_7b-q8_0.gguf bda7a6f3; chk $D/qwen3_1p7b-q8_0.gguf 31d0f479
chk $F/qwen25_7b.kvmu 7343edd3; chk $F/qwen3_1p7b.kvmu a89de7de
chk $TXT 79e64a56
chk $W/files/sg2_src.diff 163364f1   # pins the SG2 source tree (diff against the SG1 tree)
log "SG2_ACC_START mu6=[$(md5sum $B6/llama-perplexity $B6/libllama.so $B6/libggml-cuda.so $B6/libllama-perplexity-impl.so | cut -c1-8 | tr '\n' ' ')]"
# run <out> <env or -> <perplexity args...>
run(){
  local out=$1 envs=$2; shift 2
  local e=""; [ "$envs" != - ] && e=$envs
  [ -n "$(others)" ] && { log "FOREIGN process before $out; stop"; exit 7; }
  env CUDA_VISIBLE_DEVICES=$GPU LD_LIBRARY_PATH=$B6 $e "$B6/llama-perplexity" "$@" > "$R/$out.log" 2>&1 8>&- 9>&-
  local rc=$?
  log "RUN $out rc=$rc env=[$e] $(grep -E "Mean +KLD" "$R/$out.log" | head -1 | tr -s ' ')"
}
for tag in qwen25_7b qwen3_1p7b; do
  G=$D/$tag-q8_0.gguf; MU=$F/$tag.kvmu
  P="LLAMA_KV_DEBIAS_MU=$MU LLAMA_KV_DEBIAS_MU_PATH=kqpos"
  A="-m $G -f $TXT -c 2048 --chunks 32 -fa on -ngl 99"
  KB="--kl-divergence-base $K/sg2_${tag}_2k.kld"
  run ${tag}_2k_base - $A -ub 2 -ctk f16 -ctv f16 $KB
  run ${tag}_2k_kqpos   "$P"                    $A -ub 2 -ctk q4_0 -ctv q4_0 $KB --kl-divergence
  run ${tag}_2k_gate    "$P LLAMA_KV_MU_GATE=1" $A -ub 2 -ctk q4_0 -ctv q4_0 $KB --kl-divergence
  run ${tag}_2k_pf_gate "$P LLAMA_KV_MU_GATE=1" $A -ctk q4_0 -ctv q4_0 $KB --kl-divergence
  rm -f $K/sg2_${tag}_2k.kld
done
log "SG2_ACC_DONE"
