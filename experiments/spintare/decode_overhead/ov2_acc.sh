#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
# OV2 G2 (KL part) and G3 (REGISTRATION_OV2.md) on one RTX 5090 D, exclusive card (gpu0.lock).
# llama-perplexity --kl-divergence against the logits of the same model with an F16 K / F16 V cache (mu6 build, no
# mu, -ub 2), K4V4 arms with -ub 2 so that every graph is a decode graph:
#   2K context, 32 chunks: mu3 plain, mu3 kqbias (G2 references), mu6 plain, mu6 kqbias (G2: bit-identical KL output
#   to the mu3 arms), mu6 kqpos (G3 gate against mu6 kqbias);
#   32K context, 4 chunks: mu6 kqbias, mu6 kqpos (G3, descriptive).
# Text: long_text + qasper_text + mfqa_text of data/, joined by blank lines (g3_text.txt).
# Base logits are written to ${KVREL_WORK}/ov2/kl (deleted at the end of each model and context).
# usage: setsid nohup bash ov2_acc.sh > /dev/null 2>&1 < /dev/null &
. "$(cd "$(dirname "$0")/../../.." && pwd)/scripts/env.sh"  # KVREL_* paths (see docs/setup.md)
set -u
W=${KVREL_WORK}/ov2; O1=${KVREL_WORK}/ov1; D=$O1/gguf; F=$O1/files
# B6: the SpinTare tree (scripts/build_llama_cpp.sh)
B6=${KVREL_LLAMA_CPP}/build/bin; B3=$O1/src/llama.cpp-mu3/build/bin; GPU=0
R=$W/acc; K=$W/kl; mkdir -p "$R" "$K"; LOG=$W/acc.log; TXT=$W/files/g3_text.txt
exec 8>"$W/acc.lock"; flock -n 8 || exit 3
exec 9>"${KVREL_WORK}/gpu0.lock"; flock -w 14400 9 || { echo "$(date -u +%FT%TZ) GPU lock not free, exit" >> "$LOG"; exit 4; }
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU)
others(){ nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep "$uuid" \
  | awk -F', ' '{print $2":"$3}' | tr '\n' ' '; }
unset LLAMA_KV_DEBIAS_MU LLAMA_KV_DEBIAS_MU_PATH
grep -q "BUILD rc=0" "$W/build.log" || { log "mu6 build not finished, exit"; exit 5; }
chk(){ [ -s "$1" ] && [ "$(md5sum "$1" | cut -c1-8)" = "$2" ] || { log "INPUT_MISMATCH $1 (want $2), exit"; exit 6; }; }
chk $D/qwen25_7b-q8_0.gguf bda7a6f3; chk $D/qwen3_1p7b-q8_0.gguf 31d0f479
chk $F/qwen25_7b.kvmu 7343edd3; chk $F/qwen3_1p7b.kvmu a89de7de
chk $TXT 79e64a56; chk $W/files/mu6_vs_mu3.diff 28774a8c                  # text; source diff of the mu6 tree
chk $B3/llama-perplexity 91073bb6; chk $B3/libllama.so 83b73961; chk $B3/libggml-cuda.so c18ca875   # OV1 mu3 build
log "ACC_START text=$(md5sum $TXT | cut -c1-8) mu6=[$(md5sum $B6/llama-perplexity $B6/libllama.so $B6/libggml-cuda.so | cut -c1-8 | tr '\n' ' ')]" \
  "mu3=[$(md5sum $B3/llama-perplexity $B3/libllama.so $B3/libggml-cuda.so | cut -c1-8 | tr '\n' ' ')]"
# run <out> <bin dir> <env or -> <perplexity args...>
run(){
  local out=$1 bin=$2 envs=$3; shift 3
  local e=""; [ "$envs" != - ] && e=$envs
  [ -n "$(others)" ] && { log "FOREIGN process before $out; stop"; exit 7; }
  env CUDA_VISIBLE_DEVICES=$GPU LD_LIBRARY_PATH=$bin $e "$bin/llama-perplexity" "$@" > "$R/$out.log" 2>&1 8>&- 9>&-
  local rc=$?
  log "RUN $out rc=$rc env=[$e] $(grep -E "Mean +KLD" "$R/$out.log" | head -1 | tr -s ' ')"
}
for tag in qwen25_7b qwen3_1p7b; do
  G=$D/$tag-q8_0.gguf; MU=$F/$tag.kvmu; M="LLAMA_KV_DEBIAS_MU=$MU LLAMA_KV_DEBIAS_MU_PATH"
  for spec in "2k 2048 32" "32k 32768 4"; do
    set -- $spec; ctx=$1; c=$2; n=$3
    A="-m $G -f $TXT -c $c --chunks $n -fa on -ngl 99 -ub 2"
    run ${tag}_${ctx}_base $B6 - $A -ctk f16 -ctv f16 --kl-divergence-base $K/${tag}_${ctx}.kld
    if [ $ctx = 2k ]; then
      run ${tag}_${ctx}_mu3_plain  $B3 - $A -ctk q4_0 -ctv q4_0 --kl-divergence-base $K/${tag}_${ctx}.kld --kl-divergence
      run ${tag}_${ctx}_mu3_kqbias $B3 "$M=kqbias" $A -ctk q4_0 -ctv q4_0 --kl-divergence-base $K/${tag}_${ctx}.kld --kl-divergence
      run ${tag}_${ctx}_mu6_plain  $B6 - $A -ctk q4_0 -ctv q4_0 --kl-divergence-base $K/${tag}_${ctx}.kld --kl-divergence
    fi
    run ${tag}_${ctx}_mu6_kqbias $B6 "$M=kqbias" $A -ctk q4_0 -ctv q4_0 --kl-divergence-base $K/${tag}_${ctx}.kld --kl-divergence
    run ${tag}_${ctx}_mu6_kqpos  $B6 "$M=kqpos"  $A -ctk q4_0 -ctv q4_0 --kl-divergence-base $K/${tag}_${ctx}.kld --kl-divergence
    rm -f $K/${tag}_${ctx}.kld
  done
done
log "ACC_DONE"
