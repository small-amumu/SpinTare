#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
# E2 phase A (registry X-0081 / X-0085): length dependence of the engine output KL D without repair.
# Original run: one node with 2 x V100-16GB, 2026-11-06/07, script e2a_run_v3.sh (only paths changed here).
# usage: e2a_run.sh <gpu 0|1|0,1> <model tag> <native max ctx> ["<lengths>"]
#   tag = GGUF basename without .gguf in $KVREL_GGUF (the original runs used q8_0 weight GGUFs, e.g. llama31_8b-q8_0).
# Per length L (L <= native ctx): f16 K reference first (own KL base per L), then K q4_0 (engine Hadamard rotation on),
# q4_0 with LLAMA_ATTN_ROT_DISABLE=1, q8_0; V q8_0, -fa on, -b 2048. chunks = 32768 / L (each chunk scores its second
# half, so every L <= 32768 scores 16384 tokens); L > 32768 -> one chunk. Text: $E2A_TXT (default pg19_text.txt).
# Writes only $KVREL_WORK/runs/e2a/<tag>/. KL base files are deleted after each length.
. "$(cd "$(dirname "$0")/../../.." && pwd)/scripts/env.sh"
set -u
g=$1; tag=$2; nat=$3; LENS=${4:-"2048 8192 32768"}; CHILD=
trap '[ -n "$CHILD" ] && kill -TERM $CHILD 2>/dev/null; exit 143' TERM INT
B=$KVREL_LLAMA_CPP/build/bin; G=$KVREL_GGUF   # switches off -> upstream behaviour
TXT=${E2A_TXT:-$KVREL_DATA/pg19_text.txt}; TXTMD5=${E2A_TXTMD5:-be2f3aee6d166cafc63b61e74c22461e}
O=$KVREL_WORK/runs/e2a/$tag; mkdir -p $O; exec 9>>$O/.lock; flock -n 9 || { echo "$tag locked"; exit 75; }
export LD_LIBRARY_PATH=$B:${LD_LIBRARY_PATH:-}
unset LLAMA_KV_POSTMEAN LLAMA_KV_DEBIAS LLAMA_KV_DEBIAS_MU LLAMA_KV_MU_GATE LLAMA_ATTN_ROT_DISABLE
[ "$(md5sum < $TXT | cut -c1-32)" = $TXTMD5 ] || { echo "text md5"; exit 2; }
echo "$(date -u +%FT%TZ) start $tag gpu=$g ppl=$(md5sum < $B/llama-perplexity | cut -c1-8) gguf=$(md5sum < $G/$tag.gguf | cut -c1-8)" >> $O/run.log
for L in $LENS; do
  [ $L -le $nat ] || continue
  ch=$((32768 / L)); [ $ch -lt 1 ] && ch=1; base=$O/kld_$L.bin
  run() {  # run <ctk> <rot 0|1> <name> [--kl-divergence]
    local rotenv=LLAMA_ATTN_ROT_DISABLE_UNUSED=0; [ $2 = 0 ] && rotenv=LLAMA_ATTN_ROT_DISABLE=1
    env $rotenv CUDA_VISIBLE_DEVICES=$g $B/llama-perplexity -m $G/$tag.gguf -f $TXT -c $L --chunks $ch -b 2048 -fa on \
      -ctv q8_0 -ngl 99 -ctk $1 --kl-divergence-base $base ${4:-} > $O/${3}_$L.log 2>&1 < /dev/null &
    CHILD=$!; wait $CHILD; local rc=$?; CHILD=
    echo "$(date -u +%FT%TZ) $tag L=$L $3 rc=$rc chunks_done=$(grep -cE '^ +[0-9]+ ' $O/${3}_$L.log) want=$ch $(grep -E '^Mean +KLD' $O/${3}_$L.log | head -1)" >> $O/run.log
    return $rc
  }
  rm -f $base
  run f16 1 f16 || { echo "$(date -u +%FT%TZ) $tag L=$L f16 reference failed, skip" >> $O/run.log; rm -f $base; continue; }
  [ -s $base ] || { echo "$(date -u +%FT%TZ) $tag L=$L no KL base, skip" >> $O/run.log; continue; }
  run q4_0 1 q4_0 --kl-divergence
  run q4_0 0 q4_0norot --kl-divergence
  run q8_0 1 q8_0 --kl-divergence
  rm -f $base
done
echo "$(date -u +%FT%TZ) done $tag" >> $O/run.log
