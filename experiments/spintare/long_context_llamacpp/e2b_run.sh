#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# E2 phase B (registry X-0086): repairs vs context length. Same text, flags and K/V formats as phase A (e2a_run.sh).
# Original run: one node with 2 x V100-16GB, 2026-11-07, script e2b_run.sh md5 4e5643b8, all cells in one research
# llama.cpp tree (llama.cpp-krot-llama, learned-rotation switch off). Changes here: paths, and each cell runs on the
# shipped tree that carries its switch:
#   q4_0     no repair (repeat of phase A)                                   $KVREL_LLAMA_CPP     (switches off)
#   pm       LLAMA_KV_POSTMEAN=<pm file>: static post-RoPE key mean,         $KVREL_LLAMA_CPP_PM  (postmean patch)
#            calibrated on 2048 tokens
#   abmu     LLAMA_KV_DEBIAS_MU=<mu file>: pre-RoPE mean subtracted, R(p) mu $KVREL_LLAMA_CPP     (SpinTare patch;
#            added back exactly; LLAMA_KV_DEBIAS_MU_PATH=kadd = the add-back                          kadd = original
#            graph of the original tree                                                              graph)
#   abbias   LLAMA_KV_DEBIAS=1: k_proj bias subtracted pre-RoPE, added back  $KVREL_LLAMA_CPP
#            exactly (qwen2 arch only)
# The f16 reference (KL base) comes from $KVREL_LLAMA_CPP with every switch off.
# usage: e2b_run.sh <gpu 0|1|0,1> <model tag> <native max ctx> "<lengths>" <arch qwen2|other> [pm file] [mu file]
. "$(cd "$(dirname "$0")/../../.." && pwd)/scripts/env.sh"
set -u
unset LLAMA_KV_POSTMEAN LLAMA_KV_DEBIAS LLAMA_KV_DEBIAS_MU LLAMA_KV_DEBIAS_MU_PATH LLAMA_KV_DEBIAS_PATH \
      LLAMA_KV_DEBIAS_PAIRS LLAMA_KV_MU_GATE LLAMA_ATTN_ROT_DISABLE
NFAIL=0
g=$1; tag=$2; nat=$3; LENS=$4; arch=$5; PM=${6:-}; MU=${7:-}; CHILD=
trap '[ -n "$CHILD" ] && kill -TERM $CHILD 2>/dev/null; exit 143' TERM INT
B=$KVREL_LLAMA_CPP/build/bin; BPM=$KVREL_LLAMA_CPP_PM/build/bin; G=$KVREL_GGUF
TXT=${E2A_TXT:-$KVREL_DATA/pg19_text.txt}; TXTMD5=${E2A_TXTMD5:-be2f3aee6d166cafc63b61e74c22461e}
O=$KVREL_WORK/runs/e2b/$tag; mkdir -p $O; exec 9>>$O/.lock; flock -n 9 || { echo "$tag locked"; exit 75; }
[ "$(md5sum < $TXT | cut -c1-32)" = $TXTMD5 ] || { echo "text md5"; exit 2; }
for f in $PM $MU; do [ -s $f ] || { echo "missing $f"; exit 2; }; done
# The shipped mean files are KVMU (format of the research tree and of LLAMA_KV_DEBIAS_MU); the postmean patch reads
# the same header and payload under the magic "KVPM". Convert a KVMU pm file once per run (payload unchanged).
if [ -n "$PM" ] && [ "$(head -c 4 $PM)" = KVMU ]; then
  { printf KVPM; tail -c +5 $PM; } > $O/pm.kvpm; PM=$O/pm.kvpm
fi
echo "$(date -u +%FT%TZ) start $tag gpu=$g ppl=$(md5sum < $B/llama-perplexity | cut -c1-8) ppl_pm=$(md5sum < $BPM/llama-perplexity | cut -c1-8) gguf=$(md5sum < $G/$tag.gguf | cut -c1-8) pm=${PM:+$(md5sum < $PM | cut -c1-8)} mu=${MU:+$(md5sum < $MU | cut -c1-8)}" >> $O/run.log
for L in $LENS; do
  [ $L -le $nat ] || continue
  ch=$((32768 / L)); [ $ch -lt 1 ] && ch=1; base=$O/kld_$L.bin
  run() {  # run <bin dir> <ctk> <env assignment> <name> [--kl-divergence]
    env LD_LIBRARY_PATH=$1:${LD_LIBRARY_PATH:-} $3 CUDA_VISIBLE_DEVICES=$g $1/llama-perplexity -m $G/$tag.gguf -f $TXT \
      -c $L --chunks $ch -b 2048 -fa on -ctv q8_0 -ngl 99 -ctk $2 --kl-divergence-base $base ${5:-} \
      > $O/${4}_$L.log 2>&1 < /dev/null &
    CHILD=$!; wait $CHILD; local rc=$?; CHILD=
    echo "$(date -u +%FT%TZ) $tag L=$L $4 rc=$rc chunks_done=$(grep -cE '^ +[0-9]+ ' $O/${4}_$L.log) want=$ch $(grep -E '^Mean +KLD' $O/${4}_$L.log | head -1)" >> $O/run.log
    [ $rc = 0 ] || NFAIL=$((NFAIL+1))
    return $rc
  }
  rm -f $base
  run $B f16 E2B_NONE=0 f16 || { echo "$(date -u +%FT%TZ) $tag L=$L f16 reference failed, skip" >> $O/run.log; rm -f $base; continue; }
  [ -s $base ] || { echo "$(date -u +%FT%TZ) $tag L=$L no KL base, skip" >> $O/run.log; rm -f $base; NFAIL=$((NFAIL+1)); continue; }
  run $B q4_0 E2B_NONE=0 q4_0 --kl-divergence
  [ -n "$PM" ] && run $BPM q4_0 LLAMA_KV_POSTMEAN=$PM pm --kl-divergence
  [ -n "$MU" ] && run $B q4_0 "LLAMA_KV_DEBIAS_MU=$MU LLAMA_KV_DEBIAS_MU_PATH=kadd" abmu --kl-divergence
  [ "$arch" = qwen2 ] && run $B q4_0 LLAMA_KV_DEBIAS=1 abbias --kl-divergence
  rm -f $base
done
echo "$(date -u +%FT%TZ) done $tag txt=$(basename $TXT) nfail=$NFAIL" >> $O/run.log
[ $NFAIL = 0 ] || exit 1
