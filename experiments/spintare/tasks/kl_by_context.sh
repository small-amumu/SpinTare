#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# KL inside / beyond the calibration window, CH4X A7: teacher-forced KL against F16 K / F16 V inside (positions 1024-2047) and
# beyond (4096-8191) the 2K calibration window; -ub 2 decode graphs, V q8_0, PDL off, -c CTX --chunks NCH.
# <bin final>: SpinTare tree; <bin pm>: postmean tree (scripts/build_llama_cpp.sh); <mu>: KVMU pre-RoPE mean;
# <m>: static mean for LLAMA_KV_POSTMEAN; <text>: data/long_text + qasper_text + mfqa_text joined by blank
# lines; <lock...>: one or more lock files held while the GPU is used. The switch values are passed
# unquoted to env, so the mean-file paths must not contain spaces.
# llama-perplexity scores the second half of each chunk, so -c 8192 --chunks 2 scores positions 4096-8191 (8192
# tokens), against the build check's -c 2048 --chunks 8 (positions 1024-2047, 8192 tokens), on the same text.
# Arms: plain, SpinTare (kqpos + gate) on the final build; plain and postmean on the pm build (the postmean reference).
# Report deltas against plain on the same tokens; absolute KL across the two contexts is not paired. -b CTX keeps
# the whole chunk in one logical batch, as at 2048 (the default there).
# Runs 2048 x 8 first (repeats the build check, now with PDL off) and then 8192 x 2, in one GPU-lock hold.
# usage: setsid nohup bash ch4x_klctx.sh <bin final> <bin pm> <gguf> <mu> <m> <text> <out dir> <gpu> <lock...>
#        > <out dir>/launch.out 2>&1 < /dev/null &
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID                         # nvidia-smi -i and CUDA_VISIBLE_DEVICES agree
[ $# -ge 9 ] || { echo "usage: need 8 args and at least one lock"; exit 2; }
BF=$1; BP=$2; G=$3; MU=$4; PM=$5; TXT=$6; R=$7; GPU=$8; shift 8; LOCKS="$*"
mkdir -p "$R"; LOG=$R/check.log; K=$R/base.kld
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
exec 8>>"$R/run.lock"; flock -n 8 || { log "run lock held, exit"; exit 3; }
log "waiting for GPU locks [$LOCKS]"
fd=6; for l in $LOCKS; do eval "exec $fd>>\"$l\""; flock -w 14400 $fd || { log "lock $l not free, exit"; exit 4; }; fd=$((fd - 1)); done
for i in $(seq 1 30); do                                   # the previous cell's server may still be exiting
  apps=$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader) || { log "nvidia-smi failed, exit"; exit 5; }
  [ -z "$apps" ] && break; sleep 10
done
[ -z "$apps" ] || { log "FOREIGN process on GPU $GPU ($apps), exit"; exit 5; }
for v in $(env | sed -n 's/^\(\(LLAMA\|GGML\)_[A-Za-z0-9_]*\)=.*/\1/p'); do unset $v; done
FMD5=$(cat "$BF/llama-server" $(ls "$BF"/*.so | sort) | md5sum | cut -c1-8)
PMD5=$(cat "$BP/llama-server" $(ls "$BP"/*.so | sort) | md5sum | cut -c1-8)
log "START final=$FMD5 pm=$PMD5 gguf=$(md5sum < $G | cut -c1-8) mu=$(md5sum < $MU | cut -c1-8) m=$(md5sum < $PM | cut -c1-8) text=$(md5sum < $TXT | cut -c1-8)"
bad=0
run(){
  local out=$1 b=$2 envs=$3 rc; shift 3
  local e=""; [ "$envs" != - ] && e=$envs
  env CUDA_VISIBLE_DEVICES=$GPU LD_LIBRARY_PATH=$b GGML_CUDA_PDL=0 $e "$b/llama-perplexity" "$@" > "$R/${out}_$CTX.log" 2>&1 < /dev/null 5>&- 6>&- 8>&-
  rc=$?; [ $rc = 0 ] || bad=1
  log "RUN $out ctx=$CTX rc=$rc env=[$e] $(grep -E "Mean +KLD" "$R/${out}_$CTX.log" | head -1 | tr -s ' ') $(grep -c 'converting K and V to f16' "$R/${out}_$CTX.log")xconvert"
}
for spec in 2048:8 8192:2; do
CTX=${spec%%:*}; NCH=${spec##*:}
A="-m $G -f $TXT -c $CTX -b $CTX --chunks $NCH -fa on -ngl 99 -ub 2"
rm -f "$K"
run base $BF - $A -ctk f16 -ctv f16 --kl-divergence-base $K
[ $bad = 0 ] && [ -s "$K" ] || { log "no KL base (rc or file), exit"; exit 6; }
run final_plain $BF - $A -ctk q4_0 -ctv q8_0 --kl-divergence-base $K --kl-divergence
run final_gate $BF "LLAMA_KV_DEBIAS_MU=$MU LLAMA_KV_DEBIAS_MU_PATH=kqpos LLAMA_KV_MU_GATE=1" $A -ctk q4_0 -ctv q8_0 --kl-divergence-base $K --kl-divergence
run pm_plain $BP - $A -ctk q4_0 -ctv q8_0 --kl-divergence-base $K --kl-divergence
run pm_postmean $BP "LLAMA_KV_POSTMEAN=$PM" $A -ctk q4_0 -ctv q8_0 --kl-divergence-base $K --kl-divergence
rm -f "$K"
for a in base final_plain final_gate pm_plain pm_postmean; do
  n=$(grep -c "\[[0-9]*\]" "$R/${a}_$CTX.log"); log "$a ctx=$CTX: chunks reported $n"
done
done
if [ $bad = 0 ]; then log "VERDICT ok"; else log "VERDICT fail"; fi
log "DONE bad=$bad"
exit $bad
