#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# SG2 T1 (REGISTRATION_SG2.md) on one RTX 5090 D: the OV2 G4 protocol (ov2_time.sh) with the arms of SG1 (gate = SG2 build).
# llama-bench tg128 at depth 0 / 8192 / 32768, 5 repetitions, 3 rounds (forward / reverse / forward), plain k4v4 twice
# per round (A/A and drift). Arms per model: k4v4 (plain), k4v4mu_kqpos, k4v4mu_gate (kqpos + LLAMA_KV_MU_GATE=1),
# k4v8, k4v8mu_kqpos, k4v8mu_gate (all on the mu6 build with the SG1 patch). GPU state and foreign processes are logged
# before and after each bench; a bench with a foreign process is rerun once.
# usage: setsid nohup bash sg2_time.sh > /dev/null 2>&1 < /dev/null &
. "$(cd "$(dirname "$0")/../.." && pwd)/scripts/env.sh"  # KVREL_* paths (see docs/setup.md)
set -u
W=${KVREL_WORK}/ov2; O1=${KVREL_WORK}/ov1; D=$O1/gguf; F=$O1/files
# B6: the SpinTare tree (experiments/scripts/build_llama_cpp.sh)
B6=${KVREL_LLAMA_CPP}/build/bin; GPU=0
R=$W/sg2_time; mkdir -p "$R"; LOG=$W/sg2_time.log
exec 8>"$W/sg2_time.lock"; flock -n 8 || exit 3
exec 9>"${KVREL_WORK}/gpu0.lock"; flock -w 14400 9 || { echo "$(date -u +%FT%TZ) GPU lock not free, exit" >> "$LOG"; exit 4; }
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU)
others(){ nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep "$uuid" \
  | awk -F', ' '{print $2":"$3}' | tr '\n' ' '; }
gstate(){ nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate --format=csv,noheader -i $GPU \
  | tr -d ' '; }
unset LLAMA_KV_DEBIAS_MU LLAMA_KV_DEBIAS_MU_PATH LLAMA_KV_MU_GATE
chk(){ [ -s "$1" ] && [ "$(md5sum "$1" | cut -c1-8)" = "$2" ] || { log "INPUT_MISMATCH $1 (want $2), exit"; exit 6; }; }
chk $D/qwen25_7b-q8_0.gguf bda7a6f3; chk $D/qwen3_1p7b-q8_0.gguf 31d0f479
chk $F/qwen25_7b.kvmu 7343edd3; chk $F/qwen3_1p7b.kvmu a89de7de
chk $W/files/sg2_src.diff 163364f1   # pins the SG2 source tree (diff against the SG1 tree)
log "TIME_START mu6=[$(md5sum $B6/llama-bench $B6/libllama.so $B6/libggml-cuda.so | cut -c1-8 | tr '\n' ' ')]" \
  "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i $GPU)"
for tag in qwen25_7b qwen3_1p7b; do
  G=$D/$tag-q8_0.gguf; MU=$F/$tag.kvmu
  # name:ctk:ctv:arm (arm - = no mu)
  CFGS="k4v4:q4_0:q4_0:- k4v4mu_kqpos:q4_0:q4_0:kqpos k4v4mu_gate:q4_0:q4_0:gate \
k4v8:q4_0:q8_0:- k4v8mu_kqpos:q4_0:q8_0:kqpos k4v8mu_gate:q4_0:q8_0:gate"
  REV=$(echo $CFGS | tr ' ' '\n' | tac | tr '\n' ' ')
  for round in 1 2 3; do
    order="$CFGS"; [ $round = 2 ] && order="$REV"
    for c in k4v4_aa0:q4_0:q4_0:- $order k4v4_aa1:q4_0:q4_0:-; do
      IFS=: read -r name k v p <<< "$c"
      env=""; [ "$p" != - ] && env="LLAMA_KV_DEBIAS_MU=$MU LLAMA_KV_DEBIAS_MU_PATH=kqpos"
      [ "$p" = gate ] && env="$env LLAMA_KV_MU_GATE=1"
      out=$R/${tag}_${name}_r$round
      for attempt in 1 2; do
        w=0; while [ -n "$(others)" ] && [ $w -lt 30 ]; do sleep 60; w=$((w+1)); done
        [ $w -gt 0 ] && log "WAIT $tag $name round=$round waited=${w}min for foreign processes on GPU$GPU"
        o0=$(others); g0=$(gstate); l0=$(cut -d' ' -f1 /proc/loadavg)
        env CUDA_VISIBLE_DEVICES=$GPU LD_LIBRARY_PATH=$B6 $env "$B6/llama-bench" -m "$G" -ngl 99 -fa 1 -ctk $k -ctv $v \
          -p 0 -n 128 -d 0,8192,32768 -r 5 -o jsonl > "$out.jsonl" 2> "$out.err" 8>&- 9>&-
        rc=$?; o1=$(others); g1=$(gstate); l1=$(cut -d' ' -f1 /proc/loadavg)
        ts=$(python3 -c "import json,sys; d=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]; \
print(' '.join(f\"d{x['n_depth']}={x['avg_ts']:.3f}+-{x['stddev_ts']:.3f}\" for x in d))" "$out.jsonl" 2>/dev/null)
        log "TG $tag $name round=$round attempt=$attempt rc=$rc $ts env=[$env] others=[$o0|$o1]" \
          "gpu=[$g0|$g1] load=[$l0|$l1]"
        [ -z "$o0$o1" ] && break
        sleep 60
      done
    done
  done
done
log "TIME_DONE"
