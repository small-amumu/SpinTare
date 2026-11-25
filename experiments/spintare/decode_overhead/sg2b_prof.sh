#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# SG2 profile of the fused build (descriptive): where the remaining gate cost goes. nsys on the RTX 5090 D (exclusive), G0
# protocol of OV2: per decode step = (t[tg 288] - t[tg 32]) / 256 per kernel; plus the CUDA API summary (graph
# launches vs kernel launches). Arms kqpos and kqpos + gate, Qwen3-1.7B and Qwen2.5-7B, K4V4, depth 0 and 32768.
# Output: sg2b_prof/<model>_<arm>_d<depth>_n<tg>_{cuda_gpu_kern_sum,cuda_api_sum}.csv, log sg2b_prof/prof.log
. "$(cd "$(dirname "$0")/../.." && pwd)/scripts/env.sh"  # KVREL_* paths (see docs/setup.md)
set -u
W=${KVREL_WORK}/ov2; O1=${KVREL_WORK}/ov1; D=$O1/gguf; F=$O1/files
B=$W/src/llama.cpp-mu6/build/bin; P=$W/sg2b_prof; NSYS=/usr/local/cuda/bin/nsys
export CUDA_VISIBLE_DEVICES=0 TMPDIR=$W/tmp
mkdir -p "$P" "$TMPDIR"
exec 8>"$W/sg2b_prof.lock"; flock -n 8 || exit 3
exec 9>"${KVREL_WORK}/gpu0.lock"; flock -w 3600 9 || exit 4
log() { echo "$(date -u +%FT%TZ) $*" >> "$P/prof.log"; }
chk(){ [ -s "$1" ] && [ "$(md5sum "$1" | cut -c1-8)" = "$2" ] || { log "INPUT_MISMATCH $1"; exit 6; }; }
chk $D/qwen25_7b-q8_0.gguf bda7a6f3; chk $D/qwen3_1p7b-q8_0.gguf 31d0f479
chk $F/qwen25_7b.kvmu 7343edd3; chk $F/qwen3_1p7b.kvmu a89de7de; chk $W/files/sg2_src.diff 163364f1
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { log "FOREIGN process; not started"; exit 5; }
unset LLAMA_KV_DEBIAS_MU LLAMA_KV_DEBIAS_MU_PATH LLAMA_KV_MU_GATE
log "PROF_START bins=$(md5sum $B/llama-bench $B/libllama.so $B/libggml-cuda.so | cut -c1-8 | tr '\n' ' ')"
for tag in qwen3_1p7b qwen25_7b; do
  for d in 0 32768; do
    for arm in kqpos gate; do
      env="LLAMA_KV_DEBIAS_MU=$F/$tag.kvmu LLAMA_KV_DEBIAS_MU_PATH=kqpos"
      [ $arm = gate ] && env="$env LLAMA_KV_MU_GATE=1"
      for n in 32 288; do
        o=$P/${tag}_${arm}_d${d}_n$n
        rm -f "$o.nsys-rep" "$o.sqlite" "${o}_cuda_gpu_kern_sum.csv" "${o}_cuda_api_sum.csv"
        env LD_LIBRARY_PATH=$B $env "$NSYS" profile -t cuda --cuda-graph-trace=node -o "$o" -f true \
          "$B/llama-bench" -m "$D/$tag-q8_0.gguf" -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -p 0 -n $n -d $d -r 1 -o jsonl \
          > "$o.bench.jsonl" 2> "$o.err" 9>&-
        rc=$?
        "$NSYS" stats -q -r cuda_gpu_kern_sum,cuda_api_sum -f csv --force-overwrite true -o "$o" "$o.nsys-rep" \
          >> "$o.err" 2>&1 9>&-
        src=$?
        log "PROF $tag $arm d=$d n=$n rc=$rc stats=$src"
        rm -f "$o.sqlite" "$o.nsys-rep"
      done
    done
  done
done
log "PROF_DONE"
