#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# The full E2 grid (registry X-0081 / X-0085 / X-0086 and its two amendments), in the order the original
# LSF jobs ran on one node (2 x V100-16GB; e2a*.lsf, e2b*.lsf). GPUs 0 and 1; lengths > 32768 use both.
# Needs: experiments/scripts/build_llama_cpp.sh spintare and postmean; q8_0 weight GGUFs in $KVREL_GGUF named as below
# (experiments/scripts/convert_gguf.sh, then llama-quantize ... q8_0); mean files in $KVREL_RESULTS/spintare/long_context_llamacpp/kvmu.
# Afterwards: python e2_analyze.py $KVREL_WORK/runs --json e2_final.json (compare with the shipped e2_final.json).
. "$(cd "$(dirname "$0")/../.." && pwd)/scripts/env.sh"
HERE=$(cd "$(dirname "$0")" && pwd)
A=$HERE/e2a_run.sh; B=$HERE/e2b_run.sh; K=$KVREL_RESULTS/spintare/long_context_llamacpp/kvmu
LONG=$KVREL_DATA/pg19_long.txt
[ -s $LONG ] || cat $KVREL_DATA/pg19_text.txt $KVREL_DATA/pg19_text2.txt > $LONG
trap 'trap - TERM; kill -TERM 0' TERM

# phase A, L <= 32768 (e2a.lsf, e2a_b.lsf, e2a_c.lsf)
( bash $A 0 qwen25_7b-q8_0 32768; bash $A 0 q8_r1q7-q8_0 32768; bash $A 0 mistral7b_inj0-q8_0 32768 ) &
( bash $A 1 qwen3_0p6b-q8_0 40960; bash $A 1 q8_qwen25_1p5b-q8_0 32768; bash $A 1 llama31_8b-q8_0 131072 ) &
wait
bash $A 0,1 qwen25_14b-q8_0 32768
# phase A, long points on pg19_long (e2a_d.lsf)
E2A_TXT=$LONG E2A_TXTMD5=0a55cfc4bfbc889a0fc6f95623b9c39f bash $A 0,1 q8_r1q7-q8_0 131072 65536
E2A_TXT=$LONG E2A_TXTMD5=0a55cfc4bfbc889a0fc6f95623b9c39f bash $A 0,1 llama31_8b-q8_0 131072 131072

# phase B (e2b.lsf, then the denser grid of e2b_dense.lsf)
for SH in "2048 8192 32768" "4096 16384"; do
  ( bash $B 0 qwen25_7b-q8_0 32768 "$SH" qwen2 $K/pm_qwen25_7b.kvmu $K/mu_qwen25_7b.kvmu
    bash $B 0 qwen3_0p6b-q8_0 40960 "$SH" other $K/pm_qwen3_0p6b.kvmu $K/mu_qwen3_0p6b.kvmu ) &
  ( bash $B 1 q8_r1q7-q8_0 131072 "$SH" qwen2 $K/pm_r1q7.kvmu
    bash $B 1 llama31_8b-q8_0 131072 "$SH" other $K/pm_llama31_8b.kvmu ) &
  wait
done
export E2A_TXT=$LONG E2A_TXTMD5=0a55cfc4bfbc889a0fc6f95623b9c39f
bash $B 0,1 q8_r1q7-q8_0 131072 65536 qwen2 $K/pm_r1q7.kvmu
bash $B 0,1 llama31_8b-q8_0 131072 "65536 131072" other $K/pm_llama31_8b.kvmu
unset E2A_TXT E2A_TXTMD5
# phase B extension (e2b_more.lsf): exact k-bias add-back on two more qwen2-arch models
bash $B 0,1 q8_qwen25_1p5b-q8_0 32768 "2048 8192 32768" qwen2
bash $B 0,1 qwen25_14b-q8_0 32768 "2048 8192 32768" qwen2
