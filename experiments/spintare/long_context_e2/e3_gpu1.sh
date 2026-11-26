#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# E3 on one RTX PRO 5000 (GPU1, placement amendment): the same cells and settings as the E3 part of e23_run.sh.
. "$(cd "$(dirname "$0")/../.." && pwd)/scripts/env.sh"  # KVREL_* paths (see docs/setup.md)
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/common.sh"
export LD_LIBRARY_PATH=$KVREL_LD_PATH
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
GPU=1; cd $KB; mkdir -p $KB/results/e3 /dev/shm/e3
waitfree(){ local f; while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU | tr -d ' ')
  [ $f -ge 22000 ] && { echo $f; return; }; sleep 120; done; }
Q7=${KVREL_MODELS}/Qwen2.5-7B-Instruct; L8=${KVREL_MODELS}/Meta-Llama-3.1-8B-Instruct; PM7=$KB/results/b3/pm/qwen25_7b.kvmu
e3(){                                       # e3 <tag> <model_dir> <kv> <label> <pm file or -> [ref]
  local tag=$1 out=$KB/results/e3/$1_$4.json envs="" free util refflag="" try rc
  [ -f $out ] && return
  [ "$5" != - ] && envs="PYTHONPATH=$HERE/vllm_pm VLLM_KV_POSTMEAN=$5"
  [ "$6" = ref ] && refflag=--ref
  for try in 1 2 3; do                      # retry only the shared-GPU memory-profiling race
    free=$(waitfree); util=$(python3 -c "print(round(min(0.85, ($free - 8500) / 48935), 3))")
    env CUDA_VISIBLE_DEVICES=$GPU $envs ${KVREL_PYTHON} $HERE/e3_kld.py --model $2 --kv $3 --label $4 \
      --refdir /dev/shm/e3/$tag --util $util --out $out $refflag > $KB/results/e3/$1_$4.log 2>&1
    rc=$?; [ $rc = 0 ] && break
    grep -q "Error in memory profiling" $KB/results/e3/$1_$4.log || break; sleep 60
  done
  runlog E3 $tag $4 $rc "$(grep -m1 '^E3 ' $KB/results/e3/$1_$4.log | cut -c1-300)"
}
e3 qwen25_7b $Q7 auto ref - ref
for c in "fp8 fp8 -" "fp8 fp8pm $PM7" "int4_per_token_head int4 -" "int4_per_token_head int4pm $PM7" "auto bf16pm $PM7"; do
  set -- $c; e3 qwen25_7b $Q7 $1 $2 $3
done
rm -rf /dev/shm/e3/qwen25_7b
e3 llama31_8b $L8 auto ref - ref
for c in "int4_per_token_head int4 -" "fp8 fp8 -"; do set -- $c; e3 llama31_8b $L8 $1 $2 $3; done
rm -rf /dev/shm/e3/llama31_8b
runlog E3 all - 0 "E3 lane done (GPU1)"
