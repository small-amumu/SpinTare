#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# E2 then E3 on one RTX PRO 5000 (GPU3; PREDICTIONS.md "E2/E3 design", 03:07:33Z). vLLM main 0c16eee3, TRITON_ATTN, eager.
. "$(cd "$(dirname "$0")/../../.." && pwd)/scripts/env.sh"  # KVREL_* paths (see docs/setup.md)
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/common.sh"
export LD_LIBRARY_PATH=$KVREL_LD_PATH
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
GPU=3; PORT=8097; cd $KB; mkdir -p $KB/results/server $KB/results/e3 /dev/shm/e3
BACKEND_FLAG=""
${KVREL_VENV}/bin/vllm serve --help=all 2>/dev/null | grep -q -- "--attention-backend" && BACKEND_FLAG="--attention-backend TRITON_ATTN"
waitfree(){ local f; while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU | tr -d ' ')
  [ $f -ge 22000 ] && { echo $f; return; }; sleep 300; done; }
vserve(){                                   # vserve <model_dir> <kv dtype> <name> <postmean file or ->
  local free util envs=""
  [ "$4" != - ] && envs="PYTHONPATH=$HERE/vllm_pm VLLM_KV_POSTMEAN=$4"
  local try i
  for try in 1 2 3; do                      # retry only the shared-GPU memory-profiling race
    free=$(waitfree)
    util=$(python3 -c "print(round(min(0.9, ($free - 2500) / 48935), 3))")
    env CUDA_VISIBLE_DEVICES=$GPU $envs ${KVREL_VENV}/bin/vllm serve $1 --served-model-name m --port $PORT \
      --kv-cache-dtype $2 --max-model-len 9000 --gpu-memory-utilization $util --enforce-eager --seed 0 \
      $BACKEND_FLAG > $KB/results/server/e2_$3.log 2>&1 &
    SPID=$!
    for i in $(seq 180); do
      curl -sf localhost:$PORT/v1/models > /dev/null && return 0; kill -0 $SPID 2>/dev/null || break; sleep 5
    done
    kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
    grep -q "Error in memory profiling" $KB/results/server/e2_$3.log || return 1; sleep 60
  done
  return 1
}
stop(){ kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; sleep 10; }
Q7=${KVREL_MODELS}/Qwen2.5-7B-Instruct; L8=${KVREL_MODELS}/Meta-Llama-3.1-8B-Instruct; PM7=$KB/results/b3/pm/qwen25_7b.kvmu
E2D=$KB/results/e2; mkdir -p $E2D
[ -f $E2D/rc.kvmu ] || $PY $HERE/make_kvmu_e2.py $PM7 $E2D/zero.kvmu $E2D/rc.kvmu > $E2D/kvmu.log 2>&1
# ---- E2
for arm in bf16lp:- bf16pmlp:$PM7 bf16z:$E2D/zero.kvmu bf16rc:$E2D/rc.kvmu; do
  lab=${arm%%:*}; pm=${arm#*:}
  [ -f $KB/results/lc_vllm/v8_qwen25_7b/summary_lc_${lab}_n20.json ] && continue
  if vserve $Q7 auto $lab $pm && { [ $pm = - ] || grep -q "\[postmean\] Qwen2Attention patched" $KB/results/server/e2_$lab.log; }; then
    ${KVREL_PYTHON} $HERE/lc_lb_vllm_lp.py --model v8_qwen25_7b --hf $Q7 --label ${lab}_n20 --port $PORT --n 20 \
      --start 0 --max_len 8000 > $E2D/$lab.log 2>&1
    runlog E2 qwen25_7b $lab $? "$(tail -n 1 $E2D/$lab.log)"
  else
    runlog E2 qwen25_7b $lab 99 "serve failed or hook missing: $(tail -n 2 $KB/results/server/e2_$lab.log | tr '\n' ' ' | cut -c1-200)"
  fi
  stop
done
runlog E2 all - 0 "E2 lane done"
# ---- E3
e3(){                                       # e3 <tag> <model_dir> <kv> <label> <pm file or -> [ref]
  local tag=$1 out=$KB/results/e3/$1_$4.json envs="" free util refflag=""
  [ -f $out ] && return
  [ "$5" != - ] && envs="PYTHONPATH=$HERE/vllm_pm VLLM_KV_POSTMEAN=$5"
  [ "$6" = ref ] && refflag=--ref
  local try rc
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
runlog E3 all - 0 "E3 lane done"
