#!/bin/bash
# SGLang NVFP4 K/V cache, LongBench 8K (registry B4-NV, SGLang amendment): bf16, nvfp4 + postmean, nvfp4; 200 questions
# per task, paired CI against bf16. Original runs: one RTX 5090 in a container, 2026-10-21, sg_run.sh then sg_rerun.sh.
# The first Qwen3-0.6B nvfp4pm cell failed to serve (the hook put the mean on the CPU, which breaks CUDA graph
# capture); sg_rerun.sh repeated the nvfp4pm cells of both models with the fixed hook in sg_pm/ (shipped here).
# This script is sg_run.sh with paths made relative; with the fixed hook one pass covers every cell.
# usage: run_nvsg.sh   (needs two idle GPUs; SGLang 0.5.22 in $KVREL_SGLANG_VENV, `pip install "sglang[all]==0.5.22"`)
. "$(cd "$(dirname "$0")/../../.." && pwd)/scripts/env.sh"
HERE=$(cd "$(dirname "$0")" && pwd); TASKS=$KVREL_ROOT/experiments/spintare/tasks
V=${KVREL_SGLANG_VENV:-$KVREL_WORK/sglang_env}/bin; PY=$KVREL_PYTHON
R=$KVREL_WORK/results; LOG=$R/nvsg_run.log; mkdir -p $R/server $R/claim
PM=$KVREL_RESULTS/spintare/long_context_llamacpp/kvmu   # same postmean files as E2 (md5 a9ac63c9, b33d580f)
log(){ echo "$(date -u +%FT%TZ) $*" >> $LOG; }
claim(){
  while true; do
    for row in $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' '); do
      IFS=, read -r g m u <<< "$row"
      [ $m -lt 2000 ] && [ $u -lt 5 ] && mkdir $R/claim/gpu$g 2>/dev/null && { echo $g; return; }
    done
    sleep 60
  done
}
serve(){                                    # serve <gpu> <port> <model dir> <kv dtype> <postmean file or -> <mem frac> <name>
  local g=$1 port=$2 mdir=$3 dt=$4 pm=$5 util=$6 name=$7 envs=""
  [ "$pm" != - ] && envs="PYTHONPATH=$HERE/sg_pm SGLANG_KV_POSTMEAN=$pm"
  env CUDA_VISIBLE_DEVICES=$g SGLANG_ENABLE_JIT_DEEPGEMM=0 $envs $V/python -m sglang.launch_server --model-path $mdir \
    --served-model-name m --host 127.0.0.1 --port $port --kv-cache-dtype $dt --mem-fraction-static $util \
    --context-length 9000 --random-seed 0 > $R/server/sg_$name.log 2>&1 &
  SPID=$!
  for i in $(seq 360); do
    $V/python -c "import sys, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + sys.argv[1] + '/health', timeout=3)" $port > /dev/null 2>&1 && return 0
    kill -0 $SPID 2>/dev/null || return 1
    sleep 5
  done
  return 1
}
stop(){ kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; sleep 10; }
lane(){                                     # lane <tag> <model dir> <port> <mem frac> <postmean file>
  local tag=$1 mdir=$2 port=$3 util=$4 pm=$5 g
  g=$(claim); log "NVSG $tag gpu$g claimed"
  for spec in bf16:auto:- nvfp4pm:nvfp4:$pm nvfp4:nvfp4:-; do
    IFS=: read -r lab dt p <<< "$spec"
    [ -f $R/lc_vllm/$tag/summary_lc_${lab}_n200.json ] && continue
    if serve $g $port $mdir $dt $p $util ${tag}_$lab; then
      if [ "$p" != - ] && ! grep -q "\[postmean\] Qwen[23]Attention patched" $R/server/sg_${tag}_$lab.log; then
        log "NVSG $tag $lab rc=98 postmean hook not seen; cell aborted"; stop; continue
      fi
      $PY $TASKS/lc_lb_vllm.py --model $tag --hf $mdir --label ${lab}_n200 --port $port --n 200 \
        --start 0 --max_len 8000 > $R/sg_${tag}_$lab.log 2>&1
      log "NVSG $tag $lab rc=$? $(tail -n 1 $R/sg_${tag}_$lab.log) gpu$g"
    else
      log "NVSG $tag $lab rc=99 serve failed: $(grep -iE 'error|exception' $R/server/sg_${tag}_$lab.log | tail -n 2 | tr '\n' ' ' | cut -c1-200)"
    fi
    stop
    # interval as registered: the original paired_ci (one bootstrap index shared by the four tasks)
    [ $lab != bf16 ] && [ -f $R/lc_vllm/$tag/summary_lc_${lab}_n200.json ] && \
      log "NVSG-CI $tag $lab $(PYTHONUTF8=1 $PY $KVREL_ROOT/experiments/spintare/calib_and_diag/paired_ci.py $R/lc_vllm/$tag bf16_n200 ${lab}_n200 2>&1 | tail -n 1)"
  done
  rmdir $R/claim/gpu$g; log "NVSG $tag lane done"
}
M=$KVREL_MODELS
lane sg_qwen25_7b $M/Qwen2.5-7B-Instruct 8647 0.8 $PM/pm_qwen25_7b.kvmu &
sleep 30
lane sg_qwen3_0p6b $M/Qwen3-0.6B 8648 0.5 $PM/pm_qwen3_0p6b.kvmu &
wait
log "NVSG all done"
