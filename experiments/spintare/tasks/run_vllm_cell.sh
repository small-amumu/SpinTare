#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# One vLLM task cell (RULER 32K or LongBench 8K): start
# `vllm serve` for one K-cache arm, run the client, stop the server.
#
# usage: run_vllm_cell.sh <bench> <arm> <name>
#   bench  longbench | ruler
#   arm    bf16      16-bit cache (TRITON_ATTN)
#          int4      int4_per_token_head, no repair
#          pm        int4_per_token_head, static post-RoPE mean (VLLM_KV_POSTMEAN=$PM, KVMU file;
#                    import hook vllm_site/kv_postmean.py, Qwen2 / Qwen3 attention)
#          sa        int4_per_token_head, SmoothAttention (HF = folded checkpoint from
#                    ../smoothattention/sa_fold.py)
#          spintare  int4_per_token_head, backend SPINTARE_AB: rotary centring with exact
#                    add-back and the exemption gate (needs vLLM 5cb0fd265 with
#                    integrations/patches/vllm/spintare-addback.patch and `pip install -e integrations/vllm`)
#   name   model label; results go to $KVREL_WORK/results/{lc_vllm,ruler_vllm}/<name>/
# environment (after experiments/scripts/env.sh):
#   HF     Hugging Face model directory;  MU: KVMR file for spintare (python -m spintare_vllm
#          calib-mu <HF> data/long_text.txt <out.kvmr> --T 2048);  PM: KVMU mean file for pm
#   MML    max_model_len for RULER (32768 for Qwen2.5-7B, 33792 for Qwen3-1.7B in the reported runs)
#   RULER_DATA, LongBench data: as run_llamacpp_cell.sh;  PORT (default 8120), GPU (default 0,
#          PCI bus order)
#   VLLM_PYTHON  interpreter of the vLLM install (default $KVREL_PYTHON)
#   DRY_RUN=1 prints the commands only.
# Settings of the registered runs: eager mode, gpu_memory_utilization 0.9, one GPU (RTX PRO 5000);
# LongBench max_model_len 9000, prompts cut to 8000 tokens, 4 tasks x 200 items; RULER without
# prefix caching, 7 tasks x 50 items (the table prints niah_multikey_3, niah_multiquery, vt, qa_1),
# greedy with every penalty off. The SpinTare cells used the fused write kernel and tf32x3 dots
# (SPINTARE_AB_WRITE=fused SPINTARE_AB_VARIANT=1). SPINTARE_NO_HADACORE=1 (default here) was set in
# the INT4 arms as a machine workaround: vLLM's hadacore kernel was not usable on that GPU, so the
# INT4 cache used vLLM's other Hadamard path. Set SPINTARE_NO_HADACORE=0 where hadacore works.
# Needs Linux (/proc, setsid), bash >= 4.4 and curl.
set -euo pipefail
[ $# -eq 3 ] || { sed -n 2,33p "$0"; exit 2; }
BENCH=$1; ARM=$2; NAME=$3
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../../scripts/env.sh"
: "${HF:?set HF}"
PORT=${PORT:-8120}; GPU=${GPU:-0}; PY=${VLLM_PYTHON:-$KVREL_PYTHON}
NH=${SPINTARE_NO_HADACORE:-1}
BACKEND=TRITON_ATTN; KV=int4_per_token_head; SW=("SPINTARE_NO_HADACORE=$NH")
case $ARM in
  bf16)     KV=auto; SW=();;
  int4|sa)  ;;
  pm)       SW+=("VLLM_KV_POSTMEAN=${PM:?set PM}");;
  spintare) BACKEND=SPINTARE_AB
            SW+=("SPINTARE_KVMU=${MU:?set MU}" SPINTARE_AB_VARIANT=1 SPINTARE_AB_WRITE=fused
                 SPINTARE_GATE=1);;
  *) echo "unknown arm $ARM" >&2; exit 2;;
esac
SRV=(serve "$HF" --port "$PORT" --enforce-eager --served-model-name m --attention-backend "$BACKEND"
     --gpu-memory-utilization 0.9 --kv-cache-dtype "$KV")
case $BENCH in
  longbench)
    SRV+=(--max-model-len 9000)
    CLI=("$HERE/lc_lb_vllm.py" --model "$NAME" --hf "$HF" --label "$ARM" --port "$PORT"
         --tasks hotpotqa,2wikimqa,musique,passage_retrieval_en --start 0 --n "${LB_N:-200}"
         --max_len 8000);;
  ruler)
    SRV+=(--max-model-len "${MML:?set MML}" --no-enable-prefix-caching)
    T7=niah_single_2,niah_multikey_2,niah_multivalue,niah_multikey_3,niah_multiquery,vt,qa_1
    CLI=("$HERE/lc_ruler_vllm.py" --model "$NAME" --hf "$HF" --label "$ARM" --port "$PORT"
         --data_root "${RULER_DATA:?set RULER_DATA}" --length 32768
         --tasks "$T7" --n 50 --max_model_len "$MML");;
  *) echo "unknown bench $BENCH" >&2; exit 2;;
esac
# exact PYTHONPATH of the server: the start-up hook only (vLLM and the plugin are installed)
ENVS=(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" VLLM_USE_FLASHINFER_SAMPLER=0
      SPINTARE_SERVE=1 PYTHONPATH="$HERE/vllm_site" "${SW[@]}")
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "server: env ${ENVS[*]} $PY -m vllm.entrypoints.cli.main ${SRV[*]}"
  echo "client: $KVREL_PYTHON ${CLI[*]}"
  exit 0
fi
# Inherited VLLM_* / SPINTARE_* switches would change the arm: drop them before starting the server.
for v in $(env | sed -n 's/^\(\(VLLM\|SPINTARE\)_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
LOG=$KVREL_WORK/logs/vllm_${NAME}_${ARM}_$BENCH; mkdir -p "$(dirname "$LOG")"
setsid env -u LD_PRELOAD "${ENVS[@]}" "$PY" -m vllm.entrypoints.cli.main "${SRV[@]}" \
  > "$LOG.server.log" 2>&1 < /dev/null &
SPID=$!
stop() {  # the server runs in its own process group: TERM it, wait up to 120 s, then KILL
  kill -TERM -- "-$SPID" 2>/dev/null || true
  for _ in $(seq 1 60); do kill -0 "$SPID" 2>/dev/null || break; sleep 2; done
  kill -KILL -- "-$SPID" 2>/dev/null || true
  wait "$SPID" 2>/dev/null || true
}
trap stop EXIT
trap 'stop; exit 130' INT TERM
up=0
for _ in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && { up=1; break; }
  kill -0 "$SPID" 2>/dev/null || break
  sleep 5
done
[ $up = 1 ] || { echo "server did not come up, see $LOG.server.log" >&2; exit 1; }
for e in "${SW[@]}"; do
  tr '\0' '\n' < "/proc/$SPID/environ" | grep -qxF "$e" || { echo "missing $e" >&2; exit 1; }
done
"$KVREL_PYTHON" "${CLI[@]}" 2>&1 | tee "$LOG.client.log"
