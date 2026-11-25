#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# One llama.cpp task cell of the main task results and of the gate ablations
# (LongBench and RULER with / without the gate): start llama-server for one K-cache arm, run the task client, stop the server.
#
# usage: run_llamacpp_cell.sh <bench> <arm> <name>
#   bench  longbench | ruler | gsm8k | math500
#   arm    f16       16-bit K cache
#          plain     q4_0 K, no repair
#          pm        q4_0 K, static post-RoPE mean (postmean tree, LLAMA_KV_POSTMEAN=$PM)
#          sa        q4_0 K, SmoothAttention (GGUF = folded model, see ../smoothattention/)
#          spintare  q4_0 K, rotary centring with exact add-back + exemption gate
#          nogate    q4_0 K, rotary centring with exact add-back, exemption gate off
#   name   model label; results go to $KVREL_WORK/results/{lc,ruler}/<name>/ or $KVREL_WORK/reason/
# environment (after scripts/env.sh):
#   GGUF   q8_0 GGUF of the model (the tree's convert_hf_to_gguf.py --outtype q8_0)
#   HF     Hugging Face model directory (tokenizer and chat template for the client)
#   MU     pre-RoPE mean file for spintare / nogate (../calib_and_diag/mu_calib.py)
#   PM     static post-RoPE mean file for pm (python -m spintare calib, 2048 tokens)
#   PORT   server port (default 8090); GPU: CUDA device in PCI bus order (default 0)
#   data   LongBench: $KVREL_WORK/data/data/<task>.jsonl (scripts/get_longbench.sh);
#          RULER_DATA: RULER prepare.py output for this tokenizer,
#          <root>/32768/<task>/validation.jsonl;
#          reasoning: KVT_REASON_ROOT (default $KVREL_WORK/reason) with gsm8k/main/test-*.parquet
#          and math500/*.jsonl
#   R_START, R_N  item range of a reasoning cell (default 0 and 200 / 500)
#   DRY_RUN=1 prints the server and client commands without running them.
# Settings of the registered runs: V q8_0 and flash attention in every arm, one engine build and
# GPU per table block. LongBench: 8K prompts (max_len 8000), greedy; multifieldqa_en has 150 items,
# the other tasks 200. RULER: 32K, 4 tasks x 50 items, greedy. GSM8K items 0-199 and MATH-500 items
# 0-499 (the registered MATH-500 cells ran as two ranges, 0-249 and 250-499): T 0.6, top-p 0.95,
# seed 1000 + item, at most 8192 new tokens. GGML_CUDA_PDL=0 in every arm (with PDL on, 32K prefill
# timed out on Blackwell; the four-task LongBench cells ran with the default).
# Needs Linux (/proc, setsid-free process handling), bash >= 4.4 and curl.
set -euo pipefail
[ $# -eq 3 ] || { sed -n 2,33p "$0"; exit 2; }
BENCH=$1; ARM=$2; NAME=$3
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../../../scripts/env.sh"
: "${GGUF:?set GGUF}"; : "${HF:?set HF}"
PORT=${PORT:-8090}; GPU=${GPU:-0}
BIN=$KVREL_LLAMA_CPP/build/bin; KT=q4_0; SW=()
case $ARM in
  f16)      KT=f16;;
  plain|sa) ;;
  pm)       BIN=$KVREL_LLAMA_CPP_PM/build/bin; SW=("LLAMA_KV_POSTMEAN=${PM:?set PM}");;
  spintare) SW=("LLAMA_KV_DEBIAS_MU=${MU:?set MU}" LLAMA_KV_DEBIAS_MU_PATH=kqpos
                LLAMA_KV_MU_GATE=1);;
  nogate)   SW=("LLAMA_KV_DEBIAS_MU=${MU:?set MU}" LLAMA_KV_DEBIAS_MU_PATH=kqpos
                LLAMA_KV_MU_GATE=0);;
  *) echo "unknown arm $ARM" >&2; exit 2;;
esac
SRV=(-m "$GGUF" -fa on -ctk "$KT" -ctv q8_0 -ngl 99 --host 127.0.0.1 --port "$PORT")
NCTX=0
case $BENCH in
  longbench)
    SRV+=(-c 9000 -np 1)
    T6=hotpotqa,2wikimqa,musique,passage_retrieval_en,multifieldqa_en,qasper
    CLI=("$HERE/lc_lb.py" --model "$NAME" --hf "$HF" --ctk "$ARM" --max_len 8000 --port "$PORT"
         --tasks "${LB_TASKS:-$T6}" --n "${LB_N:-200}");;
  ruler)
    NCTX=32768; SRV+=(-c $((32768 + 1024)) -np 1)
    CLI=("$HERE/lc_ruler.py" --model "$NAME" --hf "$HF" --ctk "$ARM" --port "$PORT"
         --data_root "${RULER_DATA:?set RULER_DATA}" --lengths 32768
         --tasks niah_multikey_3,niah_multiquery,vt,qa_1 --n 50 --par 1);;
  gsm8k|math500)
    SRV+=(-np 4 -c 34816 -kvu --kv-unified-per-slot 8704)
    N=200; [ "$BENCH" = math500 ] && N=500
    START=${R_START:-0}; N=${R_N:-$N}
    CLI=("$HERE/lc_reason.py" --ctk "${NAME}_$ARM" --port "$PORT" --data "$BENCH"
         --start "$START" --n "$N" --max_new 8192 --par 4
         --journal "$KVREL_WORK/journal/${NAME}_${ARM}_${BENCH}_s$START.jsonl");;
  *) echo "unknown bench $BENCH" >&2; exit 2;;
esac
ENVS=(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" LD_LIBRARY_PATH="$BIN"
      GGML_CUDA_PDL=0 "${SW[@]}")
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "server: env ${ENVS[*]} $BIN/llama-server ${SRV[*]}"
  echo "client: $KVREL_PYTHON ${CLI[*]}"
  exit 0
fi
# Inherited LLAMA_* / GGML_* switches would change the arm: drop them before starting the server.
for v in $(env | sed -n 's/^\(\(LLAMA\|GGML\)_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
mkdir -p "$KVREL_WORK/journal"
LOG=$KVREL_WORK/logs/${NAME}_${ARM}_$BENCH; mkdir -p "$(dirname "$LOG")"
env "${ENVS[@]}" "$BIN/llama-server" "${SRV[@]}" > "$LOG.server.log" 2>&1 < /dev/null &
SPID=$!
stop() { kill "$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true; }
trap stop EXIT
trap 'stop; exit 130' INT TERM
up=0
for _ in $(seq 1 600); do
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && { up=1; break; }
  kill -0 "$SPID" 2>/dev/null || break
  sleep 2
done
[ $up = 1 ] || { echo "server did not come up, see $LOG.server.log" >&2; exit 1; }
# The engine reads its switches once at start-up: check that they reached the server process.
for e in "${SW[@]}"; do
  tr '\0' '\n' < "/proc/$SPID/environ" | grep -qxF "$e" || { echo "missing $e" >&2; exit 1; }
done
if [ "$NCTX" -gt 0 ]; then  # the server caps the slot at the model's training context
  ncs=$(sed -n 's/.*n_ctx_slot = \([0-9]*\).*/\1/p' "$LOG.server.log" | head -n 1)
  [ -n "$ncs" ] && [ "$ncs" -ge "$NCTX" ] || { echo "n_ctx_slot ${ncs:-?} < $NCTX" >&2; exit 1; }
fi
"$KVREL_PYTHON" "${CLI[@]}" 2>&1 | tee "$LOG.client.log"
