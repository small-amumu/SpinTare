#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
# X5 runner (CPU only). Usage: bash x5_run.sh smoke|full
# smoke: real configs with 4 random-weight layers; full: the real models. Models run one after another.
set -uo pipefail
MODE=${1:?smoke or full}
X=$HOME/kvt_bench/x5
PY=$HOME/vllm_main/bin/python
M=$HOME/models
CAL=$HOME/kvt_bench/data/diag_text.txt
TXT=$HOME/kvt_bench/data/long_text.txt
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONHASHSEED=0
cd "$X" || exit 2
mkdir -p "out_$MODE"
log() { echo "$(date -u +%FT%TZ) $*" >> "$X/out_$MODE/run.log"; }
log "START $MODE md5 $(md5sum x5_addback_check.py ggml_ref.py x5_run.sh | tr '\n' ' ')"
EXTRA=""
[[ "$MODE" == "smoke" ]] && EXTRA="--random-layers 4 --seq 512 --nq 64"
while read -r tag dir order focus; do
  [[ -z "$tag" ]] && continue
  F=""
  [[ "$MODE" == "full" ]] && F="--focus $focus"
  log "BEGIN $tag"
  nice -n 10 "$PY" -X utf8 x5_addback_check.py --model "$M/$dir" --order "$order" \
    --calib "$CAL" --text "$TXT" --out "out_$MODE/x5_$tag.json" --threads 16 $EXTRA $F \
    > "out_$MODE/x5_$tag.log" 2>&1
  rc=$?
  log "END $tag rc=$rc"
done <<'LIST'
glm4_9b glm-4-9b-chat-hf hf 21,25
phi4mini Phi-4-mini-instruct hf 0,26
smollm3_3b SmolLM3-3B gguf 34,8
qwen25_1p5b Qwen2.5-1.5B-Instruct hf 0,1
LIST
log "DONE $MODE"
