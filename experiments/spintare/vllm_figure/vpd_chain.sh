#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# The run behind the vLLM figure (all panels, one RTX PRO 5000, 2026-11-23). Frozen output and the
# analyzer report: experiments/results/spintare/vllm_figure/ (vpd_result.json, vpd_analysis.txt).
#   stage 0  tree vllm_59674 = vllm_plain + pr59674_applied.diff (PR #59674 on vLLM 5cb0fd265d;
#            md5 of the four patched files checked)
#   stage 1  output KL (spintare_vllm.kld: bf16 reference, CTX 2048, 4 chunks, eager) on
#            Qwen2.5-1.5B, Qwen2.5-7B, Llama-3.1-8B: ref, int4_stock, int4_59674, int4_ab_v1f,
#            fp8_stock, fp8_ab_v1f
#   stage 2  eager torch-profiled pass (Llama-3.1-8B, 128 requests 512 / 64): stock int4 / fp8,
#            add-back v0 + torch write, add-back v1 (tf32x3) + fused write
#   stage 3  CUDA-graph throughput (Llama-3.1-8B, 1000 requests 512 / 256): the same six arms,
#            3 repeats round robin, then a stock int4 re-time
# INT4 arms run with SPINTARE_NO_HADACORE=1 (vLLM's hadacore kernel is not usable on that GPU).
# Layout under $VPD_ROOT: vllm_plain (vLLM 5cb0fd265d source tree), vllm_ab2 (the same tree with
# integrations/patches/vllm/spintare-addback.patch), site4 (the spintare_vllm plugin, integrations/vllm/src),
# venv312 (Python with torch and vLLM's compiled parts), data/diag_text.txt (md5 4c8f1ecf),
# results/vp2/<model>/m.kvmr (pre-RoPE means from `python -m spintare_vllm calib-mu`),
# runs/vpd/ (this directory). Models in $KVREL_MODELS. Resumable: a run whose JSON exists is kept.
# usage: VPD_ROOT=<dir> bash vpd_chain.sh <gpu>
set -u
GPU=${1:?gpu}
V=${VPD_ROOT:?set VPD_ROOT}
HOME_MODELS=${KVREL_MODELS:?set KVREL_MODELS}
OUT=$V/results/vpd
PY=$V/venv312/bin/python
RD=$V/runs/vpd
exec 9>> "$RD/.lock"
flock -n 9 || { echo "VPD: another chain holds $RD/.lock"; exit 3; }
mkdir -p "$V/gpu_queue"
exec 8>> "$V/gpu_queue/gpu$GPU.lock"
echo "VPD waiting for gpu$GPU.lock $(date -u +%FT%TZ)"
flock -w 43200 8 || { echo "VPD: gpu$GPU.lock not free within 12 h"; exit 4; }
busy() {  # number of compute processes on the card, or "err"
  local u out
  u=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader) || { echo err; return; }
  out=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader) || { echo err; return; }
  printf '%s\n' "$out" | grep -c "$u"
}
# wait up to 5 min for the card to be free (a run that just ended may still be listed)
wait_free() {
  local n
  for _ in $(seq 1 30); do
    n=$(busy)
    [ "$n" = 0 ] && return 0
    sleep 10
  done
  echo "VPD STOP $1: GPU$GPU has $n compute processes"
  exit 1
}
wait_free "at start"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_COMPILE_CACHE=1 FLASHINFER_WORKSPACE_BASE=$V/fi_base
for v in $(env | sed -n 's/^\(\(SPINTARE\|VLLM_KV\)_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
unset LD_PRELOAD
# preflight: inputs present and pinned (mean files and the PR diff as used for the reported run)
pin() {  # pin <file> <md5 prefix>
  local m
  [ -s "$1" ] || { echo "VPD VOID: missing $1"; exit 1; }
  m=$(md5sum < "$1" | cut -c1-${#2})
  [ "$m" = "$2" ] || { echo "VPD VOID: md5 $1 = $m, expected $2"; exit 1; }
}
pin "$V/results/vp2/Qwen2.5-1.5B-Instruct/m.kvmr" 12b989f5
pin "$V/results/vp2/Qwen2.5-7B-Instruct/m.kvmr" d2df7510
pin "$V/results/vp2/Meta-Llama-3.1-8B-Instruct/m.kvmr" afcd070e
pin "$V/data/diag_text.txt" 4c8f1ecf
pin "$RD/pr59674_applied.diff" 4620da88
for M in Qwen2.5-1.5B-Instruct Qwen2.5-7B-Instruct Meta-Llama-3.1-8B-Instruct; do
  [ -s "$HOME_MODELS/$M/config.json" ] || { echo "VPD VOID: missing model $M"; exit 1; }
done
SITE=$V/site4
NH="SPINTARE_NO_HADACORE=1"
V1F="SPINTARE_AB_VARIANT=1 SPINTARE_AB_WRITE=fused"
V0T="SPINTARE_AB_VARIANT=0 SPINTARE_AB_WRITE=torch"
cards() {
  local q=index,memory.used,utilization.gpu,clocks.sm,temperature.gpu
  echo "cards=[$(nvidia-smi --query-gpu=$q \
    --format=csv,noheader | tr '\n' ' ')]"
}
mkdir -p "$OUT/traces"
echo "VPD START $(date -u +%FT%TZ) gpu=$GPU $(nvidia-smi -i "$GPU" \
  --query-gpu=name,uuid,driver_version --format=csv,noheader)"
echo "VPD md5 $(md5sum "$RD/vpd_chain.sh" "$RD/vpd_analyze.py" \
  "$SITE"/spintare_vllm/{kld,pf_bench,addback}.py \
  | cut -c1-8 | tr '\n' ' ')"

# ---- stage 0: PR #59674 tree ----
PRT=$V/vllm_59674
F4="vllm/model_executor/models/qwen2.py:c0ec4cc6
vllm/v1/attention/backends/triton_attn.py:291ff2fc
vllm/v1/attention/ops/int4_per_token_head.py:99fe9bde
vllm/v1/attention/ops/triton_unified_attention.py:08106e9c"
if [ ! -d "$PRT" ]; then
  rm -rf "$PRT.tmp"
  cp -a "$V/vllm_plain" "$PRT.tmp" 9>&- 8>&- \
    && (cd "$PRT.tmp" && patch -p1 --dry-run -s < "$RD/pr59674_applied.diff" > /dev/null \
      && patch -p1 -s < "$RD/pr59674_applied.diff") \
    && mv "$PRT.tmp" "$PRT" || { echo "VPD VOID: PR tree not built"; exit 1; }
fi
for fm in $F4; do
  m=$(md5sum < "$PRT/${fm%%:*}" | cut -c1-8)
  [ "$m" = "${fm##*:}" ] ||
    { echo "VPD VOID: $PRT/${fm%%:*} md5 $m != expected ${fm##*:}"; exit 1; }
done
echo "stage0 PR tree ok"

# ---- stage 1: output KL ----
for M in Qwen2.5-1.5B-Instruct Qwen2.5-7B-Instruct Meta-Llama-3.1-8B-Instruct; do
  MD=$HOME_MODELS/$M
  D=$OUT/kl/$M
  mkdir -p "$D"
  MU="SPINTARE_KVMU=$V/results/vp2/$M/m.kvmr SPINTARE_GATE=1"
  arm() {  # arm TAG TREE ENVSPEC kld-args...
    local tag=$1 tree=$2 envs=$3 rc
    shift 3
    [ "$envs" = "-" ] && envs=""
    if [ -s "$D/$tag.json" ]; then  # kld writes its JSON last, after every ref chunk
      echo "$M $tag rc=0 (kept from an earlier start)"
      return
    fi
    wait_free "before $M $tag"
    env PYTHONPATH="$SITE:$V/$tree" VLLM_CACHE_ROOT="$V/cache_vpd/kl_$tag" $envs \
      "$PY" -m spintare_vllm.kld --model "$MD" --refdir "$D/ref" --text "$V/data/diag_text.txt" \
      --out "$D/$tag.json" --util 0.85 "$@" > "$D/$tag.log" 2>&1 9>&- 8>&-
    rc=$?
    echo "$M $tag rc=$rc $(cards)"
  }
  arm ref vllm_ab2 - --kv auto --backend TRITON_ATTN --ref
  arm int4_stock vllm_plain "$NH" --kv int4_per_token_head --backend TRITON_ATTN
  arm int4_59674 vllm_59674 "$NH" --kv int4_per_token_head --backend TRITON_ATTN
  arm int4_ab_v1f vllm_ab2 "$NH $MU $V1F" --kv int4_per_token_head --backend SPINTARE_AB
  arm fp8_stock vllm_plain - --kv fp8_per_token_head --backend TRITON_ATTN
  arm fp8_ab_v1f vllm_ab2 "$MU $V1F" --kv fp8_per_token_head --backend SPINTARE_AB
done

# ---- stages 2 and 3: Llama-3.1-8B timing ----
TM=$HOME_MODELS/Meta-Llama-3.1-8B-Instruct
TMU=$V/results/vp2/Meta-Llama-3.1-8B-Instruct/m.kvmr
pf() {  # pf TAG TREE ARM MODE envs...   MODE = prof | tp
  local tag=$1 tree=$2 arm=$3 mode=$4 extra=() rc
  shift 4
  [ -s "$OUT/$tag.json" ] && { echo "$tag rc=0 (kept from an earlier start)"; return; }
  wait_free "before $tag"
  case $arm in *_ab) extra+=(--mu "$TMU") ;; esac
  if [ "$mode" = prof ]; then
    rm -rf "$OUT/traces/$tag"  # a trace left by an earlier, unfinished attempt must not be read
    extra+=(--n 16 --eager --profile-dir "$OUT/traces/$tag" --profile-n 128 --profile-out-len 64)
  else
    extra+=(--n 1000)
  fi
  env PYTHONPATH="$SITE:$V/$tree" VLLM_CACHE_ROOT="$V/cache_vpd/$tag" "$@" \
    "$PY" -m spintare_vllm.pf_bench \
    --model "$TM" --arm "$arm" --in-len 512 --out-len 256 "${extra[@]}" --out "$OUT/$tag.json" \
    > "$OUT/$tag.log" 2> "$OUT/$tag.err" 9>&- 8>&-
  rc=$?
  echo "$tag rc=$rc $(cards)"
}
pf prof_int4 vllm_plain int4 prof $NH
pf prof_fp8 vllm_plain fp8 prof
pf prof_int4_ab_v0t vllm_ab2 int4_ab prof $NH SPINTARE_GATE=1 $V0T
pf prof_int4_ab_v1f vllm_ab2 int4_ab prof $NH SPINTARE_GATE=1 $V1F
pf prof_fp8_ab_v0t vllm_ab2 fp8_ab prof SPINTARE_GATE=1 $V0T
pf prof_fp8_ab_v1f vllm_ab2 fp8_ab prof SPINTARE_GATE=1 $V1F
for r in 1 2 3; do
  pf "tp_int4_r$r" vllm_plain int4 tp $NH
  pf "tp_fp8_r$r" vllm_plain fp8 tp
  pf "tp_int4_ab_v0t_r$r" vllm_ab2 int4_ab tp $NH SPINTARE_GATE=1 $V0T
  pf "tp_int4_ab_v1f_r$r" vllm_ab2 int4_ab tp $NH SPINTARE_GATE=1 $V1F
  pf "tp_fp8_ab_v0t_r$r" vllm_ab2 fp8_ab tp SPINTARE_GATE=1 $V0T
  pf "tp_fp8_ab_v1f_r$r" vllm_ab2 fp8_ab tp SPINTARE_GATE=1 $V1F
done
pf tp_int4_end vllm_plain int4 tp $NH
"$PY" "$RD/vpd_analyze.py" "$OUT" > "$OUT/vpd_analysis.txt" 2>&1 9>&- 8>&-
rc=$?
echo "VPD CHAIN DONE analyze rc=$rc $(date -u +%FT%TZ)"
