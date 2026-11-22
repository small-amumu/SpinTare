#!/usr/bin/env bash
# PRED-P probe runner = GATE-PRED a1 run_gp_probe.sh with the PRED-P paths (X, model list); otherwise unchanged. Per model of models_pp_probe.txt, in order: the engine K order and
# GGUF BOS flag (reuse PM-X results/<tag>/check_layout.out when present, else check_layout.py on the census GGUF;
# "-=<tag>" reuses that tag's result; "@<order>:<add_bos>" is a fixed layout read from the converter source and the
# GGUF header, for models check_layout.py cannot read), then kvprobe_unified.py (with the GGUF BOS id):
#   gpu models: the first of GPU0-GPU3 with >= need GB free (need = 2.2 GB per B parameters + 6 GB; the idle
#   capacity of all four GPUs may be used; GPU1 is skipped while the PM-X GPU runner is alive), polled every 2 min
#   (6 h cap per model, then FAILED); TF32 off.
#   cpu models: only after the PM-X chain logs A1_CHAIN_DONE (its CPU side needs the RAM), bf16 on CPU, 32 threads,
#   after MemAvailable >= 3 GB per B + 16 GB.
# Writes only ${KVREL_WORK}/predp/. One FAILED line per failed model; never touches another process.
. "$(cd "$(dirname "$0")/../../../.." && pwd)/scripts/env.sh"  # KVREL_* paths (see docs/setup.md)
set -uo pipefail
W=${KVREL_WORK}; X=$W/predp; C=$X/code; PY=${KVREL_PYTHON}; LT=${KVREL_LLAMA_CPP_KROT}
GG=${KVREL_GGUF}; PMX=$W/pmx/results; T=$W/data/long_text.txt; CAP=21600
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$X/run_probe.log"; }
exec 9> "$X/.lock_probe"; flock -n 9 || { echo "another PRED-P probe run holds the lock"; exit 2; }
export HF_HUB_OFFLINE=1 GGML_CUDA_CUBLAS_NO_TF32=1 NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 PYTHONHASHSEED=0
log "start; code md5 $(cd "$C" && md5sum kvprobe_unified.py run_pp_probe.sh models_pp_probe.txt | tr '\n' ' ')"
mkdir -p "$X/results"
size_b() { "$PY" -I -c 'import json,sys
c=json.load(open(sys.argv[1]+"/config.json"))
h=c.get("hidden_size") or c.get("d_model"); l=c.get("num_hidden_layers") or c.get("num_layers"); v=c.get("vocab_size",0)
i=c.get("intermediate_size") or 4*h
print(max(1, round((l*(4*h*h+3*h*i)+2*v*h)/1e9)))' "$1"; }
while read -r tag dir gguf dev; do
  [[ -z "$tag" || "$tag" == \#* ]] && continue
  R=$X/results/$tag; mkdir -p "$R"
  [[ -s "$R/probe.json" ]] && { log "$tag already probed"; continue; }
  fail() { log "$tag FAILED: $1"; echo "$1" > "$R/FAILED"; }
  HF=${KVREL_MODELS}/$dir
  [[ -f "$HF/config.json" ]] || { fail "no HF dir $HF"; continue; }
  bosid=-1
  if [[ "$gguf" == @* ]]; then
    fx=${gguf#@}
    echo "{\"order\": \"${fx%%:*}\", \"gguf_add_bos\": $([[ ${fx#*:} == 1 ]] && echo true || echo false)}" > "$R/layout.json"
    echo "fixed:$gguf" > "$R/layout_src"
  elif [[ "$gguf" == -=* ]]; then
    src=$X/results/${gguf#-=}/layout.json
    [[ -s "$src" ]] || { fail "base layout ${gguf#-=} missing"; continue; }
    cp "$src" "$R/layout.json"; echo "reused:${gguf#-=}" > "$R/layout_src"
  elif [[ -s "$PMX/$tag/check_layout.out" ]]; then
    tail -n 1 "$PMX/$tag/check_layout.out" > "$R/layout.json"; echo "pmx:$PMX/$tag/check_layout.out" > "$R/layout_src"
  else
    [[ -f "$GG/$gguf" ]] || { fail "no GGUF $GG/$gguf"; continue; }
    if ! "$PY" -I "$W/pmx/code/check_layout.py" "$GG/$gguf" "$HF" "$LT/gguf-py" > "$R/check_layout.out" 2> "$R/check_layout.err"; then
      fail "check_layout: $(tail -c 300 "$R/check_layout.out")"; continue
    fi
    tail -n 1 "$R/check_layout.out" > "$R/layout.json"; echo "gguf:$GG/$gguf" > "$R/layout_src"
  fi
  read -r order bos bosid < <("$PY" -I -c 'import json,sys
d=json.load(open(sys.argv[1])); b=d.get("gguf_bos_id")
print(d["order"], {True: 1, False: 0}.get(d.get("gguf_add_bos"), -1), -1 if b is None else int(b))' "$R/layout.json")
  [[ "$order" == hf || "$order" == gguf ]] || { fail "order $order"; continue; }
  if [[ "$bos" != 0 && "$bos" != 1 ]]; then
    ov=$W/pmx/add_bos_override/$tag
    [[ -s "$ov" ]] && read -r bos _ < "$ov"
    [[ "$bos" == 0 || "$bos" == 1 ]] || { fail "GGUF add_bos missing and no override"; continue; }
  fi
  # after the override, so that bos is final (review r32b M-1): a layout without gguf_bos_id is re-read from the GGUF
  if [[ "$bos" == 1 && "$bosid" == -1 && -f "$GG/$gguf" ]]; then
    if "$PY" -I "$W/pmx/code/check_layout.py" "$GG/$gguf" "$HF" "$LT/gguf-py" > "$R/check_layout_bos.out" \
        2> "$R/check_layout_bos.err"; then
      bosid=$("$PY" -I -c 'import json,sys; b=json.loads(open(sys.argv[1]).read().splitlines()[-1]).get("gguf_bos_id")
print(-1 if b is None else int(b))' "$R/check_layout_bos.out")
      [[ "$bosid" =~ ^-?[0-9]+$ ]] || bosid=-1
      log "$tag layout had no gguf_bos_id; re-read from the GGUF: $bosid"
    else
      log "$tag layout had no gguf_bos_id and the GGUF re-read failed"
    fi
  fi
  nb=$(size_b "$HF")
  if [[ "$dev" == gpu ]]; then
    need=$(( nb * 22 / 10 + 6 )); t1=$SECONDS; gpu=""
    while [[ -z "$gpu" ]]; do
      for g in 0 1 2 3; do
        [[ $g == 1 ]] && pgrep -f run_pmx_gpu.sh > /dev/null && continue
        f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" | tr -d ' ')
        (( f / 1024 >= need )) && { gpu=$g; break; }
      done
      [[ -n "$gpu" ]] && break
      (( SECONDS - t1 > CAP )) && break
      sleep 120
    done
    [[ -n "$gpu" ]] || { fail "no GPU0-3 with ${need} GB free in 6 h"; continue; }
    args=(--device cuda); envs=(CUDA_VISIBLE_DEVICES=$gpu)
  else
    t1=$SECONDS
    until grep -q A1_CHAIN_DONE "$W/pmx/a1_chain.log" 2>/dev/null; do
      (( SECONDS - t1 > 86400 )) && break
      sleep 120
    done
    grep -q A1_CHAIN_DONE "$W/pmx/a1_chain.log" 2>/dev/null || { fail "PM-X chain not done in 24 h"; continue; }
    need=$(( nb * 3 + 16 )); t1=$SECONDS
    until (( $(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo) >= need )); do
      (( SECONDS - t1 > CAP )) && break
      sleep 120
    done
    (( $(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo) >= need )) || { fail "MemAvailable < ${need} GB in 6 h"; continue; }
    args=(--device cpu); envs=(CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=32 MKL_NUM_THREADS=32); gpu=cpu
  fi
  log "$tag order $order bos $bos size ~${nb}B on ${gpu} ($(cat "$R/layout_src"))"
  if env "${envs[@]}" "$PY" -I "$C/kvprobe_unified.py" --model "$HF" --order "$order" --add-bos "$bos" --bos-id "$bosid" \
      --text "$T" --out "$R/probe.tmp.json" "${args[@]}" > "$R/probe.out" 2> "$R/probe.err"; then
    mv -f "$R/probe.tmp.json" "$R/probe.json"; rm -f "$R/FAILED"; log "$tag done: $(tail -n 1 "$R/probe.out" | cut -c1-200)"
  else
    fail "probe exit: $(tail -c 300 "$R/probe.err" | tr '\n' ' ')"
  fi
done < "$C/models_pp_probe.txt"
nf=$(ls "$X"/results/*/FAILED 2>/dev/null | wc -l)
log "GP_PROBE_DONE failed=$nf"
exit $(( nf > 0 ))
