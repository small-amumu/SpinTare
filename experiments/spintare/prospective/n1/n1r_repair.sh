#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# N1 repair of five left-out candidates (registered run notes "N1 repair": repair and re-test, and the repaired
# models must not fail again), before the N1 selection. Uses the hold of n1_host.sh 0d1599af after A2 (it
# waits up to 2 h for BOS overrides, re-running run_n1_probe.sh on every model without probe.json), so the repaired
# models enter the registered selection rule.
#  r1l_8b, r1q3_8b  convert with the census converter through n1r_convert.py (pre-tokenizer mapping for two hashes)
#                   to a STAGING name, token check (engine = HF of <dir>_tokfix), engine D with the A1 commands of
#                   n1_host.sh on GPU1 (idle memory only: need + 6 GB, the GPU1 load lock, a shared /tmp/pp_gpu1.lock)
#  probe dirs       <DeepSeek>_tokfix (tokenizer_class PreTrainedTokenizerFast: transformers 5.18 rebuilds a wrong
#                   pre-tokenizer from LlamaTokenizerFast + legacy), Falcon3-1B-Instruct_bos11 (bos_token
#                   <|endoftext|>, as Falcon3-3B-Instruct_bos11 of PRED-P), Phi-4-mini-instruct_native (no *.py, no
#                   auto_map: the repo's modeling_phi3.py fails under transformers 5.18) with fixed layout @hf:0
#  task dirs        SmolLM3-3B_tmpl, EXAONE-4.0-1.2B_tmpl, ERNIE-4.5-0.3B-PT_tmpl (official chat_template.jinja)
#  overrides        falcon3_1b 1, smollm3_3b 0 (llama-vocab.cpp)
# All dirs are made and checked in the pre-check; this script verifies them. The final GGUF names, both list edits,
# the FAILED removals and the overrides are applied together under run_n1_probe.sh's lock (review r48b M-2), so no
# probe pass sees a GGUF with an old list entry. Writes only ${KVREL_WORK}/n1/, ${KVREL_GGUF}/n1r_* and n1_<tag> of the
# two repaired models, and ${KVREL_WORK}/pmx/add_bos_override/<tag>.
. "$(cd "$(dirname "$0")/../../../.." && pwd)/scripts/env.sh"  # KVREL_* paths (see docs/setup.md)
set -uo pipefail
W=${KVREL_WORK}; X=$W/n1; C=$X/code; B=${KVREL_LLAMA_CPP_BASE}/build/bin; CONV=${KVREL_LLAMA_CPP_BASE}
PY=${KVREL_PYTHON}; G=${KVREL_GGUF}; TXT=$W/data/diag_text.txt; LOG=$X/run.log; OV=$W/pmx/add_bos_override
TOK=/tmp/n1r_tokbuild/bin/llama-tokenize; MD=${KVREL_MODELS}
export LD_LIBRARY_PATH=$B:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} HF_HUB_OFFLINE=1  # as n1_host.sh (r48 M-1)
log() { echo "$(date -u +%FT%TZ) [repair] $*" | tee -a $LOG; }
die() { log "$*"; exit 3; }
# ---------- guards (r48 M-2: only while the runner is before its selection)
[[ ! -e $X/selected.txt && ! -s $X/D_all.json ]] || { log "selection already started; repair not applied"; exit 2; }
pgrep -f "[n]1_host.sh" > /dev/null || { log "n1_host.sh not running; repair not applied"; exit 2; }
[[ $(git -C $CONV rev-parse --short HEAD) == 3bb63cb ]] || die "census converter is not 3bb63cb"
[[ $(md5sum < $B/llama-perplexity | cut -c1-8) == $(grep -o "llama-perplexity [0-9a-f]*" $LOG | head -1 | cut -d' ' -f2) ]] \
  || die "llama-perplexity md5 differs from the N1 start line"
[[ $(md5sum < $C/n1r_convert.py | cut -c1-8) == 180937b7 ]] || die "n1r_convert md5"
[[ $(md5sum < $C/tokcheck.py | cut -c1-8) == 9bfa181f ]] || die "tokcheck md5"
[[ -x $TOK ]] || die "no $TOK"
# r48b S-3: the premises established in the pre-check
for d in DeepSeek-R1-Distill-Llama-8B_tokfix DeepSeek-R1-0528-Qwen3-8B_tokfix; do
  $PY -I -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["tokenizer_class"] == "PreTrainedTokenizerFast" else 1)' \
    $MD/$d/tokenizer_config.json || die "$d: tokenizer_class"
done
$PY -I -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("bos_token") == "<|endoftext|>" else 1)' \
  $MD/Falcon3-1B-Instruct_bos11/tokenizer_config.json || die "Falcon3-1B-Instruct_bos11: bos_token"
[[ -z $(ls $MD/Phi-4-mini-instruct_native/*.py 2>/dev/null) ]] || die "Phi-4-mini-instruct_native has *.py"
$PY -I -c 'import json,sys; sys.exit(1 if "auto_map" in json.load(open(sys.argv[1])) else 0)' \
  $MD/Phi-4-mini-instruct_native/config.json || die "Phi-4-mini-instruct_native: auto_map"
for d in SmolLM3-3B_tmpl EXAONE-4.0-1.2B_tmpl ERNIE-4.5-0.3B-PT_tmpl; do
  [[ -s $MD/$d/chat_template.jinja && -s $MD/$d/tokenizer_config.json ]] || die "$d: chat template"
done
# r48c S-5: every variant dir resolves config.json (the B lanes read the native context from it)
for d in DeepSeek-R1-Distill-Llama-8B_tokfix DeepSeek-R1-0528-Qwen3-8B_tokfix Falcon3-1B-Instruct_bos11 \
         Phi-4-mini-instruct_native SmolLM3-3B_tmpl EXAONE-4.0-1.2B_tmpl ERNIE-4.5-0.3B-PT_tmpl; do
  [[ -s $MD/$d/config.json ]] || die "$d: no config.json"
done
# r48c S-4: the D of the three light repairs exists and is complete (select needs it)
for t in falcon3_1b smollm3_3b phi4mini; do
  $PY -I -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if all(d[c] is not None for c in ("q4_0","q4_0norot","q4_1")) else 1)' \
    $X/d/$t.json || die "$t: engine D missing or incomplete"
done
log "start; code $(md5sum < $C/n1r_repair.sh | cut -c1-8) $(md5sum < $C/n1r_convert.py | cut -c1-8)"
iters() { grep -c 'A2 waiting for BOS overrides' $LOG; }
CUT=20   # r48c S-1: past 20 of the 24 hold iterations, the heavy path is abandoned and the light repairs go in
# ---------- r1l_8b / r1q3_8b (heavy path): staging GGUF, token check, engine D (A1 commands); a failure leaves the
# model out of READY (registered rule: a failed candidate is left out) and never stops the light repairs
take1() {  # GPU1 load lock (fd 7) and shared pp lock (fd 6) with need + 6 GB free; gives up at the cut-off
  local f
  while (( $(iters) < CUT )); do
    exec 7> /tmp/kvt_gpu1_load.lock; flock 7
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')
    if [[ $f =~ ^[0-9]+$ ]] && (( f >= NEED + 6144 )); then
      exec 6> /tmp/pp_gpu1.lock; flock -s -n 6 && return 0; exec 6>&-
    fi
    exec 7>&-; sleep 60
  done
  return 1
}
heavy() {  # heavy <tag> <dir> <pre>; returns 0 only when the staging GGUF passed every check and D is complete
  local tag=$1 dir=$2 pre=$3 hf=$MD/$2 gs=$G/n1r_$1-q8_0.gguf gf=$G/n1_$1-q8_0.gguf got rc base g cfg k rot s
  [[ -e $gf ]] && { log "$tag: final GGUF $gf exists before the critical section; left out"; return 1; }
  if [[ ! -s $gs ]]; then
    (( $(iters) < CUT )) || { log "$tag: hold iterations past $CUT before conversion; left out"; return 1; }
    (cd $CONV && $PY $C/n1r_convert.py $CONV $hf --outtype q8_0 --outfile $gs.part > $X/d/${tag}_convert_r.log 2>&1 < /dev/null) \
      && mv $gs.part $gs
    [[ -s $gs ]] || { log "$tag FAILED: conversion $(tail -n 2 $X/d/${tag}_convert_r.log | tr '\n' ' ' | cut -c1-200)"; return 1; }
    log "$tag staging gguf md5 $(md5sum $gs | cut -c1-32) ($(grep -o 'gguf from [^ ]*' $X/d/${tag}_convert_r.log))"
  fi
  # pre check outside the conversion branch (r48c NIT 1), with the census gguf-py (r48 M-3: -I ignores PYTHONPATH)
  got=$($PY -I -c 'import sys; sys.path.insert(0, sys.argv[2] + "/gguf-py"); from gguf import GGUFReader
print(GGUFReader(sys.argv[1]).fields["tokenizer.ggml.pre"].contents())' $gs $CONV)
  [[ $got == $pre ]] || { log "$tag FAILED: tokenizer.ggml.pre '$got' != $pre"; mv -f $gs $gs.bad; return 1; }
  # token check: engine tokens of diag_text (llama-tokenize built CPU-only from census 3bb63cb; no BOS) must equal the
  # HF tokens of <dir>_tokfix; a tool failure leaves the model out but keeps the GGUF (r48b S-1)
  if [[ ! -s $X/d/${tag}_tokcheck.ok ]]; then
    $TOK -m $gs -f $TXT --ids --no-bos --log-disable > $X/d/${tag}_tok_engine.txt 2> $X/d/${tag}_tok_engine.err \
      || { log "$tag: llama-tokenize failed ($(tail -c 200 $X/d/${tag}_tok_engine.err | tr '\n' ' ')); left out, GGUF kept"; return 1; }
    grep -q '^\[' $X/d/${tag}_tok_engine.txt || { log "$tag: llama-tokenize wrote no id list; left out, GGUF kept"; return 1; }
    $PY -I $C/tokcheck.py ${hf}_tokfix $TXT $X/d/${tag}_tok_engine.txt > $X/d/${tag}_tokcheck.out 2>&1; rc=$?
    if (( rc == 0 )); then cp $X/d/${tag}_tokcheck.out $X/d/${tag}_tokcheck.ok
    elif grep -q "first mismatch" $X/d/${tag}_tokcheck.out; then
      log "$tag FAILED: engine tokens differ from HF ($(head -n 1 $X/d/${tag}_tokcheck.out))"; mv -f $gs $gs.bad; return 1
    else log "$tag: tokcheck.py failed ($(tail -n 1 $X/d/${tag}_tokcheck.out)); left out, GGUF kept"; return 1; fi
  fi
  log "$tag token check: $(head -n 1 $X/d/${tag}_tokcheck.ok)"
  if [[ ! -s $X/d/$tag.json ]]; then
    NEED=$(( $(stat -c %s $gs) / 1048576 + 6000 ))
    take1 || { log "$tag: hold iterations past $CUT while waiting for GPU1; left out"; return 1; }
    g=1; base=$X/d/$tag.kld
    run_ppl() {  # identical to n1_host.sh A1 (on the staging file: same bytes as the final GGUF)
      local env=(); [[ $2 == 0 ]] && env=(LLAMA_ATTN_ROT_DISABLE=1)
      env "${env[@]}" CUDA_VISIBLE_DEVICES=$g $B/llama-perplexity -m $gs -f $TXT -c 2048 --chunks 4 -b 2048 -fa on \
        -ctv q8_0 -ngl 99 -ctk $1 --kl-divergence-base $base ${4:-} 7>&- 6>&- > $3 2>&1 < /dev/null
    }
    run_ppl f16 1 $X/d/${tag}_f16.log
    exec 7>&-
    for s in q4_0:q4_0:1 q4_0norot:q4_0:0 q4_1:q4_1:1; do IFS=: read -r cfg k rot <<< "$s"; run_ppl $k $rot $X/d/${tag}_$cfg.log --kl-divergence; done
    exec 6>&-; rm -f $base
    $PY -I - "$X/d" "$tag" > $X/d/$tag.json.tmp <<'EOF'
import json, re, sys
from pathlib import Path
d, tag = Path(sys.argv[1]), sys.argv[2]
out = {}
for cfg in ("q4_0", "q4_0norot", "q4_1"):
    t = (d / f"{tag}_{cfg}.log").read_text(errors="replace")
    m = re.search(r"Mean\s+KLD:\s+(\S+)", t)
    try:
        x = float(m.group(1)) if m else None
    except ValueError:  # "-nan" and the like (review r38 S-3)
        x = None
    out[cfg] = x if x is not None and x == x else None
    out[cfg + "_rot_disabled_logged"] = "force disabled" in t
print(json.dumps(out))
EOF
    if $PY -I -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if all(d[c] is not None for c in ("q4_0","q4_0norot","q4_1")) else 1)' $X/d/$tag.json.tmp; then
      mv $X/d/$tag.json.tmp $X/d/$tag.json; log "A1 $tag D $(cat $X/d/$tag.json) gpu 1"
    else
      log "$tag FAILED: engine D has a null value $(cat $X/d/$tag.json.tmp)"; mv -f $X/d/$tag.json.tmp $X/d/$tag.json.bad; return 1
    fi
  fi
  return 0
}
READY=()
heavy r1l_8b DeepSeek-R1-Distill-Llama-8B llama-bpe && READY+=(r1l_8b)
heavy r1q3_8b DeepSeek-R1-0528-Qwen3-8B qwen2 && READY+=(r1q3_8b)
log "heavy path ready for: ${READY[*]:-none}"
# ---------- the two list edits on copies, checked before anything is applied (r48b S-2); the DeepSeek probe lines are
# changed only for models in READY (a left-out model keeps its original line and has no final GGUF)
P=$C/models_n1_probe.txt; M=$C/models_n1.txt
cp $P $X/.probe_list.new; cp $M $X/.task_list.new
sed -i -E 's#^(falcon3_1b +)Falcon3-1B-Instruct( +)#\1Falcon3-1B-Instruct_bos11\2#
  s#^(phi4mini +)Phi-4-mini-instruct( +)n1_phi4mini-q8_0\.gguf#\1Phi-4-mini-instruct_native\2@hf:0#' $X/.probe_list.new
want=2
for tag in ${READY[@]+"${READY[@]}"}; do
  case $tag in
    r1l_8b) sed -i -E 's#^(r1l_8b +)DeepSeek-R1-Distill-Llama-8B( +)#\1DeepSeek-R1-Distill-Llama-8B_tokfix\2#' $X/.probe_list.new ;;
    r1q3_8b) sed -i -E 's#^(r1q3_8b +)DeepSeek-R1-0528-Qwen3-8B( +)#\1DeepSeek-R1-0528-Qwen3-8B_tokfix\2#' $X/.probe_list.new ;;
  esac
  want=$(( want + 1 ))
done
sed -i -E 's#^(smollm3_3b \| )SmolLM3-3B$#\1SmolLM3-3B_tmpl#; s#^(exaone4_1b \| )EXAONE-4.0-1.2B$#\1EXAONE-4.0-1.2B_tmpl#
  s#^(ernie45_0p3b \| )ERNIE-4.5-0.3B-PT$#\1ERNIE-4.5-0.3B-PT_tmpl#' $X/.task_list.new
[[ $(diff $P $X/.probe_list.new | grep -c '^>') == $want && $(grep -cE "_bos11|_native +@hf:0|_tokfix" $X/.probe_list.new) == $want ]] \
  || die "probe list edit is not the expected $want lines"
[[ $(diff $M $X/.task_list.new | grep -c '^>') == 3 && $(grep -c "_tmpl$" $X/.task_list.new) == 3 ]] \
  || die "task list edit is not the expected 3 lines"
# ---------- critical section under run_n1_probe.sh's lock: final GGUF names, lists, FAILED removal, overrides, and the
# hold sentinel (r48c S-3)
H=$X/results/_n1r_hold
[[ ! -e $X/selected.txt && ! -s $X/D_all.json ]] || { log "selection started during the repair; nothing applied"; exit 2; }
exec 5> $X/.lock_probe; flock 5
[[ ! -e $X/selected.txt && ! -s $X/D_all.json ]] || { log "selection started during the repair; nothing applied"; exit 2; }
# the sentinel keeps the runner's BOS hold looping (its FAILED text contains add_bos; no override file has its name, so
# the loop never deletes it; it is in no model list) until every repaired model has a complete probe
mkdir -p $H; echo "n1r hold: waiting for the repaired probes (add_bos overrides written by n1r_repair.sh)" > $H/FAILED
OK=()
for tag in ${READY[@]+"${READY[@]}"}; do
  mv -n $G/n1r_$tag-q8_0.gguf $G/n1_$tag-q8_0.gguf
  if [[ -s $G/n1_$tag-q8_0.gguf && ! -e $G/n1r_$tag-q8_0.gguf ]]; then OK+=($tag); rm -f $X/results/$tag/FAILED   # r48c S-2
  else log "$tag: final rename failed; left out"; fi
done
[[ ${#OK[@]} == ${#READY[@]} ]] || die "a final rename failed; lists not applied (sentinel left: remove $H/FAILED by hand)"
[[ -s $P.orig_2a741c7c ]] || cp $P $P.orig_2a741c7c
[[ -s $M.orig_10c3b22d ]] || cp $M $M.orig_10c3b22d
mv -f $X/.probe_list.new $P; mv -f $X/.task_list.new $M
rm -f $X/results/phi4mini/FAILED
mkdir -p $OV
echo "1 llama-vocab.cpp (census): tokenizer.ggml.pre falcon3 -> add_bos = true; GGUF n1_falcon3_1b has no add_bos_token and no bos_token_id key, so the BPE default special_bos_id = 11 is used (as falcon3_3b); written 10-26 for N1 before its probe" > $OV/falcon3_1b
echo "0 llama-vocab.cpp (census): tokenizer.ggml.pre smaug-bpe sets no add_bos, the BPE default add_bos = false holds; GGUF n1_smollm3_3b has no add_bos_token key; written 10-26 for N1 before its probe" > $OV/smollm3_3b
exec 5>&-
grep -E "^(falcon3_1b|phi4mini|r1l_8b|r1q3_8b) " $P | tee -a $LOG
grep -E "^(smollm3_3b|exaone4_1b|ernie45_0p3b) " $M | tee -a $LOG
for tag in ${OK[@]+"${OK[@]}"}; do log "A1 $tag gguf md5 $(md5sum $G/n1_$tag-q8_0.gguf | cut -c1-32) (final name)"; done
log "repair applied (hold iterations used so far: $(iters) of 24); waiting for the probes of: glm4_9b falcon3_1b smollm3_3b phi4mini ${OK[*]:-}"
# ---------- release the sentinel when every repaired model has a complete probe, or at 22 hold iterations
complete() { $PY -I -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d["layers_captured"] == d["layers_config"] else 1)' "$1" 2>/dev/null; }
while true; do
  left=()
  for tag in glm4_9b falcon3_1b smollm3_3b phi4mini ${OK[@]+"${OK[@]}"}; do   # glm4_9b: r49 S-2
    complete $X/results/$tag/probe.json || left+=($tag)
  done
  (( ${#left[@]} == 0 )) && { log "all repaired probes complete; sentinel released"; break; }
  (( $(iters) >= 22 )) && { log "hold iterations at 22; sentinel released with probes missing: ${left[*]}"; break; }
  [[ -e $X/selected.txt ]] && { log "selection started; sentinel released"; break; }
  sleep 60
done
rm -f $H/FAILED; rmdir $H 2>/dev/null
log "repair done"
