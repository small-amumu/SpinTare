#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
# PM-X GGUF preparation (no measurement): fresh q8_0 conversions of the HF checkpoints by the llama.cpp-krot-llama
# tree's convert_hf_to_gguf.py into ${KVREL_WORK}/pmllama/gguf/<tag>-q8_0.gguf, one log per tag (the runner requires its
# "Model successfully exported" line). CPU only; skips a tag whose log already has that line.
. "$(cd "$(dirname "$0")/../../.." && pwd)/scripts/env.sh"  # KVREL_* paths (see docs/setup.md)
set -uo pipefail
P=${KVREL_WORK}/pmllama/gguf; LT=${KVREL_LLAMA_CPP_KROT}; PY=${KVREL_PYTHON}
mkdir -p "$P"
while read -r tag dir; do
  [[ -z "$tag" || "$tag" == \#* ]] && continue
  lg=$P/conv_$tag.log; out=$P/$tag-q8_0.gguf
  if grep -q "Model successfully exported to $out" "$lg" 2>/dev/null; then echo "$tag already converted"; continue; fi
  if (cd "$LT" && "$PY" convert_hf_to_gguf.py "${KVREL_MODELS}/$dir" --outtype q8_0 --outfile "$out") > "$lg" 2>&1; then
    echo "$(date -u +%FT%TZ) $tag converted"
  else
    echo "$(date -u +%FT%TZ) $tag FAILED (see $lg)"; rm -f "$out"
  fi
done < "$(dirname "$0")/models_pmx.txt"
echo "CONVERT_ALL_DONE $(date -u +%FT%TZ)"
