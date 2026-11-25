#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# Build the llama.cpp engine trees used by the experiments: upstream llama.cpp at 3bb63cb, optionally with one of the
# patches in patches/llama.cpp/.
#
#   scripts/build_llama_cpp.sh [target] [cuda_arch]
#     target     spintare (default)  SpinTare repair: rotary add-back + exemption gate   -> $KVREL_LLAMA_CPP
#                base                upstream 3bb63cb, unpatched (engine labels)     -> $KVREL_LLAMA_CPP_BASE
#                postmean            static post-RoPE mean (LLAMA_KV_POSTMEAN): the static-centring baseline
#                                    and step 2 of `spintare check`                  -> $KVREL_LLAMA_CPP_PM
#                all                 spintare, base, postmean
#     cuda_arch  default: native  (e.g. 120 for RTX 5090 D / RTX PRO 5000)
#
# Each tree is a fresh clone owned by this script (marker .git/kvrel_tree). The script refuses to touch an existing
# directory without that marker, so it never resets a checkout of yours.
set -euo pipefail
. "$(cd "$(dirname "$0")" && pwd)/env.sh"
TARGET=${1:-spintare}
ARCH=${2:-native}
BASE=3bb63cb

build_one() {  # build_one <target>
  local t=$1 dir patch=""
  case "$t" in
    spintare) dir=$KVREL_LLAMA_CPP;      patch=spintare-rotary-addback-exemption-gate ;;
    base)     dir=$KVREL_LLAMA_CPP_BASE ;;
    postmean) dir=$KVREL_LLAMA_CPP_PM;   patch=postmean-static-mean ;;
    *) echo "unknown target $t" >&2; exit 2 ;;
  esac
  local pf=""
  if [ -n "$patch" ]; then
    pf="$KVREL_ROOT/patches/llama.cpp/$patch.patch"
    [ -f "$pf" ] || { echo "no patch $pf" >&2; exit 2; }
  fi
  if [ ! -e "$dir" ]; then
    git clone https://github.com/ggml-org/llama.cpp "$dir"
    echo "$t" > "$dir/.git/kvrel_tree"
  elif [ ! -f "$dir/.git/kvrel_tree" ]; then
    echo "$dir exists but was not created by this script; refusing to reset it (move it or set the path variable)" >&2
    exit 3
  fi
  (
    cd "$dir"
    git checkout -q "$BASE"
    git reset -q --hard "$BASE"
    git clean -fdq            # drop files added by a previous patch; build/ is git-ignored and kept
    if [ -n "$pf" ]; then git apply --check "$pf" && git apply "$pf"; fi
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
    cmake --build build -j "$(nproc)" --target llama-perplexity llama-bench llama-server llama-cli
  )
  echo "built $t at $dir (base $BASE${patch:+ + $patch})"
}

if [ "$TARGET" = all ]; then
  for t in spintare base postmean; do build_one "$t"; done
else
  build_one "$TARGET"
fi
