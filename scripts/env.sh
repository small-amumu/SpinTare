# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# Environment for the SpinTare scripts. Source it (every experiment driver sources it itself):
#     . scripts/env.sh
# Every path can be overridden before sourcing, e.g. KVREL_MODELS=/data/hf . scripts/env.sh
# Nothing here is machine-specific: defaults are relative to the repository root.

KVREL_ROOT="${KVREL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
export KVREL_ROOT

# Working tree for logs, intermediate files and results (the role of the former per-server work directories).
export KVREL_WORK="${KVREL_WORK:-$KVREL_ROOT/work}"
# Hugging Face model directories (one sub-directory per model, named as in the experiment scripts).
export KVREL_MODELS="${KVREL_MODELS:-$KVREL_WORK/models}"
# GGUF files converted from those models (scripts/convert_gguf.sh).
export KVREL_GGUF="${KVREL_GGUF:-$KVREL_WORK/gguf}"
# Input texts and task data (long_text.txt, qasper_text.txt, mfqa_text.txt, LongBench, RULER).
export KVREL_DATA="${KVREL_DATA:-$KVREL_WORK/data}"
# Small result files and aggregates.
export KVREL_RESULTS="${KVREL_RESULTS:-$KVREL_ROOT/results}"
# llama.cpp trees (all upstream 3bb63cb; built by scripts/build_llama_cpp.sh <target>):
#   KVREL_LLAMA_CPP       + spintare-rotary-addback-exemption-gate.patch  (the SpinTare repair; target spintare)
#   KVREL_LLAMA_CPP_BASE  unpatched upstream: engine labels and GGUF conversion of the prospective sets (target base)
#   KVREL_LLAMA_CPP_KROT  the tree whose gguf-py / convert_hf_to_gguf.py the probes and GGUF conversions use. The
#                         original runs used a research tree with a learned-rotation patch (switch off); that patch
#                         touches no gguf-py or conversion file, so the SpinTare tree is equivalent and is the default.
export KVREL_LLAMA_CPP="${KVREL_LLAMA_CPP:-$KVREL_WORK/llama.cpp}"
export KVREL_LLAMA_CPP_BASE="${KVREL_LLAMA_CPP_BASE:-$KVREL_WORK/llama.cpp-base}"
export KVREL_LLAMA_CPP_KROT="${KVREL_LLAMA_CPP_KROT:-$KVREL_LLAMA_CPP}"
#   KVREL_LLAMA_CPP_PM    + postmean-static-mean.patch: static post-RoPE mean (LLAMA_KV_POSTMEAN), the static-centring
#                         baseline and step 2 of `spintare check` (target postmean)
export KVREL_LLAMA_CPP_PM="${KVREL_LLAMA_CPP_PM:-$KVREL_WORK/llama.cpp-pm}"
# Python interpreter / virtual environment that has torch, transformers (and vllm for the vLLM experiments).
export KVREL_PYTHON="${KVREL_PYTHON:-python3}"
export KVREL_VENV="${KVREL_VENV:-$(dirname "$(dirname "$(command -v "$KVREL_PYTHON" 2>/dev/null || echo /usr/bin/python3)")")}"
# Optional extra Python path and library prefix (some vLLM builds need a newer libstdc++: KVREL_ENV_PREFIX/lib).
export KVREL_PYLIB="${KVREL_PYLIB:-}"
export KVREL_ENV_PREFIX="${KVREL_ENV_PREFIX:-}"
# LD_LIBRARY_PATH with $KVREL_ENV_PREFIX/lib in front when the prefix is set (drivers: export LD_LIBRARY_PATH=$KVREL_LD_PATH)
export KVREL_LD_PATH="${KVREL_ENV_PREFIX:+$KVREL_ENV_PREFIX/lib${LD_LIBRARY_PATH:+:}}${LD_LIBRARY_PATH:-}"

export PYTHONPATH="$KVREL_ROOT${KVREL_PYLIB:+:$KVREL_PYLIB}${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$KVREL_WORK" "$KVREL_RESULTS" 2>/dev/null || true

