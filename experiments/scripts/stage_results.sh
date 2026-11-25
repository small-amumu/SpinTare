#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# Link the shipped frozen results into $KVREL_WORK under the directory names the aggregators were written for
# (gatepred, predp, n1, n4, n13, lc1), so that e.g.
#   python experiments/analysis/spintare/z_analyses.py
# run without a GPU. Links only; nothing is copied or modified.
set -euo pipefail
. "$(cd "$(dirname "$0")" && pwd)/env.sh"
R="$KVREL_RESULTS/spintare"
mkdir -p "$KVREL_WORK"
ln -sfn "$R/detection_gate" "$KVREL_WORK/gatepred"
for s in predp n1 n4 n13; do ln -sfn "$R/prospective/$s" "$KVREL_WORK/$s"; done
mkdir -p "$KVREL_WORK/lc1"
ln -sfn "$R/long_context_lc1/res_lc1_agg.json" "$KVREL_WORK/lc1/res_lc1_agg.json"
[ -e "$KVREL_WORK/data" ] || ln -sfn "$KVREL_ROOT/experiments/data" "$KVREL_WORK/data"
echo "staged results under $KVREL_WORK"
