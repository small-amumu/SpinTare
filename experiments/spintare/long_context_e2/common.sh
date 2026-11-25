# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# Helpers for the E2/E3 drivers in this directory (the subset of the shared server helper file they used).
# Sourced after scripts/env.sh.
PY=${KVREL_PYTHON}
KB=${KVREL_WORK}
GG=${KVREL_GGUF}
B=${KVREL_LLAMA_CPP_BASE}/build/bin
mkdir -p "$KB/results/server"
utc(){ date -u +%FT%TZ; }
# run.log line: <UTC> <task> <model> <config> rc=<rc> <last line of script output>
runlog(){ echo "$(utc) $1 $2 $3 rc=$4 $5" >> "$KB/results/run.log"; }
