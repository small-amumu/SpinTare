#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
# Download the LongBench (v1) English subtasks used by `spintare longbench` into DATA_DIR
# (default ./data/longbench). The download is pinned to one revision of the dataset repository and
# checked against its sha256. The four default tasks of `spintare longbench` are hotpotqa, 2wikimqa,
# musique and passage_retrieval_en; multifieldqa_en and qasper are extracted too and can be
# selected with --tasks.
set -euo pipefail
DATA_DIR="${1:-data/longbench}"
REV="5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
SHA256="cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"   # set HF_ENDPOINT to use a mirror
URL="${HF_ENDPOINT}/datasets/zai-org/LongBench/resolve/${REV}/data.zip"
PY="$(command -v python3 || command -v python)"
mkdir -p "$DATA_DIR"
curl -L --fail -o "$DATA_DIR/data.zip" "$URL"
"$PY" - "$DATA_DIR" "$SHA256" <<'PY'
import hashlib, sys, zipfile, pathlib
d, want_sha = pathlib.Path(sys.argv[1]), sys.argv[2]
h = hashlib.sha256((d / "data.zip").read_bytes()).hexdigest()
if h != want_sha:
    sys.exit(f"data.zip sha256 {h} does not match the pinned {want_sha}")
want = {"hotpotqa", "2wikimqa", "musique", "passage_retrieval_en", "multifieldqa_en", "qasper"}
with zipfile.ZipFile(d / "data.zip") as z:
    for n in z.namelist():
        p = pathlib.PurePosixPath(n)
        if p.suffix == ".jsonl" and p.stem in want:
            (d / p.name).write_bytes(z.read(n))
(d / "data.zip").unlink()
print("LongBench tasks in", d, sorted(x.name for x in d.glob("*.jsonl")))
PY
