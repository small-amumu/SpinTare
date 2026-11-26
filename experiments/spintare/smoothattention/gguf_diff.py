# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""CH4Z gate G1: tensor-level comparison of an original GGUF and its SmoothAttention-folded GGUF.

Every tensor outside the folded set must have the same type, shape and bytes; every tensor in the
folded set must differ in bytes (same type and shape). Folded set: qwen2
blk.<i>.attn_q.{weight,bias}, blk.<i>.attn_k.{weight,bias}; qwen3 blk.<i>.attn_q_norm.weight,
blk.<i>.attn_k_norm.weight. Metadata keys whose values differ are listed. Prints a JSON summary;
exit 0 only if the tensor conditions hold and the folded count is n_layer x 4 (qwen2) / x 2. usage:
PYTHONPATH=<llama.cpp>/gguf-py python gguf_diff.py <orig.gguf> <sa.gguf> <qwen2|qwen3> <out.json>
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from gguf import GGUFReader

PAT = {
    "qwen2": re.compile(r"^blk\.\d+\.attn_[qk]\.(weight|bias)$"),
    "qwen3": re.compile(r"^blk\.\d+\.attn_[qk]_norm\.weight$"),
}
PER_LAYER = {"qwen2": 4, "qwen3": 2}


def field_value(f) -> str:
    return hashlib.md5(b"".join(bytes(p) for p in f.parts)).hexdigest()


def main() -> int:
    a, b, arch, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    ra, rb = GGUFReader(a), GGUFReader(b)
    ta = {t.name: t for t in ra.tensors}
    tb = {t.name: t for t in rb.tensors}
    res: dict = {"orig": a, "sa": b, "arch": arch, "n_tensors": [len(ta), len(tb)]}
    bad: list[str] = []
    if set(ta) != set(tb):
        bad.append(
            f"tensor names differ: only orig {sorted(set(ta) - set(tb))[:5]}, only sa {sorted(set(tb) - set(ta))[:5]}"
        )
    folded = same = 0
    for name in sorted(set(ta) & set(tb)):
        x, y = ta[name], tb[name]
        if x.tensor_type != y.tensor_type or list(x.shape) != list(y.shape):
            bad.append(f"{name}: type or shape differs")
            continue
        eq = bytes(x.data.tobytes()) == bytes(y.data.tobytes())
        if PAT[arch].match(name):
            folded += 1
            if eq:
                bad.append(f"{name}: folded tensor unchanged")
        else:
            same += 1
            if not eq:
                bad.append(f"{name}: non-folded tensor changed")
    n_layer = len({m.group(0) for m in (re.match(r"^blk\.\d+\.", n) for n in ta) if m})
    bc = ra.fields.get(f"{arch}.block_count")
    if bc is None or int(bc.parts[bc.data[0]][0]) != n_layer:
        bad.append(f"n_layer {n_layer} from tensor names does not match {arch}.block_count")
    if folded != n_layer * PER_LAYER[arch]:
        bad.append(f"folded tensors {folded} != n_layer {n_layer} x {PER_LAYER[arch]}")
    fa, fb = ra.fields, rb.fields
    meta_diff = sorted(
        k
        for k in set(fa) | set(fb)
        if k not in fa or k not in fb or field_value(fa[k]) != field_value(fb[k])
    )
    res.update(
        n_layer=n_layer,
        folded_changed=folded,
        other_identical=same,
        metadata_keys_differing=meta_diff,
        problems=bad[:20],
        n_problems=len(bad),
        passed=not bad,
    )
    Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(
        "GGUFDIFF",
        "ok" if not bad else "FAIL",
        json.dumps(
            {
                k: res[k]
                for k in (
                    "n_layer",
                    "folded_changed",
                    "other_identical",
                    "metadata_keys_differing",
                    "n_problems",
                )
            }
        ),
        flush=True,
    )
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
