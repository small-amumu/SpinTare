# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""PRED-P amendment 1 scoring: the natural-only thresholds (P_top2n, P_klmn) on the PRED-P cells,
prospective cells separated from the others (prospective = no task summary when amendment 1 was
appended, 04:05:15Z)."""

import os
import json
from pathlib import Path

X = Path(os.environ.get("KVREL_WORK", "work")) / "predp"
dec = json.load(open(X / "decisions.json"))
lab = json.load(open(X / "labels.json", encoding="utf-8"))
T2N, KLN = 2.0280476994784027, 0.05584895683245526
existing = {l.strip() for l in open(X / "labels_existing_at_A1.txt") if l.strip()}
early = {
    "qwen2_0p5b",
    "qwen2_1p5b",
    "coder_3b",
    "qwen3_1p7b_base",
}  # all their summaries existed at 04:05:15Z
Y = {0: "lossless", 1: "partial", 2: "collapse", None: "-"}
rows = []
for k, v in sorted(dec.items()):
    L = lab.get(k, {})
    f = v["features"]
    bad = L.get("f16_avg") is not None and (L["f16_avg"] < 5 or L.get("f16_loop_rate", 0) >= 0.2)
    y = None if bad else L.get("y")
    pros = v["tag"] not in early and k not in existing
    d = {
        "check": v["decision"]["check"] != "allow",
        "top2": v["decision"]["top2"] == "refuse",
        "top2n": f["b_top2"] <= T2N,
        "klm": v["decision"]["klm"] == "refuse",
        "klmn": f["kl_mean"] >= KLN,
    }
    rows.append((k, y, bad, pros, d, L, f))
    flag = " UNUSABLE" if bad else ""
    print(
        f"{k:28s} {Y[y]:9s}{flag:9s} {'prospective' if pros else 'not-prosp':11s} diff {L.get('diff')} {L.get('ci95')} "
        f"b_top2 {f['b_top2']:.3f} kl {f['kl_mean']:.4f} | "
        + " ".join(f"{p}={'R' if r else 'A'}" for p, r in d.items())
    )


def summ(sel, name):
    for p in ("check", "top2", "top2n", "klm", "klmn"):
        dm = [r for r in sel if r[1] is not None and r[1] > 0]
        ok = [r for r in sel if r[1] == 0]
        mis = sum(1 for r in dm if not r[4][p])
        fa = sum(1 for r in ok if r[4][p])
        ba = (
            ((len(dm) - mis) / len(dm) + (len(ok) - fa) / len(ok)) / 2
            if dm and ok
            else float("nan")
        )
        print(
            f"  {name:22s} {p:6s} damaged {len(dm)} lossless {len(ok)} misclear {mis} false alarm {fa} BA {ba:.3f}"
        )


summ(rows, "all usable cells")
summ([r for r in rows if r[3]], "prospective only")
pr = [r for r in rows if r[3] and r[1] is not None]
ok = all(not r[4]["top2n"] for r in pr if r[1] == 0) and all(r[4]["top2n"] for r in pr if r[1] > 0)
nfa_n = sum(1 for r in pr if r[1] == 0 and r[4]["top2n"])
nfa_c = sum(1 for r in pr if r[1] == 0 and r[4]["check"])
ncol = sum(1 for r in pr if r[1] == 2)
print(
    f"VERDICT PP4: prospective labelled cells {len(pr)} ({ncol} collapse); P_top2n false alarms {nfa_n} vs P_check "
    f"{nfa_c}; collapse allowed {sum(1 for r in pr if r[1] == 2 and not r[4]['top2n'])} -> "
    + ("UNTESTED for misses (no collapse cell); " if ncol == 0 else "")
    + ("PASS" if nfa_n <= nfa_c and all(r[4]["top2n"] for r in pr if r[1] == 2) else "FAIL")
)
