# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""E2 scoring (PREDICTIONS.md E2/E3 design): answer identity vs bf16lp and the bf16 top-1/top-2
margin at the first differing generated token."""

import os
import json
from pathlib import Path

D = Path(os.environ.get("KVREL_WORK", "work")) / "results/lc_vllm/v8_qwen25_7b"
TASKS = ("hotpotqa", "2wikimqa", "musique", "passage_retrieval_en")


def load(lab: str) -> list[dict]:
    out = []
    for t in TASKS:
        out += [dict(r, task=t) for r in json.load(open(D / f"{t}_lc_{lab}.json"))["records"]]
    return out


def same(a: list[dict], b: list[dict]) -> int:
    return sum(x["pred"].strip() == y["pred"].strip() for x, y in zip(a, b))


ref = load("bf16lp_n20")
print("P-E2a bf16lp vs bf16san1:", same(ref, load("bf16san1_n20")), "/ 80")
print("P-E2a bf16pmlp vs bf16pm:", same(load("bf16pmlp_n20"), load("bf16pm_n20")), "/ 80")
for lab in ("bf16z_n20", "bf16rc_n20", "bf16pmlp_n20"):
    arm = load(lab)
    print(
        f"{lab}: identical {same(ref, arm)}/80, mean score diff {100 * sum(y['score'] - x['score'] for x, y in zip(ref, arm)) / 80:+.2f}"
    )
    margins = []
    for x, y in zip(ref, arm):
        if x["pred"].strip() == y["pred"].strip():
            continue
        g1, g2 = x["gen"], y["gen"]
        j = next((k for k in range(min(len(g1), len(g2))) if g1[k] != g2[k]), min(len(g1), len(g2)))
        m_ref = (
            x["top2"][j][0] - x["top2"][j][1]
            if j < len(x["top2"]) and len(x["top2"][j]) == 2
            else None
        )
        m_arm = (
            y["top2"][j][0] - y["top2"][j][1]
            if j < len(y["top2"]) and len(y["top2"][j]) == 2
            else None
        )
        margins.append((x["task"], x["i"], j, m_ref, m_arm, round(y["score"] - x["score"], 3)))
    for m in margins:
        print("   ", m)
    vals = [m[3] for m in margins if m[3] is not None]
    if vals:
        print(
            f"   bf16 margin at first divergence: max {max(vals):.4f}, median {sorted(vals)[len(vals) // 2]:.4f}, n {len(vals)}"
        )
