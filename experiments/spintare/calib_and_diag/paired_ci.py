# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Item-paired bootstrap interval for LongBench and the registered verdict rule.
Usage: python paired_ci.py <result dir results/lc/<model>> <base label, e.g. f16_n50>
       <quant label, e.g. q4_0_n50> [--B 10000] [--seed 0]
Per-item json: <dir>/<task>_lc_<label>.json, records[i] with score (0-1 or 0-100) and loop.
Verdict on diff = quant - base (points, four tasks weighted equally, one bootstrap index shared by
the tasks):
  collapse:  drop >= 20, or relative drop >= 50%, or looping rate >= 20%;
  lossless:  lower bound >= -3 (the 3-point margin);
  partial:   upper bound < -3 and not a collapse;
  otherwise: undecided.
"""

import json
import random
import sys
from pathlib import Path

TASKS = ["2wikimqa", "hotpotqa", "musique", "passage_retrieval_en"]


def load(d, tag):
    recs = [json.load(open(Path(d) / f"{t}_lc_{tag}.json"))["records"] for t in TASKS]
    n = min(len(r) for r in recs)
    s = [[float(x["score"]) for x in r[:n]] for r in recs]
    loops = sum(bool(x.get("loop")) for r in recs for x in r[:n])
    return s, loops, n


def main():
    d, base, quant = sys.argv[1:4]
    arg = lambda k, v: type(v)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else v
    B, seed = arg("--B", 10000), arg("--seed", 0)
    f, _, n1 = load(d, base)
    q, loops, n2 = load(d, quant)
    n = min(n1, n2)
    f = [x[:n] for x in f]
    q = [x[:n] for x in q]
    sc = 100 if max(max(x) for x in f) <= 1 else 1
    mean = lambda s, idx: (
        sum(s[i][j] for i in range(len(TASKS)) for j in idx) / len(idx) / len(TASKS) * sc
    )
    allidx = list(range(n))
    fb, qb = mean(f, allidx), mean(q, allidx)
    diff = qb - fb
    rnd = random.Random(seed)
    ds = []
    for _ in range(B):
        idx = [rnd.randrange(n) for _ in range(n)]
        ds.append(mean(q, idx) - mean(f, idx))
    ds.sort()
    lo, hi = ds[int(0.025 * B)], ds[int(0.975 * B) - 1]
    loop_rate = loops / (n * len(TASKS))
    rel = diff / fb if fb else 0.0
    if diff <= -20 or rel <= -0.5 or loop_rate >= 0.2:
        verdict = "collapse"
    elif lo >= -3:
        verdict = "lossless"
    elif hi < -3:
        verdict = "partial"
    else:
        verdict = "undecided"
    print(
        json.dumps(
            {
                "dir": d,
                "base": base,
                "quant": quant,
                "n_per_task": n,
                "base_avg": round(fb, 2),
                "quant_avg": round(qb, 2),
                "diff": round(diff, 2),
                "ci95": [round(lo, 2), round(hi, 2)],
                "rel": round(rel, 3),
                "loop_rate": round(loop_rate, 3),
                "verdict": verdict,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
