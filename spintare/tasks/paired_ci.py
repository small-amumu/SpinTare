# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Paired bootstrap interval and verdict for LongBench scores of a quantised cache against the
16-bit cache.

Usage: python -m spintare paired-ci <results dir> <base tag, e.g. f16> <quant tag, e.g. q4_0>
       [--B 10000 (>= 1000)] [--seed 0]
Per-question files: <dir>/<task>_lc_<tag>.json from `spintare longbench`; each record has an id (the
LongBench _id, or the position i for files without it), a score in [0, 1] and a loop flag. Questions
are paired by id: both runs must cover the same questions of every task.
diff = quant - base (points, equal-weight mean of the four tasks), with a 95% paired bootstrap
interval that resamples the questions of each task independently (stratified; tasks may have
different numbers of questions). The verdict:
  collapse:      drop >= 20 points, or relative drop >= 50%, or repetition-loop rate >= 20%;
  lossless:      interval lower bound >= -3 (the 3-point non-inferiority margin);
  partial_drop:  interval upper bound < -3 and not a collapse;
  undecided:     otherwise.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

TASKS = ["2wikimqa", "hotpotqa", "musique", "passage_retrieval_en"]


def load(d: str, tag: str) -> list[dict[object, tuple[float, bool]]]:
    """Per task: {question id: (score, loop)}."""
    out = []
    for t in TASKS:
        path = Path(d) / f"{t}_lc_{tag}.json"
        if not path.exists():
            raise SystemExit(f"{path} not found (paired-ci reads the four tasks {TASKS})")
        with open(path, encoding="utf-8") as f:
            recs = json.load(f)["records"]
        by_id: dict[object, tuple[float, bool]] = {}
        for x in recs:
            qid = x.get("id") if x.get("id") is not None else x.get("i")
            if qid is None:
                raise SystemExit(f"{path}: a record has neither id nor i")
            if qid in by_id:
                raise SystemExit(f"{path}: duplicate question id {qid!r}")
            by_id[qid] = (float(x["score"]), bool(x.get("loop")))
        out.append(by_id)
    return out


def main() -> None:
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    d, base, quant = sys.argv[1:4]

    def arg(k: str, v: int) -> int:
        if k not in sys.argv:
            return v
        i = sys.argv.index(k) + 1
        if i >= len(sys.argv) or not sys.argv[i].lstrip("-").isdigit():
            raise SystemExit(f"{k} needs an integer value")
        return int(sys.argv[i])

    B, seed = arg("--B", 10000), arg("--seed", 0)
    if B < 1000:
        raise SystemExit("--B must be >= 1000")
    fb_, qb_ = load(d, base), load(d, quant)
    tasks: list[tuple[list[float], list[float]]] = []
    loops = n_all = 0
    for t, a, b in zip(TASKS, fb_, qb_):
        if set(a) != set(b):
            raise SystemExit(
                f"{t}: base and quant runs cover different questions "
                f"({len(set(a) ^ set(b))} ids differ)"
            )
        ids = sorted(a, key=str)
        if not ids:
            raise SystemExit(f"{t}: no questions in the result files")
        if not all(math.isfinite(s) and 0.0 <= s <= 1.0 for i in ids for s in (a[i][0], b[i][0])):
            raise SystemExit(
                f"{t}: scores must be fractions in [0, 1] as `spintare longbench` writes them"
            )
        tasks.append(([a[i][0] for i in ids], [b[i][0] for i in ids]))
        loops += sum(b[i][1] for i in ids)
        n_all += len(ids)

    def avg(x: list[float]) -> float:
        return sum(x) / len(x)

    fb = 100 * sum(avg(f) for f, _ in tasks) / len(tasks)
    qb = 100 * sum(avg(q) for _, q in tasks) / len(tasks)
    diff = qb - fb
    rnd = random.Random(seed)
    ds = []
    for _ in range(B):  # stratified: the questions of each task are resampled on their own
        acc = 0.0
        for f, q in tasks:
            n = len(f)
            idx = [rnd.randrange(n) for _ in range(n)]
            acc += sum(q[j] - f[j] for j in idx) / n
        ds.append(100 * acc / len(tasks))
    ds.sort()
    lo, hi = ds[int(0.025 * B)], ds[int(0.975 * B) - 1]
    loop_rate = loops / n_all
    rel = diff / fb if fb else 0.0
    if diff <= -20 or rel <= -0.5 or loop_rate >= 0.2:
        verdict = "collapse"
    elif lo >= -3:
        verdict = "lossless"
    elif hi < -3:
        verdict = "partial_drop"
    else:
        verdict = "undecided"
    print(
        json.dumps(
            {
                "dir": d,
                "base": base,
                "quant": quant,
                "n_per_task": [len(f) for f, _ in tasks],
                "base_avg": round(fb, 2),
                "quant_avg": round(qb, 2),
                "diff": round(diff, 2),
                "ci95": [round(lo, 2), round(hi, 2)],
                "rel": round(rel, 3),
                "loop_rate": round(loop_rate, 3),
                "verdict": verdict,
            }
        )
    )


if __name__ == "__main__":
    main()
