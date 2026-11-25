"""e2_analyze (v2): SpinTare E2, phases A (X-0081/X-0085) and B (X-0086).
Reads runs/e2a/<tag>/run.log and runs/e2b/<tag>/run.log pulled from the run machine and judges the
frozen X-0086 predictions on a fixed expected grid (missing data -> INCOMPLETE, never HOLDS/FAILS):
  P1 each collapsed model: the add-back cell with the lowest D at the model's registered longest
  length Lmax has
     D(Lmax) <= 1.5 x its own D(2048)            [registered reading; the min-ratio reading is printed for reference]
  P2 pm D(Lmax)/D(2048) > the P1 cell's ratio on >= 2 of the 3 collapsed models P3 Llama-3.1-8B: pm
  <= 1.3 x no-repair q4_0 at every registered length (no-repair = phase-B repeat; P4 shows it
     equals phase A)
  P4 phase-B q4_0 equals phase A within 1% at every registered (model, L) ACT activation: each
  repair cell differs from q4_0 by more than 2 combined standard errors (the engine's INFO line
     'KV de-offset' is not printed by llama-perplexity, so the registered string check is
     unavailable)
Lengths 65536 / 131072 use pg19_long (pg19_text + pg19_text2); a 1-chunk run scores a prefix of the
same PG19 text. Usage: python3 e2_analyze.py <runs_dir> [--json out.json]; exit 0 = complete and
printed, 3 = incomplete grid.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

LINE = re.compile(
    r"^(\S+) (\S+) L=(\d+) (\S+) rc=(\d+) chunks_done=(\d+) want=(\d+)"
    r"(?:\s+Mean\s+KLD:\s+([-+0-9.eE]+)\s+±\s+([-+0-9.eE]+))?"
)
# registered grid (X-0086): model -> (lengths, add-back cells)
GRID = {
    "qwen25_7b-q8_0": ([2048, 8192, 32768], ["abmu", "abbias"]),
    "q8_r1q7-q8_0": ([2048, 8192, 32768, 65536], ["abbias"]),
    "qwen3_0p6b-q8_0": ([2048, 8192, 32768], ["abmu"]),
    "llama31_8b-q8_0": ([2048, 8192, 32768, 131072], []),
}
COLLAPSED = ["qwen25_7b-q8_0", "q8_r1q7-q8_0", "qwen3_0p6b-q8_0"]
SAFE = "llama31_8b-q8_0"
LONG_TEXT = {65536, 131072}


def load(runs: Path):
    D, bad, dup = {}, [], []
    for phase in ("e2a", "e2b"):
        for log in sorted((runs / phase).glob("*/run.log")):
            for ln in log.read_text(encoding="utf-8", errors="replace").splitlines():
                m = LINE.match(ln)
                if not m:
                    continue
                ts, tag, L, cell, rc, done, want, kld, se = m.groups()
                if cell == "f16":
                    if rc != "0":
                        bad.append(
                            {
                                "phase": phase,
                                "tag": tag,
                                "cell": "f16",
                                "L": int(L),
                                "rc": rc,
                                "done": done,
                                "want": want,
                                "ts": ts,
                            }
                        )
                    continue
                key = "q4_0_B" if (phase == "e2b" and cell == "q4_0") else cell
                if rc == "0" and done == want and kld is not None:
                    cur = D.setdefault(tag, {}).setdefault(key, {})
                    if int(L) in cur:
                        dup.append(
                            {
                                "tag": tag,
                                "cell": key,
                                "L": int(L),
                                "old": cur[int(L)],
                                "new": (float(kld), float(se)),
                            }
                        )
                    cur[int(L)] = (
                        float(kld),
                        float(se),
                    )  # chronological within a log: last success wins
                else:
                    bad.append(
                        {
                            "phase": phase,
                            "tag": tag,
                            "cell": key,
                            "L": int(L),
                            "rc": rc,
                            "done": done,
                            "want": want,
                            "ts": ts,
                        }
                    )
    bad = [b for b in bad if b["L"] not in D.get(b["tag"], {}).get(b["cell"], {})]
    return D, bad, dup


def get(D, t, c, L):
    return D.get(t, {}).get(c, {}).get(L)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    runs = Path(sys.argv[1])
    D, bad, dup = load(runs)
    expected = []
    for t, (Ls, ab) in GRID.items():
        for L in Ls:
            for c in ["q4_0", "q4_0_B", "pm"] + ab:
                if get(D, t, c, L) is None:
                    expected.append(f"{t}/{c}@{L}")
    cols = ["q4_0", "q4_0norot", "q8_0", "q4_0_B", "pm", "abmu", "abbias"]
    for tag in sorted(D):
        Ls = sorted({L for c in D[tag].values() for L in c})
        print(f"\n== {tag}   (D ± SE; * = pg19_long)")
        print(
            "cell".ljust(10)
            + "".join(f"{str(L) + ('*' if L in LONG_TEXT else ''):>22}" for L in Ls)
        )
        for c in cols:
            if c in D[tag]:
                print(
                    c.ljust(10)
                    + "".join(
                        f"{D[tag][c][L][0]:>12.6f} ±{D[tag][c][L][1]:<8.6f}"
                        if L in D[tag][c]
                        else f"{'-':>22}"
                        for L in Ls
                    )
                )

    def st(ok, complete):
        return "INCOMPLETE" if not complete else ("HOLDS" if ok else "FAILS")

    V = {}
    # P1 (registered reading) and the min-ratio reading
    p1, p1_alt, c1 = {}, {}, True
    for t in COLLAPSED:
        Ls, ab = GRID[t]
        Lmax = Ls[-1]
        cand = [(get(D, t, c, Lmax)[0], c) for c in ab if get(D, t, c, Lmax) and get(D, t, c, 2048)]
        if len(cand) != len(ab) or not cand:
            c1 = False
            continue
        dbest, cbest = min(cand)
        p1[t] = {"cell": cbest, "Lmax": Lmax, "ratio": dbest / get(D, t, cbest, 2048)[0]}
        p1_alt[t] = min((get(D, t, c, Lmax)[0] / get(D, t, c, 2048)[0], c) for c in ab)
    V["P1"] = {
        "status": st(all(v["ratio"] <= 1.5 for v in p1.values()), c1),
        "per_model": p1,
        "min_ratio_reading": p1_alt,
    }
    # P2
    p2, c2 = {}, c1
    for t in COLLAPSED:
        Lmax = GRID[t][0][-1]
        a, b = get(D, t, "pm", Lmax), get(D, t, "pm", 2048)
        if not (a and b) or t not in p1:
            c2 = False
            continue
        p2[t] = {
            "pm_ratio": a[0] / b[0],
            "addback_ratio": p1[t]["ratio"],
            "pm_worse": a[0] / b[0] > p1[t]["ratio"],
        }
    n2 = sum(v["pm_worse"] for v in p2.values())
    V["P2"] = {"status": st(n2 >= 2, c2), "n_true": n2, "per_model": p2}
    # P3
    p3, c3 = {}, True
    for L in GRID[SAFE][0]:
        a, b = get(D, SAFE, "pm", L), get(D, SAFE, "q4_0_B", L)
        if not (a and b) or b[0] <= 0:
            c3 = False
            continue
        p3[L] = a[0] / b[0]
    V["P3"] = {"status": st(all(v <= 1.3 for v in p3.values()), c3), "pm_over_none": p3}
    # P4
    p4, c4 = {}, True
    for t, (Ls, _) in GRID.items():
        for L in Ls:
            a, b = get(D, t, "q4_0_B", L), get(D, t, "q4_0", L)
            if not (a and b) or b[0] <= 0:
                c4 = False
                continue
            p4[f"{t}@{L}"] = a[0] / b[0] - 1
    V["P4"] = {"status": st(all(abs(v) <= 0.01 for v in p4.values()), c4), "rel_diff": p4}
    # activation
    act = {}
    for t, (Ls, ab) in GRID.items():
        for L in Ls:
            base = get(D, t, "q4_0_B", L)
            for c in ["pm"] + ab:
                x = get(D, t, c, L)
                if base and x:
                    z = abs(x[0] - base[0]) / max((x[1] ** 2 + base[1] ** 2) ** 0.5, 1e-12)
                    act[f"{t}/{c}@{L}"] = round(z, 1)
    V["ACT"] = {"status": st(all(z > 2 for z in act.values()), bool(act)), "z": act}
    print("\n== X-0086 verdicts")
    for k, v in V.items():
        print(
            f"{k}: {v['status']}  "
            + json.dumps({x: y for x, y in v.items() if x != "status"}, default=str)
        )
    for title, rows in (
        ("missing from the registered grid", expected),
        ("failed records (not counted)", bad),
        ("duplicate successes (last kept)", dup),
    ):
        if rows:
            print(f"\n== {title}:")
            for r in rows:
                print("  ", r)
    if "--json" in sys.argv:
        i = sys.argv.index("--json")
        if i + 1 >= len(sys.argv):
            sys.exit("--json needs a path")
        Path(sys.argv[i + 1]).write_text(
            json.dumps(
                {
                    "D": {
                        t: {c: {str(L): v for L, v in d.items()} for c, d in cs.items()}
                        for t, cs in D.items()
                    },
                    "verdict": V,
                    "missing": expected,
                    "bad": bad,
                    "dup": dup,
                },
                indent=1,
                default=str,
            )
        )
    sys.exit(3 if expected else 0)


if __name__ == "__main__":
    main()
