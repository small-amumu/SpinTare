# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Final-round zero-GPU analyses Z1-Z9 on existing data (GATE-PRED a1 natural cells, PRED-P cells,
logs).

All fits are by the same rule as before: candidate thresholds = observed values, maximise balanced
accuracy, ties to the more cautious threshold. "natural" = cells of released checkpoints (model name
without "inj"). Output: Markdown report on stdout.
"""

from __future__ import annotations

import json
import os
import math
import random
import re
import statistics as st
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
RES = Path(os.environ.get("KVREL_RESULTS", HERE.parents[1] / "results")) / "spintare"
A1 = RES / "detection_gate"  # gp_score.json + results/<tag>/probe.json
PP = RES / "prospective" / "predp"  # decisions.json, labels.json, results/<tag>/probe.json
FK = {"q4_0": "hadamard/q4_0", "q4_0norot": "identity/q4_0", "q4_1": "hadamard/q4_1"}
WRONG = {"q4_0": "identity/q4_0", "q4_0norot": "hadamard/q4_0", "q4_1": "identity/q4_1"}
QWEN2 = ("Qwen2", "R1-Qwen", "qwen2_", "coder_", "math_")
FAMILY = [
    ("Qwen2", "qwen2"),
    ("R1-Qwen", "qwen2"),
    ("Qwen3", "qwen3"),
    ("Llama", "llama"),
    ("Mistral", "mistral"),
    ("gemma-2", "gemma2"),
    ("gemma-3", "gemma3"),
    ("SmolLM2", "smollm"),
    ("Yi", "yi"),
    ("StableLM", "stablelm"),
    ("Pythia", "pythia"),
    ("StarCoder2", "starcoder"),
    ("deepseek", "deepseek"),
    ("OLMo", "olmo"),
    ("Falcon3", "falcon"),
    ("granite", "granite"),
]
nan = float("nan")


def fam(m: str) -> str:
    return next((f for p, f in FAMILY if m.startswith(p)), m)


def layers(pj: dict, key: str) -> dict:
    L = [pj["per_layer"][str(i)][key] for i in range(pj["layers_config"])]
    return {
        "bm": [x["b_meas"] for x in L],
        "bmin": [x["b_meas_min"] for x in L],
        "kl": [x["attn_kl"] for x in L],
    }


def topk_energy(b: list[float], k: int | None) -> float:
    e = sorted((4.0 ** (-x) for x in b), reverse=True)
    e = e if k is None else e[:k]
    return -math.log(sum(e) / len(e), 4)


# ---------- cells
def load_a1() -> list[dict]:
    cells = json.load(open(A1 / "gp_score.json", encoding="utf-8"))["cells"]
    out, cache = [], {}
    for c in cells:
        t = str(c.get("tag"))
        if t not in cache:
            f = A1 / "results" / t / "probe.json"
            cache[t] = json.load(open(f, encoding="utf-8")) if f.exists() else None
        pj = cache[t]
        if pj is None or c["cfg"] not in FK or c.get("y") is None:
            continue
        out.append(
            {
                "src": "a1",
                "model": c["model"],
                "tag": t,
                "cfg": c["cfg"],
                "y": c["y"],
                "D": c["D"],
                "inj": "inj" in c["model"],
                "fam": fam(c["model"].replace("T6m-", "")),
                "dec": c.get("dec", {}),
                "L": layers(pj, FK[c["cfg"]]),
                "Lw": layers(pj, WRONG[c["cfg"]]),
                "bias_stat": pj["weights"]["bias_stat"],
            }
        )
    return out


def load_pp() -> list[dict]:
    dec = json.load(open(PP / "decisions.json"))
    lab = json.load(open(PP / "labels.json", encoding="utf-8"))
    out = []
    for k, v in dec.items():
        L = lab.get(k, {})
        pj = json.load(open(PP / "results" / v["tag"] / "probe.json", encoding="utf-8"))
        out.append(
            {
                "src": "pp",
                "model": v["tag"],
                "tag": v["tag"],
                "cfg": v["cfg"],
                "D": v["D"],
                "y": L.get("y"),
                "diff": L.get("diff"),
                "f16_avg": L.get("f16_avg"),
                "f16_loop": L.get("f16_loop_rate"),
                "inj": False,
                "fam": v["tag"].split("_")[0],
                "L": layers(pj, FK[v["cfg"]]),
                "Lw": layers(pj, WRONG[v["cfg"]]),
                "bias_stat": pj["weights"]["bias_stat"],
            }
        )
    return out


a1 = load_a1()
nat = [c for c in a1 if not c["inj"]]
inj = [c for c in a1 if c["inj"]]
pp = load_pp()
pp_ok = [
    c
    for c in pp
    if c["y"] is not None
    and not (c["f16_avg"] is not None and (c["f16_avg"] < 5 or c["f16_loop"] >= 0.2))
]


# ---------- fit / score helpers (refuse iff value <= threshold, for bit-like scores)
def ba_of(pairs: list[tuple[bool, bool]]) -> float:
    d = [p for p, y in pairs if y]
    o = [p for p, y in pairs if not y]
    if not d or not o:
        return nan
    return (sum(d) / len(d) + sum(not p for p in o) / len(o)) / 2


def fit_bits(cells: list[dict], f) -> float:
    vals = [(f(c), c["y"] > 0) for c in cells]
    if all(y for _, y in vals) or not any(y for _, y in vals):
        return nan
    best = None
    for t in sorted({v for v, _ in vals}):
        b = ba_of([(v <= t, y) for v, y in vals])
        if best is None or b > best[0] + 1e-12 or (abs(b - best[0]) <= 1e-12 and t > best[1]):
            best = (b, t)
    return best[1]


def lofo(cells: list[dict], f) -> tuple[float, int, int, int]:
    pairs = []
    for fa in sorted({c["fam"] for c in cells}):
        tr = [c for c in cells if c["fam"] != fa]
        t = fit_bits(tr, f)
        pairs += [
            ((f(c) <= t) if not math.isnan(t) else False, c["y"] > 0, c["y"])
            for c in cells
            if c["fam"] == fa
        ]
    miss = sum(1 for p, d, y in pairs if d and not p)
    missc = sum(1 for p, d, y in pairs if y == 2 and not p)
    fa_ = sum(1 for p, d, y in pairs if p and not d)
    return ba_of([(p, d) for p, d, _ in pairs]), miss, missc, fa_


F = {
    "bit mean, all layers (a1 original)": lambda c: st.mean(c["L"]["bm"]),
    "energy mean, all layers": lambda c: topk_energy(c["L"]["bm"], None),
    "worst layer (k=1)": lambda c: topk_energy(c["L"]["bm"], 1),
    "energy top-2 (b_top2)": lambda c: topk_energy(c["L"]["bm"], 2),
    "energy top-3": lambda c: topk_energy(c["L"]["bm"], 3),
    "energy top-4": lambda c: topk_energy(c["L"]["bm"], 4),
    "worst block per layer, energy top-2": lambda c: topk_energy(c["L"]["bmin"], 2),
    "b_top2 in the wrong basis": lambda c: topk_energy(c["Lw"]["bm"], 2),
}
print(
    "# Final-round zero-GPU analyses (exploratory unless stated; natural = released checkpoints)\n"
)
print(
    f"cells: a1 natural {len(nat)} ({sum(c['y'] > 0 for c in nat)} damaged), a1 injected {len(inj)}, "
    f"PRED-P usable decided {len(pp_ok)} ({sum(c['y'] > 0 for c in pp_ok)} damaged)\n"
)

# ---------- Z2 single-variable ablations (LOFO within natural)
print("## Z2 single-variable ablations: leave-one-family-out within natural a1 cells\n")
print("| feature | LOFO BA | misclear (collapse) | false alarm |\n|---|---|---|---|")
for n, f in F.items():
    b, m, mc, fa_ = lofo(nat, f)
    print(f"| {n} | {b:.3f} | {m} ({mc}) | {fa_} |")

# ---------- Z1 nested selection of k
print(
    "\n## Z1 nested selection of k (outer: leave one family out; inner: LOFO on the training families)\n"
)
KS = [1, 2, 3, 4, None]
pairs, chosen = [], []
for fa in sorted({c["fam"] for c in nat}):
    tr = [c for c in nat if c["fam"] != fa]
    inner = []
    for k in KS:
        b = lofo(tr, lambda c, k=k: topk_energy(c["L"]["bm"], k))[0]
        inner.append((-1 if math.isnan(b) else b, -KS.index(k), k))
    k = max(inner)[2]
    chosen.append("all" if k is None else k)
    f = lambda c, k=k: topk_energy(c["L"]["bm"], k)
    t = fit_bits(tr, f)
    pairs += [(f(c) <= t, c["y"] > 0, c["y"]) for c in nat if c["fam"] == fa]
print(
    f"outer BA {ba_of([(p, d) for p, d, _ in pairs]):.3f}; misclear {sum(1 for p, d, _ in pairs if d and not p)} "
    f"(collapse {sum(1 for p, d, y in pairs if y == 2 and not p)}); false alarm {sum(1 for p, d, _ in pairs if p and not d)}; "
    f"k chosen per outer fold: {dict((str(x), chosen.count(x)) for x in set(chosen))}"
)

# ---------- Z3 threshold source and bootstrap
print("\n## Z3 threshold source (b_top2)\n")
f2 = F["energy top-2 (b_top2)"]
th = {"natural": fit_bits(nat, f2), "injected": fit_bits(inj, f2), "pooled": fit_bits(a1, f2)}
print("| fitted on | threshold | BA on a1 natural | BA on PRED-P usable |\n|---|---|---|---|")
for n, t in th.items():
    print(
        f"| {n} | {t:.4f} | {ba_of([(f2(c) <= t, c['y'] > 0) for c in nat]):.3f} | "
        f"{ba_of([(f2(c) <= t, c['y'] > 0) for c in pp_ok]):.3f} |"
    )
rng = random.Random(0)
models = sorted({c["model"] for c in nat})
boots = []
for _ in range(2000):
    pick = [rng.choice(models) for _ in models]
    s = [c for m in pick for c in nat if c["model"] == m]
    t = fit_bits(s, f2)
    if not math.isnan(t):
        boots.append(t)
boots.sort()
print(
    f"\nnatural threshold, bootstrap over models (2000): median {st.median(boots):.3f}, 95% interval "
    f"[{boots[int(0.025 * len(boots))]:.3f}, {boots[int(0.975 * len(boots)) - 1]:.3f}]"
)

# ---------- Z4 null controls
print("\n## Z4 null controls (b_top2, natural a1 cells)\n")
obs = ba_of([(f2(c) <= th["natural"], c["y"] > 0) for c in nat])
null = []
ys = [c["y"] for c in nat]
for i in range(2000):
    perm = ys[:]
    rng.shuffle(perm)
    tmp = [dict(c, y=y) for c, y in zip(nat, perm)]
    t = fit_bits(tmp, f2)
    null.append(ba_of([(f2(c) <= t, c["y"] > 0) for c in tmp]))
p = (sum(1 for x in null if x >= obs) + 1) / (len(null) + 1)
print(
    f"label permutation (refit each time, in-sample): observed BA {obs:.3f}; null median {st.median(null):.3f}, "
    f"95th pct {sorted(null)[int(0.95 * len(null))]:.3f}; p = {p:.4f}"
)
rb = []
for s in range(200):
    r2 = random.Random(s)

    def frand(c, r2=r2):
        e = sorted(range(len(c["L"]["bm"])), key=lambda i: -(4.0 ** (-c["L"]["bm"][i])))
        rest = e[2:]
        pick = r2.sample(rest, 2) if len(rest) >= 2 else e[:2]
        return -math.log(sum(4.0 ** (-c["L"]["bm"][i]) for i in pick) / 2, 4)

    rb.append(lofo(nat, frand)[0])
print(
    f"two random layers other than the worst two, LOFO BA over 200 seeds: median {st.median(rb):.3f}, "
    f"max {max(rb):.3f} (b_top2 LOFO {lofo(nat, f2)[0]:.3f})"
)

# ---------- Z5 gate comparison incl. zero-cost baselines
print(
    "\n## Z5 gate comparison (natural cells; rules with a fitted threshold are LOFO on a1, frozen on PRED-P)\n"
)


def fam_rule(c):
    m = c["model"]
    return (
        (m.startswith(QWEN2) or c["tag"].startswith(("qwen2_", "coder_")))
        and "14B" not in m
        and "14b" not in c["tag"]
    )


rules = {
    "engine check (frozen)": lambda c: c["D"] >= 0.1016,
    "b_top2, natural threshold": lambda c: f2(c) <= 2.0280476994784027,
    "vLLM PR 63081 bias rule (> 100)": lambda c: c["bias_stat"] > 100,
    "family rule (Qwen2 family < 14B)": fam_rule,
}
print("| rule | a1 natural BA (miss / FA) | PRED-P usable BA (miss / FA) |\n|---|---|---|")
for n, r in rules.items():
    row = []
    for cs in (nat, pp_ok):
        pr = [(r(c), c["y"] > 0) for c in cs]
        row.append(
            f"{ba_of(pr):.3f} ({sum(1 for p, d in pr if d and not p)} / {sum(1 for p, d in pr if p and not d)})"
        )
    print(f"| {n} | {row[0]} | {row[1]} |")
b, m, mc, fa_ = lofo(nat, f2)
print(f"| b_top2, LOFO-fitted (honest a1 figure) | {b:.3f} ({m} / {fa_}) | - |")

# ---------- Z6 policy-level evaluation on PRED-P
print(
    "\n## Z6 policy level on PRED-P (cells with a task difference, unusable-label models excluded)\n"
)
cells6 = [c for c in pp if c["diff"] is not None and not (c["f16_avg"] < 5 or c["f16_loop"] >= 0.2)]
pol = {
    "always 4-bit": lambda c: False,
    "always 8-bit": lambda c: True,
    **{k: v for k, v in rules.items()},
    "oracle (refuse iff label damaged or diff < -3)": lambda c: (c["y"] or 0) > 0 or c["diff"] < -3,
}
print(
    "assumption: a refused cell falls back to 8-bit K, taken as lossless (K8V4 / q8 K lossless in the earlier runs)\n"
)
print(
    "| policy | mean score change | worst cell | mean K bits per element | cells at 4 bit |\n|---|---|---|---|---|"
)
for n, r in pol.items():
    ch = [0.0 if r(c) else c["diff"] for c in cells6]
    bits = st.mean([8.5 if r(c) else 4.5 for c in cells6])
    print(
        f"| {n} | {st.mean(ch):+.2f} | {min(ch):+.2f} | {bits:.2f} | {sum(not r(c) for c in cells6)} / {len(cells6)} |"
    )

# ---------- Z7 statistics: cluster bootstrap and label sensitivity
print("\n## Z7 cluster bootstrap (by model) of BA, and label-threshold sensitivity\n")
for name, cs in (("a1 natural", nat), ("PRED-P usable", pp_ok)):
    ms = sorted({c["model"] for c in cs})
    for rn in (
        "engine check (frozen)",
        "b_top2, natural threshold",
        "family rule (Qwen2 family < 14B)",
    ):
        vals = []
        for _ in range(2000):
            pick = [rng.choice(ms) for _ in ms]
            s = [c for m in pick for c in cs if c["model"] == m]
            v = ba_of([(rules[rn](c), c["y"] > 0) for c in s])
            if not math.isnan(v):
                vals.append(v)
        vals.sort()
        print(
            f"- {name}, {rn}: BA 95% cluster interval [{vals[int(0.025 * len(vals))]:.3f}, "
            f"{vals[int(0.975 * len(vals)) - 1]:.3f}] ({len(ms)} models)"
        )
pp15 = [c for c in pp if c["y"] is not None and c["f16_avg"] >= 15 and c["f16_loop"] < 0.2]
print(
    f"- PRED-P with label floor f16 >= 15 instead of 5: {len(pp15)} cells ({sum(c['y'] > 0 for c in pp15)} damaged); "
    + "; ".join(f"{n} BA {ba_of([(r(c), c['y'] > 0) for c in pp15]):.3f}" for n, r in rules.items())
)
und = [c for c in pp if c["y"] is None and c["diff"] is not None and not (c["f16_avg"] < 5)]
print(f"- undecidable PRED-P cells: {[(c['tag'], c['cfg'], c['diff']) for c in und]}")

# ---------- Z8 timing audit
print("\n## Z8 timing audit of PRED-P amendment 1 (04:05:15Z)\n")
if not (PP / "run.log").exists():
    print("(the PRED-P run log is not shipped; Z8 needs the raw log of the run machine)")
else:
    log = (PP / "run.log").read_text(encoding="utf-8")
    done = {}
    for m in re.finditer(r"^(\S+) B (\S+) (\S+) gpu .*DONE", log, re.M):
        done[(m.group(2), m.group(3))] = m.group(1)
    amend = "2026-10-26T04:05:15Z"
    before = sorted(f"{t}|{c}" for (t, c), ts in done.items() if ts < amend)
    print(
        f"decisions (with every new cell's b_top2 and D) were written at 03:00:12Z and were visible before amendment 1; "
        f"the amendment's thresholds were fitted on a1 cells only. Task summaries finished before 04:05:15Z: {len(before)} "
        f"cells: {before}"
    )

# ---------- Z9 cost
print("\n## Z9 cost from logs\n")
if not (PP / "run_probe.log").exists():
    print("(the PRED-P probe log is not shipped; Z9 needs the raw logs of the run machine)")
else:
    pl = (PP / "run_probe.log").read_text(encoding="utf-8")
    starts = {m.group(2): m.group(1) for m in re.finditer(r"^(\S+) (\S+) order ", pl, re.M)}
    ends = {m.group(2): m.group(1) for m in re.finditer(r"^(\S+) (\S+) done:", pl, re.M)}
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    for t in sorted(ends):
        if t in starts:
            sec = (
                datetime.strptime(ends[t], fmt) - datetime.strptime(starts[t], fmt)
            ).total_seconds()
            dd = [PP.parent / "predp" / "d" / f"{t}_{x}.log" for x in ("f16", "q4_0")]
            eng = []
            for f in dd:
                txt = f.read_text(errors="replace") if f.exists() else ""
                m = re.search(r"total time =\s*([\d.]+) ms", txt)
                eng.append(float(m.group(1)) / 1000 if m else nan)
            print(
                f"- {t}: probe wall time {sec:.0f} s (model load included, all features); engine check one format "
                f"(f16 reference + q4_0 pass) {sum(eng):.0f} s"
            )
