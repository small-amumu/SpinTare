# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Score MECH-D: which model property predicts the 4-bit K failure (attention pattern vs shared key
offset).

Reads the GATE-PRED a1 probe JSONs and the same cells (gp_score.load_cells). Prints per-feature AUC
(hypothesised orientation), exact / Monte-Carlo permutation p-values, Spearman rho with log10 D of
the q4_0 rot-on cell, and the Llama-3.1-8B injection intervention table. Usage: python
mechd_score.py --probe DIR --peel DIR --registry CSV --out JSON
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I drops the script directory
from gp_score import load_cells  # noqa: E402

# training-lineage families for the leave-one-family-out AUC (review r31 S-8)
FAMILY = [
    ("Qwen2", "qwen2"),
    ("R1-Qwen", "qwen2"),
    ("Qwen3", "qwen3"),
    ("Llama", "llama"),
    ("gemma", "gemma"),
    ("SmolLM2", "smollm"),
    ("Yi", "yi"),
    ("StableLM", "stablelm"),
    ("Pythia", "pythia"),
    ("StarCoder2", "starcoder"),
    ("deepseek", "deepseek"),
    ("OLMo", "olmo"),
    ("Falcon3", "falcon"),
    ("granite", "granite"),
    ("Mistral", "mistral"),
]
# a left-out subset counts only if >= 5 collapse and >= 5 other models remain (review r32b S-1)
LOFO_MIN = 5
D_PRIMARY = [
    "b_meas",
    "b_eff",
]  # probe-measured offset features; r_max is secondary (it is a check training feature)
A_FEATS = ["diffuse_head_frac", "norm_entropy", "top10_mass", "sink_mass"]
# +1: larger value -> collapse is the hypothesised direction; -1: smaller value -> collapse
ORIENT = {
    "diffuse_head_frac": 1,
    "norm_entropy": 1,
    "top10_mass": -1,
    "sink_mass": -1,
    "has_bias": 1,
    "bias_stat": 1,
    "size_b": -1,
    "r_max": 1,
    "b_meas": -1,
    "b_eff": -1,
}


def auc(scores: list[float], pos: list[bool]) -> float:
    p = [s for s, y in zip(scores, pos) if y]
    n = [s for s, y in zip(scores, pos) if not y]
    if not p or not n:
        return float("nan")
    return sum((x > y) + 0.5 * (x == y) for x in p for y in n) / (len(p) * len(n))


def perm_p(scores: list[float], pos: list[bool]) -> float:
    a0 = auc(scores, pos)
    k, n = sum(pos), len(pos)
    if n <= 20:
        combos = list(itertools.combinations(range(n), k))
        hits = sum(1 for c in combos if auc(scores, [i in c for i in range(n)]) >= a0 - 1e-12)
        return hits / len(combos)
    rng = random.Random(0)
    hits = 0
    for _ in range(100000):
        lab = pos[:]
        rng.shuffle(lab)
        hits += auc(scores, lab) >= a0 - 1e-12
    return (hits + 1) / 100001


def spearman(x: list[float], y: list[float]) -> float:
    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def family(model: str) -> str:
    return next((f for pre, f in FAMILY if model.startswith(pre)), model)


def lofo_auc(rows: list[dict], feat: str, o: int) -> dict:
    """AUC with each family left out in turn; reports the minimum and the family that gives it."""
    res = {}
    for f in sorted({r["family"] for r in rows}):
        rs = [r for r in rows if r["family"] != f and r[feat] is not None]
        res[f] = auc([o * r[feat] for r in rs], [r["collapse"] for r in rs])
    ok = {k: v for k, v in res.items() if not math.isnan(v)}
    return {
        "min": min(ok.values()) if ok else float("nan"),
        "argmin": min(ok, key=ok.get) if ok else None,
    }


def model_rows(cells: list[dict], probe: Path, voids: list[str]) -> list[dict]:
    rows = {}
    for c in cells:
        # natural models only (the injected base models moved to TRAIN-I in GATE-PRED a1 stay in
        # this sample)
        if "_inj" in str(c.get("tag")) or c["cfg"] != "q4_0" or not c.get("tag"):
            continue
        f = probe / c["tag"] / "probe.json"
        if not f.exists():
            voids.append(f"{c['model']}: no probe")
            continue
        pj = json.loads(f.read_text(encoding="utf-8"))
        if pj["layers_captured"] != pj["layers_config"]:
            voids.append(
                f"{c['model']}: captured {pj['layers_captured']} of {pj['layers_config']} layers"
            )
            continue
        s = pj["summary"]
        rows[c["model"]] = {
            "model": c["model"],
            "family": family(c["model"]),
            "collapse": c["y"] == 2,
            "logD": math.log10(c["D"]),
            **{x: s[f"attn/{x}_mean"] for x in A_FEATS},
            "has_bias": float(pj["weights"]["has_k_bias"]),
            "bias_stat": pj["weights"]["bias_stat"],
            "size_b": pj["params_b"],
            "r_max": c.get("r"),
            "b_meas": s["hadamard/q4_0/b_meas_mean"],
            "b_eff": s["hadamard/q4_0/b_eff_mean"],
        }
    return list(rows.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--peel", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    probe = Path(a.probe)
    cells = load_cells(Path(a.peel), Path(a.registry))
    voids: list[str] = []
    rows = model_rows(cells, probe, voids)
    print(
        f"models with a q4_0 rot-on cell and a valid probe: {len(rows)}; void:", *voids, sep="\n  "
    )
    res: dict = {"features": {}, "voids": voids}
    print(
        "| feature | n | AUC (hypothesised dir.) | perm p | min leave-one-family-out AUC | "
        "Spearman vs log10 D (raw, not oriented) |"
    )
    print("|---|---|---|---|---|---|")
    for feat, o in ORIENT.items():
        rs = [r for r in rows if r[feat] is not None]
        sc = [o * r[feat] for r in rs]
        pos = [r["collapse"] for r in rs]
        au = auc(sc, pos)
        p = perm_p(sc, pos) if rs and any(pos) and not all(pos) else float("nan")
        rho = (
            spearman([r[feat] for r in rs], [r["logD"] for r in rs])
            if len(rs) > 2
            else float("nan")
        )
        lf = lofo_auc(rs, feat, o)
        res["features"][feat] = {
            "n": len(rs),
            "auc": au,
            "perm_p": p,
            "lofo": lf,
            "spearman_logD": rho,
        }
        print(
            f"| {feat} | {len(rs)} | {au:.3f} | {p:.4f} | {lf['min']:.3f} (without {lf['argmin']}) | {rho:+.3f} |"
        )
    # M1 on the models where every compared feature has a value (review r31 S-7)
    common = [r for r in rows if all(r[f] is not None for f in D_PRIMARY + A_FEATS)]
    cauc = {
        f: auc([ORIENT[f] * r[f] for r in common], [r["collapse"] for r in common])
        for f in D_PRIMARY + A_FEATS
    }
    best_d = max(cauc[f] for f in D_PRIMARY)
    best_a = max(cauc[f] for f in A_FEATS)
    m1_fam = {}
    for fam in sorted({r["family"] for r in common}):
        rs = [r for r in common if r["family"] != fam]
        pos = [r["collapse"] for r in rs]
        if sum(pos) < LOFO_MIN or len(pos) - sum(pos) < LOFO_MIN:
            m1_fam[fam] = {"excluded": f"{sum(pos)} collapse / {len(pos) - sum(pos)} other left"}
            continue
        fa = {f: auc([ORIENT[f] * r[f] for r in rs], pos) for f in D_PRIMARY + A_FEATS}
        bd, ba = max(fa[f] for f in D_PRIMARY), max(fa[f] for f in A_FEATS)
        m1_fam[fam] = {
            "best_d": bd,
            "best_a": ba,
            "holds": (bd >= 0.7 and bd - ba >= 0.1)
            if not (math.isnan(bd) or math.isnan(ba))
            else None,
        }
    used = [v["holds"] for v in m1_fam.values() if "holds" in v and v["holds"] is not None]
    # magnitude gate (review r32c): the ordering alone passes 26.5% of null draws
    m1 = best_d >= 0.7 and best_d - best_a >= 0.1 and bool(used) and sum(used) >= 0.8 * len(used)
    res["m1_common"] = {"n": len(common), "auc": cauc, "by_left_out_family": m1_fam, "holds": m1}
    print(
        f"M1 (common n {len(common)}) best offset AUC {best_d:.3f} vs best attention AUC {best_a:.3f}; "
        f"per left-out family: {m1_fam}"
    )
    print(
        f"VERDICT M1: {'PASS' if m1 else 'FAIL'} (full sample, and >= 80% of the {len(used)} usable left-out "
        f"subsets)"
    )
    lofo_max = {}
    for f in A_FEATS:
        vals = []
        for fam in sorted({r["family"] for r in rows}):
            rs = [r for r in rows if r["family"] != fam and r[f] is not None]
            pos = [r["collapse"] for r in rs]
            if sum(pos) < LOFO_MIN or len(pos) - sum(pos) < LOFO_MIN:
                continue
            vals.append(auc([ORIENT[f] * r[f] for r in rs], pos))
        ok_vals = [v for v in vals if not math.isnan(v)]
        lofo_max[f] = max(ok_vals) if ok_vals else float("nan")
    a_full = max(res["features"][f]["auc"] for f in A_FEATS)
    if all(math.isnan(v) for v in lofo_max.values()):
        print("M2: no usable left-out subsets")
    m2 = a_full < 0.8 and all(math.isnan(v) or v < 0.8 for v in lofo_max.values())
    print(
        f"VERDICT M2: {'PASS' if m2 else 'FAIL'} (best A AUC full {a_full:.3f}, leave-one-family-out max {lofo_max})"
    )
    m4 = res["features"]["has_bias"]["auc"] < 0.8
    print(
        f"VERDICT M4: {'PASS' if m4 else 'FAIL'} (has_bias AUC {res['features']['has_bias']['auc']:.3f})"
    )
    res["verdict"] = {"M1": m1, "M2": m2, "M4": m4}
    # intervention: Llama-3.1-8B base vs injected doses
    base = probe / "llama31_8b" / "probe.json"
    inj = sorted(p for p in probe.glob("llama31_8b_inj*/probe.json"))
    if not (base.exists() and inj):
        print("VERDICT M3: not evaluated (base or injected probes missing)")
    if base.exists() and inj:
        b = json.loads(base.read_text(encoding="utf-8"))["summary"]
        worst: dict[str, int] = {}
        for c in cells:
            if c.get("tag") and (c["tag"] == "llama31_8b" or c["tag"].startswith("llama31_8b_inj")):
                worst[c["tag"]] = max(worst.get(c["tag"], 0), c["y"])
        print(
            "Llama-3.1-8B worst registered label per dose (0 lossless, 1 partial, 2 collapse):",
            worst,
        )
        res["labels_by_dose"] = worst
        span = (
            {x: (max(r[x] for r in rows) - min(r[x] for r in rows)) for x in A_FEATS}
            if rows
            else {}
        )
        ok = True
        print("| injected model | " + " | ".join(f"{x} rel. change" for x in A_FEATS) + " |")
        res["intervention"] = {}
        for p in inj:
            pj = json.loads(p.read_text(encoding="utf-8"))
            s = pj["summary"]
            if not (pj["weights"]["has_k_bias"] and pj["weights"]["bias_stat"] > 0):
                print(f"| {p.parent.name} | void: injected key bias not loaded |")
                ok = False
                continue
            rel = {
                x: ((s[f"attn/{x}_mean"] - b[f"attn/{x}_mean"]) / b[f"attn/{x}_mean"])
                if b[f"attn/{x}_mean"]
                else float("inf")
                for x in A_FEATS
            }
            ab = {x: abs(s[f"attn/{x}_mean"] - b[f"attn/{x}_mean"]) for x in A_FEATS}
            good = all(abs(rel[x]) < 0.05 and (not span or ab[x] < span[x] / 10) for x in A_FEATS)
            ok &= good
            res["intervention"][p.parent.name] = {"rel": rel, "abs": ab, "within": good}
            print(
                f"| {p.parent.name} | "
                + " | ".join(f"{rel[x]:+.2%}" for x in A_FEATS)
                + f" | within {good} | bias_stat {pj['weights']['bias_stat']:.1f} |"
            )
        lab = worst.get("llama31_8b") == 0 and all(
            v >= 1 for k, v in worst.items() if k != "llama31_8b"
        )
        print(
            f"VERDICT M3: {'PASS' if ok and lab else 'FAIL'} (attention unchanged {ok}; labels lossless -> "
            f"damaged {lab})"
        )
        res["verdict_m3"] = bool(ok and lab)
    Path(a.out).write_text(json.dumps({**res, "rows": rows}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
