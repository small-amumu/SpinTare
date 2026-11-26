# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""PRED-P: frozen predictors and scoring for the prospective test of the corrected single-forward
predictors.

Rules (frozen at registration; thresholds fitted on the 96 GATE-PRED a1 cells, exploratory analysis
2026-10-26):
  P_check  engine D (llama-perplexity KL, census flags): D < 0.1016 allow, 0.1016-0.3577 fall back, > 0.3577 alarm
  P_top2   per-layer b_meas of the cell's own format in engine coordinates -> error energy 4^-b per layer -> mean of
           the two largest -> back to bits: b_top2 = -log4(mean). Refuse iff b_top2 <=
           2.342775337149283
  P_klm    layer mean of the simulated K-only attention KL of the cell's format. Refuse iff >= 0.09453133872843214
"not allow" = refuse / fall back / alarm. damaged = label partial or collapse. Label validity (fixed
before any task run): a model whose f16 cell scores below 5 points (LongBench average) or loops in
>= 20% of its items gives no usable labels; its cells are listed but left out of the scored set.
Baseline (descriptive): P_family refuses every cell of a Qwen2-family model below 14B (QWEN2_TAGS),
allows the rest.
Usage:
  python pp_rules.py decide <probe_dir> <d_json> <out_json>          (before any task run)
  python pp_rules.py score  <decisions_json> <labels_json> <out_json>  (after the task runs)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

TA, TR = 0.1016, 0.3577
TOP2_MAX_BITS = 2.342775337149283
KLM_MIN = 0.09453133872843214
FEATKEY = {"q4_0": "hadamard/q4_0", "q4_0norot": "identity/q4_0", "q4_1": "hadamard/q4_1"}
CFGS = ("q4_0", "q4_0norot", "q4_1")
QWEN2_TAGS = {"qwen2_0p5b", "qwen2_1p5b", "coder_3b"}
F16_MIN_AVG, F16_MAX_LOOP = 5.0, 0.2
PREDS = ("check", "top2", "klm", "family")


def features(pj: dict, cfg: str) -> dict:
    k = FEATKEY[cfg]
    layers = [pj["per_layer"][str(i)][k] for i in range(pj["layers_config"])]
    if not all(math.isfinite(x["b_meas"]) and math.isfinite(x["attn_kl"]) for x in layers):
        raise ValueError(
            "non-finite per-layer feature"
        )  # review r36b S-1: one NaN layer can corrupt the top-2 sort
    energy = sorted((4.0 ** (-x["b_meas"]) for x in layers), reverse=True)
    b_top2 = -math.log(sum(energy[:2]) / 2, 4)
    klm = sum(x["attn_kl"] for x in layers) / len(layers)
    if not (math.isfinite(b_top2) and math.isfinite(klm)):
        raise ValueError("non-finite feature")
    return {
        "b_top2": b_top2,
        "kl_mean": klm,
        "b_mean": sum(x["b_meas"] for x in layers) / len(layers),
        "b_min": min(x["b_meas"] for x in layers),
    }


def decide_cell(D: float | None, f: dict | None) -> dict:
    out = {
        "check": None if D is None else ("allow" if D < TA else ("alarm" if D > TR else "fallback"))
    }
    out["top2"] = None if f is None else ("refuse" if f["b_top2"] <= TOP2_MAX_BITS else "allow")
    out["klm"] = None if f is None else ("refuse" if f["kl_mean"] >= KLM_MIN else "allow")
    return out


def decide(probe_dir: Path, d_json: Path, out: Path) -> None:
    dvals = json.load(open(d_json, encoding="utf-8"))  # {tag: {cfg: D}}
    res = {}
    for tag, dd in sorted(dvals.items()):
        f = probe_dir / tag / "probe.json"
        pj = json.load(open(f, encoding="utf-8")) if f.exists() else None
        if pj is not None and pj["layers_captured"] != pj["layers_config"]:
            pj = None
        for cfg in CFGS:
            try:
                ft = features(pj, cfg) if pj else None
            except (
                Exception
            ) as e:  # a malformed probe is void for P_top2 / P_klm, never a crash of the whole step
                print(f"void {tag} {cfg}: {e!r}", file=sys.stderr)
                ft = None
            dec = decide_cell(dd.get(cfg), ft)
            dec["family"] = "refuse" if tag in QWEN2_TAGS else "allow"
            res[f"{tag}|{cfg}"] = {
                "tag": tag,
                "cfg": cfg,
                "D": dd.get(cfg),
                "features": ft,
                "decision": dec,
                "probe_void": ft is None,
            }
    Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("| model | cell | D | check | b_top2 | P_top2 | kl_mean | P_klm |")
    print("|---|---|---|---|---|---|---|---|")
    for v in res.values():
        f, d = v["features"] or {}, v["decision"]
        D = "-" if v["D"] is None else f"{v['D']:.4f}"
        bt = f"{f['b_top2']:.3f}" if f else "void"
        km = f"{f['kl_mean']:.4f}" if f else "void"
        print(
            f"| {v['tag']} | {v['cfg']} | {D} | {d['check']} | {bt} | {d['top2']} | {km} | {d['klm']} |"
        )


def metrics(rows: list[tuple[str | None, int]]) -> dict:
    rows = [(a, y) for a, y in rows if a is not None and y is not None]
    dmg = [(a, y) for a, y in rows if y > 0]
    ok = [(a, y) for a, y in rows if y == 0]
    mis = sum(1 for a, _ in dmg if a == "allow")
    mis_c = sum(1 for a, y in dmg if a == "allow" and y == 2)
    fa = sum(1 for a, _ in ok if a != "allow")
    ba = (
        ((len(dmg) - mis) / len(dmg) + (len(ok) - fa) / len(ok)) / 2 if dmg and ok else float("nan")
    )
    return {
        "n": len(rows),
        "damaged": len(dmg),
        "misclear": mis,
        "misclear_collapse": mis_c,
        "false_alarm": fa,
        "bal_acc": ba,
        "cost": 5 * mis + fa,
    }


def score(dec_json: Path, lab_json: Path, out: Path) -> None:
    dec = json.load(open(dec_json, encoding="utf-8"))
    lab = json.load(open(lab_json, encoding="utf-8"))  # {"tag|cfg": {"y": 0/1/2 or null, ...}}
    ymap, unusable = {}, {}
    for k, v in lab.items():
        bad = v.get("f16_avg") is not None and (
            v["f16_avg"] < F16_MIN_AVG or v.get("f16_loop_rate", 0) >= F16_MAX_LOOP
        )
        if bad:
            unusable[k] = {"f16_avg": v["f16_avg"], "f16_loop_rate": v.get("f16_loop_rate")}
        ymap[k] = None if bad else v.get("y")
    res = {
        "unusable_labels": unusable,
        "undecidable_or_missing": [k for k in dec if k not in unusable and ymap.get(k) is None],
        "probe_void": [k for k in dec if dec[k]["probe_void"]],
    }
    for p in PREDS:
        res[p] = metrics([(dec[k]["decision"][p], ymap.get(k)) for k in dec])
    # common cells: decided by check, top2 and klm and labelled
    common = [
        k
        for k in dec
        if ymap.get(k) is not None
        and all(dec[k]["decision"][p] is not None for p in ("check", "top2", "klm"))
    ]
    res["common_cells"] = common
    res["common"] = {p: metrics([(dec[k]["decision"][p], ymap[k]) for k in common]) for p in PREDS}
    c = res["common"]
    n_col = sum(1 for k in common if ymap[k] == 2)
    n_dmg, n_ok = c["check"]["damaged"], c["check"]["n"] - c["check"]["damaged"]
    n_q2 = sum(1 for k in common if dec[k]["tag"] in QWEN2_TAGS)
    counts = (
        f"[common {len(common)} cells: {n_col} collapse, {n_dmg} damaged, {n_ok} lossless, {n_q2} Qwen2-family; "
        f"left out: {len(res['undecidable_or_missing'])} undecidable / missing, {len(unusable)} unusable labels, "
        f"{len(res['probe_void'])} probe-void]"
    )
    v = {}
    if n_col == 0:
        v["PP1"] = f"UNTESTED (no collapse cell) {counts}"
    else:
        v["PP1"] = (
            f"P_top2 allows {c['top2']['misclear_collapse']} of {n_col} collapse cells -> "
            + ("PASS" if c["top2"]["misclear_collapse"] == 0 else "FAIL")
            + f" {counts}"
        )
    for name, p in (("PP2", "top2"), ("PP3", "klm")):
        a, b = c[p]["bal_acc"], c["check"]["bal_acc"]
        if math.isnan(a) or math.isnan(b):
            v[name] = f"UNTESTED (one class missing) {counts}"
            continue
        ok = a >= b - 0.05 - 1e-9
        v[name] = (
            f"P_{p} BA {a:.3f} (misclear {c[p]['misclear']}, false alarm {c[p]['false_alarm']}) vs P_check "
            f"{b:.3f} (misclear {c['check']['misclear']}, false alarm {c['check']['false_alarm']}); "
            f"P_family baseline {c['family']['bal_acc']:.3f} -> "
            + ("PASS" if ok else "FAIL")
            + f" {counts}"
        )
    res["verdict"] = v
    Path(out).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: res["common"][k] for k in res["common"]}, indent=1, default=str))
    print("unusable labels:", unusable, "probe void:", res["probe_void"])
    for k, s in v.items():
        print(f"VERDICT {k}: {s}")


if __name__ == "__main__":
    if sys.argv[1] == "decide":
        decide(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    elif sys.argv[1] == "score":
        score(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        raise SystemExit(__doc__)
