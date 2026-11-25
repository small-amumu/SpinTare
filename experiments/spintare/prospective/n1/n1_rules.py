# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""N1: blind-selected non-Qwen2 prospective set. Selection, frozen decisions and scoring.

Selection (registered before any N1 probe): per candidate model, m = min over q4_0 / q4_0norot /
q4_1 of b_top2 (PRED-P features()). Candidates whose conversion, engine D or probe failed, or whose
probe is void, are left out. The SELECT_N models with the lowest m are selected (ties by tag); no
replacement afterwards, whatever their labels. Rules (thresholds frozen before N1; natural-only fit,
PRED-P amendment 1):
  P_top2n  refuse iff b_top2 <= 2.0280476994784027                 (primary)
  P_check  engine D: < 0.1016 allow, 0.1016-0.3577 fall back, > 0.3577 alarm   (engine baseline)
  P_family refuse every cell of a Qwen2-family model below 14B: none in N1, so it allows every cell
  (baseline)
  P_top2   pooled threshold 2.342775337149283                        (descriptive)
Label validity as in PRED-P (f16 average >= 5 and f16 loop rate < 20%).
Usage:
  python n1_rules.py select <probe_dir> <d_json> <models_txt> <out_txt>
  python n1_rules.py decide <probe_dir> <d_json> <selected_txt> <out_json>
  python n1_rules.py score  <decisions_json> <labels_json> <out_json>
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pp_rules import CFGS, F16_MIN_AVG, F16_MAX_LOOP, TA, TR, features, metrics  # noqa: E402

TOP2N_MAX_BITS = 2.0280476994784027
TOP2_MAX_BITS = 2.342775337149283
SELECT_N = 9
PREDS = ("top2n", "check", "family", "top2")


def load_probe(probe_dir: Path, tag: str) -> dict | None:
    f = probe_dir / tag / "probe.json"
    if not f.exists():
        return None
    pj = json.load(open(f, encoding="utf-8"))
    return pj if pj["layers_captured"] == pj["layers_config"] else None


def cell_features(pj: dict | None, cfg: str) -> dict | None:
    if pj is None:
        return None
    try:
        return features(pj, cfg)
    except Exception as e:
        print(f"void {cfg}: {e!r}", file=sys.stderr)
        return None


def tags_of(models_txt: Path) -> list[str]:
    rows = [ln.split("|")[0].strip() for ln in open(models_txt, encoding="utf-8")]
    return [t for t in rows if t and not t.startswith("#")]


def select(probe_dir: Path, d_json: Path, models_txt: Path, out: Path) -> None:
    try:
        dvals = json.load(open(d_json, encoding="utf-8"))
    except Exception:  # review r38 S-3: an unreadable D_all means every D is missing, not a crash
        dvals = {}
    ranked, dropped = [], []
    for tag in tags_of(models_txt):
        pj = load_probe(probe_dir, tag)
        fs = {c: cell_features(pj, c) for c in CFGS}
        dd = dvals.get(tag, {})
        why = []
        if pj is None:
            fl = probe_dir / tag / "FAILED"
            why.append(
                "probe "
                + (
                    (fl.read_text(errors="replace").splitlines() or [""])[0][:120]
                    if fl.exists()
                    else "missing"
                )
            )
        why += [f"probe void {c}" for c in CFGS if pj is not None and fs[c] is None]
        why += [f"D missing {c}" for c in CFGS if dd.get(c) is None]
        if why:
            bt = {c: round(f["b_top2"], 3) for c, f in fs.items() if f is not None}
            dropped.append(f"{tag}: {'; '.join(why)}; b_top2 {bt or '-'}")
            continue
        ranked.append((min(f["b_top2"] for f in fs.values()), tag))
    ranked.sort()
    sel = [t for _, t in ranked[:SELECT_N]]
    Path(out).write_text("".join(f"{t}\n" for t in sel), encoding="utf-8")
    print(
        f"eligible {len(ranked)}, selected {len(sel)} (rule: the {SELECT_N} lowest; all eligible when fewer)"
    )
    print()
    print("| rank | model | min b_top2 over the 3 formats | selected |")
    print("|---|---|---|---|")
    for i, (m, t) in enumerate(ranked):
        print(f"| {i + 1} | {t} | {m:.3f} | {'yes' if t in sel else 'no'} |")
    print("left out (conversion, engine D or probe failed / void):")
    for d in dropped or ["none"]:
        print(f"- {d}")


def decide(probe_dir: Path, d_json: Path, selected_txt: Path, out: Path) -> None:
    dvals = json.load(open(d_json, encoding="utf-8"))
    res = {}
    for tag in [t.strip() for t in open(selected_txt, encoding="utf-8") if t.strip()]:
        pj = load_probe(probe_dir, tag)
        for cfg in CFGS:
            f = cell_features(pj, cfg)
            D = dvals[tag].get(cfg)
            dec = {
                "top2n": None
                if f is None
                else ("refuse" if f["b_top2"] <= TOP2N_MAX_BITS else "allow"),
                "check": None
                if D is None
                else ("allow" if D < TA else ("alarm" if D > TR else "fallback")),
                "family": "allow",
                "top2": None
                if f is None
                else ("refuse" if f["b_top2"] <= TOP2_MAX_BITS else "allow"),
            }
            res[f"{tag}|{cfg}"] = {
                "tag": tag,
                "cfg": cfg,
                "D": D,
                "features": f,
                "decision": dec,
                "probe_void": f is None,
            }
    Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("| model | cell | D | P_check | b_top2 | P_top2n | P_top2 (pooled) |")
    print("|---|---|---|---|---|---|---|")
    for v in res.values():
        f, d = v["features"] or {}, v["decision"]
        D = "-" if v["D"] is None else f"{v['D']:.4f}"
        bt = f"{f['b_top2']:.3f}" if f else "void"
        print(
            f"| {v['tag']} | {v['cfg']} | {D} | {d['check']} | {bt} | {d['top2n']} | {d['top2']} |"
        )


def score(dec_json: Path, lab_json: Path, out: Path) -> None:
    dec = json.load(open(dec_json, encoding="utf-8"))
    lab = json.load(open(lab_json, encoding="utf-8"))
    ymap, unusable = {}, {}
    for k, v in lab.items():
        bad = v.get("f16_avg") is not None and (
            v["f16_avg"] < F16_MIN_AVG or v.get("f16_loop_rate", 0) >= F16_MAX_LOOP
        )
        if bad:
            unusable[k] = {"f16_avg": v["f16_avg"], "f16_loop_rate": v.get("f16_loop_rate")}
        ymap[k] = None if bad else v.get("y")
    common = [
        k
        for k in dec
        if ymap.get(k) is not None
        and all(dec[k]["decision"][p] is not None for p in ("top2n", "check"))
    ]
    res = {
        "unusable_labels": unusable,
        "common_cells": common,
        "undecidable_or_missing": [k for k in dec if k not in unusable and ymap.get(k) is None],
        "common": {p: metrics([(dec[k]["decision"][p], ymap[k]) for k in common]) for p in PREDS},
    }
    c = res["common"]
    n_col = sum(1 for k in common if ymap[k] == 2)
    n_dmg = c["check"]["damaged"]
    counts = (
        f"[common {len(common)} cells: {n_col} collapse, {n_dmg} damaged, {len(common) - n_dmg} lossless; "
        f"left out {len(res['undecidable_or_missing'])} undecidable / missing, {len(unusable)} unusable]"
    )
    v = {}
    a, b, fam = c["top2n"]["bal_acc"], c["check"]["bal_acc"], c["family"]["bal_acc"]
    # NP1 has two clauses, each PASS / FAIL / UNTESTED (review r38 M-1): (a) no collapse cell
    # allowed, UNTESTED
    # without a collapse cell; (b) BA >= P_check BA - 0.05, UNTESTED when either BA is undefined
    # (one class missing)
    ca = "UNTESTED" if n_col == 0 else ("PASS" if c["top2n"]["misclear_collapse"] == 0 else "FAIL")
    cb = (
        "UNTESTED"
        if math.isnan(a) or math.isnan(b)
        else ("PASS" if a >= b - 0.05 - 1e-9 else "FAIL")
    )
    whole = "FAIL" if "FAIL" in (ca, cb) else ("PASS" if ca == cb == "PASS" else "PARTLY UNTESTED")
    v["NP1"] = (
        f"{whole}: (a) collapse clause {ca} (collapse allowed {c['top2n']['misclear_collapse']} of {n_col}); "
        f"(b) BA clause {cb} (P_top2n BA {a:.3f}: misclear {c['top2n']['misclear']}, false alarm "
        f"{c['top2n']['false_alarm']}; P_check BA {b:.3f}: misclear {c['check']['misclear']}, false alarm "
        f"{c['check']['false_alarm']}) {counts}"
    )
    # NP2 compares with allow-all (P_family allows every N1 cell, BA 0.5 by construction; review r38
    # S-5)
    v["NP2"] = (
        "UNTESTED (one class missing)"
        if math.isnan(a) or math.isnan(fam)
        else ("PASS" if a > fam + 1e-9 else "FAIL")
    ) + (
        f": P_top2n BA {a:.3f} vs allow-all {fam:.3f}; recall "
        f"{c['top2n']['damaged'] - c['top2n']['misclear']}/{c['top2n']['damaged']}, specificity "
        f"{len(common) - c['top2n']['damaged'] - c['top2n']['false_alarm']}/"
        f"{len(common) - c['top2n']['damaged']} {counts}"
    )
    # near-threshold cells (review r38 S-6): |b_top2 - threshold| < 0.05, and BA without them
    near = [k for k in common if abs(dec[k]["features"]["b_top2"] - TOP2N_MAX_BITS) < 0.05]
    far = [k for k in common if k not in near]
    mf = metrics([(dec[k]["decision"]["top2n"], ymap[k]) for k in far])
    v["NEAR"] = (
        f"{len(near)} common cells within 0.05 bit of the threshold {near}; P_top2n BA without them "
        f"{mf['bal_acc']:.3f} (misclear {mf['misclear']}, false alarm {mf['false_alarm']}, n {mf['n']})"
    )
    res["verdict"] = v
    Path(out).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res["common"], indent=1, default=str))
    print("unusable labels:", unusable)
    for k, s in v.items():
        print(f"VERDICT {k}: {s}")


if __name__ == "__main__":
    cmd, a = sys.argv[1], [Path(x) for x in sys.argv[2:]]
    if cmd == "select":
        select(*a)
    elif cmd == "decide":
        decide(*a)
    elif cmd == "score":
        score(*a)
    else:
        raise SystemExit(__doc__)
