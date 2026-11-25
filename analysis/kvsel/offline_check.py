# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""Offline check of the kvsel selector on existing data (no GPU, reads stored results only).

A. SpinTare cells (llama.cpp K formats, LongBench labels): fitting set (60 natural cells), held-out
   (PRED-P 20 + N13 12 usable cells), the blind batch N1 and the injected-dose batch N4. The flat
   decision of kvsel for a cell is "flag" iff b_top2 <= tau_b in the cell's engine basis.
   Loaders and
   filters are SpinTare's analysis/spintare/z_analyses.py (definitions executed unchanged, as in
   the tokenizer re-check of the census). Also: wrong-basis sensitivity and the serving policy
   (flag -> 8-bit fallback) on the PRED-P cells.
B. Whole-selector scenarios on every probed model (descriptive).
Usage: python -I -X utf8 offline_check.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from kvsel import evidence as ev  # noqa: E402
from kvsel.metrics import ModelMetrics, config_check, from_probe  # noqa: E402
from kvsel.select import Deployment, Engine, select  # noqa: E402

RES = ev.SPINTARE_REPO / "results" / "spintare"
INPUTS = REPO / "results" / "kvsel" / "inputs"  # HF configs of the probed models
FK = {"q4_0": "hadamard/q4_0", "q4_0norot": "identity/q4_0", "q4_1": "hadamard/q4_1"}
WRONG = {"q4_0": "identity/q4_0", "q4_0norot": "hadamard/q4_0", "q4_1": "identity/q4_1"}


def z_defs() -> dict[str, Any]:
    """Execute the definition part of z_analyses.py (loaders, fit_bits, lofo, topk_energy)."""
    os.environ["KVREL_RESULTS"] = str(ev.SPINTARE_REPO / "results")
    src_path = ev.SPINTARE_REPO / "analysis" / "spintare" / "z_analyses.py"
    src = src_path.read_text(encoding="utf-8")
    hit = re.search(r'print\(\s*"# Final-round', src)  # the call is split over lines in the repo
    if hit is None:
        raise SystemExit("cut point not found in z_analyses.py")
    cut = hit.start()
    g: dict[str, Any] = {"__file__": str(src_path), "__name__": "z_defs"}
    exec(compile(src[:cut], str(src_path), "exec"), g)
    return g


def load_batch(g: dict[str, Any], name: str) -> list[dict[str, Any]]:
    g["PP"] = RES / "prospective" / name
    return g["load_pp"]()


def usable(c: dict[str, Any]) -> bool:
    """z_analyses pp_ok filter: decided label, usable 16-bit baseline."""
    return c["y"] is not None and not (
        c.get("f16_avg") is not None and (c["f16_avg"] < 5 or c["f16_loop"] >= 0.2))


def probe_path(c: dict[str, Any], batch: str) -> Path:
    if batch == "fit":
        return RES / "detection_gate" / "results" / c["tag"] / "probe.json"
    return RES / "prospective" / batch / "results" / c["tag"] / "probe.json"


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dmg = [r for r in rows if r["y"] > 0]
    safe = [r for r in rows if r["y"] == 0]
    miss = [r for r in dmg if not r["flag"]]
    fa = [r for r in safe if r["flag"]]
    ba = float("nan")
    if dmg and safe:
        ba = (1 - len(miss) / len(dmg) + 1 - len(fa) / len(safe)) / 2
    return {"n": len(rows), "damaged": len(dmg), "misclear": len(miss),
            "misclear_collapse": sum(r["y"] == 2 for r in miss), "false_alarm": len(fa), "ba": ba}


def part_a(g: dict[str, Any], tau: float) -> dict[str, Any]:
    sets: dict[str, list[tuple[dict[str, Any], str]]] = {
        "fit (60 natural cells)": [(c, "fit") for c in g["nat"]],
        "held-out PRED-P (usable)": [(c, "predp") for c in g["pp"] if usable(c)],
        "held-out N13 (usable)": [(c, "n13") for c in load_batch(g, "n13") if usable(c)],
        "blind N1 (usable)": [(c, "n1") for c in load_batch(g, "n1") if usable(c)],
        "injected N4 (usable)": [(c, "n4") for c in load_batch(g, "n4") if usable(c)],
    }
    out: dict[str, Any] = {}
    cache: dict[Path, ModelMetrics] = {}
    max_dev = 0.0
    for name, cells in sets.items():
        rows = []
        for c, batch in cells:
            p = probe_path(c, batch)
            if p not in cache:
                cache[p] = from_probe(p, name=c["tag"])
            m = cache[p]
            b = m.b_top2[FK[c["cfg"]]]
            ref = g["topk_energy"](c["L"]["bm"], 2)  # z_analyses' own value for the same cell
            max_dev = max(max_dev, abs(b - ref))
            bw = m.b_top2[WRONG[c["cfg"]]]
            rows.append({"model": c["model"], "tag": c["tag"], "cfg": c["cfg"], "y": c["y"],
                         "D": c["D"], "diff": c.get("diff"), "f16_avg": c.get("f16_avg"),
                         "b_top2": b, "flag": b <= tau, "flag_wrong_basis": bw <= tau,
                         "bias_stat": m.bias_stat, "rot_dim": m.rot_dim, "head_dim": m.head_dim,
                         "kvh": m.n_kv_heads, "interleaved": m.interleaved_rope,
                         "worst_layers": m.worst_layers[FK[c["cfg"]]]})
        wrong = [dict(r, flag=r["flag_wrong_basis"]) for r in rows]
        out[name] = {"score": score(rows), "score_wrong_basis": score(wrong),
                     "errors": [r for r in rows if (r["y"] > 0) != r["flag"]],
                     "basis_flips": sum(r["flag"] != r["flag_wrong_basis"] for r in rows),
                     "rows": rows}
    lofo = g["lofo"](g["nat"], lambda c: g["topk_energy"](c["L"]["bm"], 2))
    out["_lofo_fit"] = {"ba": lofo[0], "misclear": lofo[1], "misclear_collapse": lofo[2],
                        "false_alarm": lofo[3]}
    f2 = lambda c: g["topk_energy"](c["L"]["bm"], 2)  # noqa: E731
    out["_insample_fit"] = {"tau_refit": g["fit_bits"](g["nat"], f2),
                            "ba_at_frozen_tau": g["ba_of"]([(f2(c) <= tau, c["y"] > 0)
                                                            for c in g["nat"]])}
    out["_max_dev_vs_z_analyses"] = max_dev
    # serving policy: PRED-P cells with a usable baseline and a task diff;
    # this includes 4 undecided cells (y None) - their measured diff is used as is
    pp_all = [c for c in g["pp"] if c.get("diff") is not None and not (
        c.get("f16_avg") is not None and (c["f16_avg"] < 5 or c["f16_loop"] >= 0.2))]
    srv = []
    for c in pp_all:
        m = cache.get(probe_path(c, "predp")) or from_probe(probe_path(c, "predp"))
        flag = m.b_top2[FK[c["cfg"]]] <= tau
        srv.append({"cell": f"{c['tag']}|{c['cfg']}", "diff": c["diff"], "flag": flag,
                    "q41": c["cfg"] == "q4_1", "undecided": c["y"] is None})
    n = len(srv)
    if n:
        def pol_row(delta: list[float], bits: list[float]) -> dict[str, float]:
            return {"mean_delta": sum(delta) / n, "bits": sum(bits) / n}
        out["_serving"] = {
            "n_cells": n,
            "n_undecided_cells": sum(s["undecided"] for s in srv),
            "convention": "every 4-bit cell counted as 4.5 bits (q4_1 too); 8-bit cells "
                          "are assumed to lose 0 points",
            "always_4bit (4.5-bit convention)": pol_row([s["diff"] for s in srv], [4.5] * n),
            "always_4bit, q4_1 counted 5.0": pol_row(
                [s["diff"] for s in srv], [5.0 if s["q41"] else 4.5 for s in srv]),
            "always_8bit": pol_row([0.0] * n, [8.5] * n),
            "flag -> q8_0 (SpinTare check, 8-bit fallback; kvsel on upstream llama.cpp)": pol_row(
                [0.0 if s["flag"] else s["diff"] for s in srv],
                [8.5 if s["flag"] else 4.5 for s in srv]),
            "flag -> q8_0, q4_1 counted 5.0": pol_row(
                [0.0 if s["flag"] else s["diff"] for s in srv],
                [8.5 if s["flag"] else (5.0 if s["q41"] else 4.5) for s in srv]),
            "four_bit_cells": sum(not s["flag"] for s in srv),
            "kvsel_patched (flag -> q4 + SpinTare)": "task delta of SpinTare not measured on "
                                                    "these cells (to test)",
        }
    return out


# HF config.json of every probed model (local checkpoint copies, dumped 2026-11-24, see
# "_meta"), matched by the probe's model directory name. Supplies NoPE / sliding-window layers,
# model_type and the RoPE scaling type through the same parser the CLI uses (metrics._arch_flags).
CFGS: dict[str, Any] = json.loads((INPUTS / "model_configs.json").read_text(encoding="utf-8"))


def metrics_with_config(p: Path, name: str) -> ModelMetrics:
    mdir = Path(json.loads(p.read_text(encoding="utf-8"))["model"]).name
    cfg = CFGS.get(mdir)
    if cfg is None or mdir == "_meta":
        raise SystemExit(f"{name}: model dir {mdir} not in model_configs.json")
    return from_probe(p, name=name, config=cfg)


# ---------------------------------------------------------------- part B
def part_b() -> list[dict[str, Any]]:
    seen: dict[str, Path] = {}
    for d in (RES / "detection_gate" / "results").iterdir():
        if "inj" not in d.name and (d / "probe.json").exists():
            seen[d.name] = d / "probe.json"
    for b in ("predp", "n13", "n1"):
        for d in (RES / "prospective" / b / "results").iterdir():
            if (d / "probe.json").exists():
                seen.setdefault(d.name, d / "probe.json")
    scen = {
        "llama.cpp upstream, 32K, 6 GiB": Deployment([Engine("llama.cpp")], 32768,
                                                     gpu_kv_gib=6.0),
        "llama.cpp + SpinTare, 32K, 6 GiB": Deployment([Engine("llama.cpp", spintare_kernel=True)],
                                                       32768, gpu_kv_gib=6.0),
        "vLLM + SpinTare plugin, 32K, 6 GiB": Deployment([Engine("vllm", spintare_kernel=True)],
                                                         32768, gpu_kv_gib=6.0),
    }
    out = []
    for tag, p in sorted(seen.items()):
        m = metrics_with_config(p, tag)
        row: dict[str, Any] = {"model": tag, "model_type": m.model_type, "rope": m.rope_type,
                               "d": m.head_dim, "L": m.n_layers,
                               "kvh": m.n_kv_heads, "gqa": m.gqa_ratio, "rot_dim": m.rot_dim,
                               "b_top2_had_q4_0": round(m.b_top2["hadamard/q4_0"], 3)}
        for sname, dep in scen.items():
            s = select(m, dep)
            c = s.chosen
            if c is None:
                row[sname] = "none"
            else:
                row[sname] = f"{c.k_format}{'+' + c.centring if c.centring != 'none' else ''}"
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "results" / "kvsel" / "offline_check")
    a = ap.parse_args()
    tau = float(config_check().TAU_B)
    g = z_defs()
    res = {"tau_b": tau, "A": part_a(g, tau), "B": part_b()}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "offline_check.json").write_text(json.dumps(res, indent=1, ensure_ascii=False,
                                                         default=str), encoding="utf-8")
    A = res["A"]
    print(f"tau_b {tau:.4f}; max |b_top2 kvsel - z_analyses| = {A['_max_dev_vs_z_analyses']:.2e}")
    print(f"fit LOFO (z_analyses.lofo): {A['_lofo_fit']}; in-sample {A['_insample_fit']}")
    for k, v in A.items():
        if k.startswith("_"):
            continue
        print(f"\n## {k}: {v['score']}  wrong basis: {v['score_wrong_basis']}  "
              f"basis flips {v['basis_flips']}")
        for r in v["errors"]:
            print(f"  ERR {r['model']} {r['cfg']} y={r['y']} b_top2={r['b_top2']:.3f} "
                  f"D={r['D']} diff={r['diff']} f16={r['f16_avg']} bias={r['bias_stat']} "
                  f"rot={r['rot_dim']}/{r['head_dim']} kvh={r['kvh']} il={r['interleaved']}")
    print(f"\n## serving policy: {json.dumps(A.get('_serving'), indent=1)}")
    print("\n## scenarios")
    for r in res["B"]:
        print("  ", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
