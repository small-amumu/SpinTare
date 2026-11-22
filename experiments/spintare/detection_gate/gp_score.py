"""Score GATE-PRED amendment 1: every predictor against the registered task labels on TRAIN-I /
TEST-N / TEST-P.

Inputs: --probe DIR (results/<tag>/probe.json from run_gp_probe.sh), --peel DIR (mac_cells.csv,
mac_prospective.csv, pr.out, lr_freeze.json), --registry CSV (gp_cells_registry.csv). Prints the
tables of the outcome entry and writes a JSON with every cell's features and decisions (--out).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

TA, TR = 0.1016, 0.3577
YMAP = {"lossless": 0, "partial": 1, "collapse": 2}
FEATKEY = {"q4_0": "hadamard/q4_0", "q4_0norot": "identity/q4_0", "q4_1": "hadamard/q4_1"}
TAG = {
    "Qwen2.5-0.5B": "qwen25_0p5b",
    "Qwen2.5-1.5B": "qwen25_1p5b",
    "Qwen2.5-3B": "qwen25_3b",
    "Qwen2.5-7B": "qwen25_7b",
    "Qwen2.5-14B": "qwen25_14b",
    "R1-Qwen-1.5B": "r1q_1p5b",
    "R1-Qwen-7B": "r1q_7b",
    "Qwen3-0.6B": "qwen3_0p6b",
    "Qwen3-1.7B": "qwen3_1p7b",
    "Qwen3-4B": "qwen3_4b",
    "Qwen3-8B": "qwen3_8b",
    "Qwen3-14B": "qwen3_14b",
    "Llama-3.2-1B": "llama32_1b",
    "Llama-3.2-3B": "llama32_3b",
    "Llama-3.1-8B": "llama31_8b",
    "gemma-2-2b": "gemma2_2b",
    "SmolLM2-1.7B": "smollm2_1p7b",
    "Yi-1.5-6B": "yi15_6b",
    "StableLM-2-1.6B": "stablelm2",
    "Pythia-1.4B": "pythia_1p4b",
    "StarCoder2-7B": "starcoder2_7b",
    "deepseek-llm-7b-chat": "dsllm7b",
    "OLMo-2-1124-7B": "olmo2_7b",
    "Falcon3-7B": "falcon3_7b",
    "granite-3.3-8B": "granite33_8b",
    "Qwen2-7B": "qwen2_7b",
    "Qwen2.5-Coder-7B": "qwen25coder_7b",
    "Qwen2.5-Math-7B": "qwen25math_7b",
    "gemma-3-1b": "gemma3_1b",
}
# dose-0 base models of the injected families: their natural cells train the thresholds (review r31
# M-2), so the
# Llama family has lossless and damaged cells and the threshold carries dose information, not family
# identity
TRAIN_BASE = {"Llama-3.1-8B": "llama31", "gemma-2-2b": "gemma2", "Mistral-7B": "mistral"}
NEED_BOTH = (
    "llama31",
    "mistral",
)  # families that must have lossless and damaged probed cells (gemma has no damaged)
G4_MODELS = [
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen2.5-14B",
    "Yi-1.5-6B",
    "StableLM-2-1.6B",
]


def inj_tag(name: str) -> str | None:
    m = re.match(r"Llama-3\.1-8B-inj(\w+)$", name)
    if m:
        return f"llama31_8b_inj{m.group(1)}"
    m = re.match(r"Mistral-7B-inj(\w+)$", name)
    if m:
        return "mistral7b" if m.group(1) == "0" else f"mistral7b_inj{m.group(1)}"
    m = re.match(r"T6m-gemma-2-2b-inj(\d+)$", name)
    if m and int(m.group(1)) > 0:
        return f"gemma2_2b_inj{m.group(1)}"
    return None


def load_cells(peel: Path, reg: Path) -> list[dict]:
    train_check = {
        (r["model"], r["cfg"])
        for r in json.load(open(peel / "lr_freeze.json", encoding="utf-8"))["loo_detail"]["M1"]
    }
    cells: list[dict] = []
    for r in csv.DictReader(open(peel / "mac_cells.csv", encoding="utf-8")):
        if r["label"] in YMAP:
            split = "TRAIN-I" if r["model"] in TRAIN_BASE else "TEST-N"
            cells.append(
                {
                    "split": split,
                    "model": r["model"],
                    "cfg": r["cfg"],
                    "D": float(r["D"]),
                    "y": YMAP[r["label"]],
                    "r": float(r["rblk"]) if r["rblk"] else None,
                    "check_insample": (r["model"], r["cfg"]) in train_check,
                    "src": r["source"],
                }
            )
    for r in csv.DictReader(open(peel / "mac_prospective.csv", encoding="utf-8")):
        if r["label"] not in YMAP:
            continue
        if r["model"].startswith("T6m-"):
            t = inj_tag(r["model"])
            if t:
                cells.append(
                    {
                        "split": "TRAIN-I",
                        "model": r["model"],
                        "tag": t,
                        "cfg": r["cfg"],
                        "D": float(r["D"]),
                        "y": YMAP[r["label"]],
                        "r": None,
                        "src": r["source"],
                    }
                )
            continue
        cells.append(
            {
                "split": "TEST-P",
                "model": r["model"],
                "cfg": r["cfg"],
                "D": float(r["D"]),
                "y": YMAP[r["label"]],
                "r": None,
                "src": r["source"],
            }
        )
    for line in (peel / "pr.out").read_text(encoding="utf-8").splitlines():
        if line.startswith("| ") and "prospective (LongBench, injection)" in line:
            p = [x.strip() for x in line.split("|")]
            t = inj_tag(p[1])
            if p[4] in YMAP and t:
                cells.append(
                    {
                        "split": "TRAIN-I",
                        "model": p[1],
                        "tag": t,
                        "cfg": p[2],
                        "D": float(p[3]),
                        "y": YMAP[p[4]],
                        "r": None,
                        "src": "pr.out",
                    }
                )
    for r in csv.DictReader(open(reg, encoding="utf-8")):
        cells.append(
            {
                "split": "TEST-P",
                "model": r["model"],
                "cfg": r["cfg"],
                "D": float(r["D"]),
                "y": YMAP[r["label"]],
                "r": float(r["r"]) if r["r"] else None,
                "src": r["source"],
            }
        )
    seen = set()
    for c in cells:
        key = (c["model"], c["cfg"])
        assert key not in seen, f"duplicate cell {key}"
        seen.add(key)
        c.setdefault("tag", TAG.get(c["model"]))
        c["family"] = None
        if c["split"] == "TRAIN-I":
            c["family"] = next(
                f
                for m, f in TRAIN_BASE.items()
                if c["model"].startswith(m) or c["model"].startswith("T6m-" + m)
            )
    return cells


def attach(cells: list[dict], probe: Path) -> list[str]:
    voids = []
    cache: dict[str, dict | None] = {}
    for c in cells:
        t = c["tag"]
        if t not in cache:
            f = probe / str(t) / "probe.json"
            cache[t] = json.loads(f.read_text(encoding="utf-8")) if t and f.exists() else None
            if cache[t] and cache[t]["layers_captured"] != cache[t]["layers_config"]:
                voids.append(
                    f"{t}: captured {cache[t]['layers_captured']} of {cache[t]['layers_config']} layers"
                )
                cache[t] = None
        if (
            cache[t]
            and "_inj" in str(t)
            and not (cache[t]["weights"]["has_k_bias"] and cache[t]["weights"]["bias_stat"] > 0)
        ):
            voids.append(f"{t}: injected model without a loaded key bias")
            cache[t] = None
        pj = cache[t]
        c["feat"] = None
        if pj is None:
            fr = probe / str(t) / "FAILED"
            voids.append(
                f"{c['model']} {c['cfg']}: no probe ({fr.read_text().strip() if fr.exists() else 'not run'})"
            )
            continue
        k = FEATKEY[c["cfg"]]
        s = pj["summary"]
        c["feat"] = {
            "b_meas": s[f"{k}/b_meas_mean"],
            "b_meas_min": s[f"{k}/b_meas_min"],
            "b_eff": s[f"{k}/b_eff_mean"],
            "b_eff_min": s[f"{k}/b_eff_min"],
            "attn_kl": s[f"{k}/attn_kl_mean"],
            "key_mse": s[f"{k}/key_rel_mse_mean"],
            "cos_min": s[f"{k}/out_cos_min"],
            "bias_stat": pj["weights"]["bias_stat"],
            "has_bias": pj["weights"]["has_k_bias"],
            "sparsity": {
                x: s[f"attn/{x}_mean"]
                for x in ("norm_entropy", "top10_mass", "diffuse_head_frac", "sink_mass")
            },
        }
    return voids


def fit_threshold(train: list[dict], key: str, higher_is_safe: bool) -> float | None:
    pts = [(c["feat"][key], c["y"] > 0) for c in train if c["feat"]]
    if not pts or all(d for _, d in pts) or not any(d for _, d in pts):
        return None
    best = None
    for t, _ in sorted(pts):
        allow = [(v >= t) if higher_is_safe else (v <= t) for v, _ in pts]
        sens = sum(1 for a, (_, d) in zip(allow, pts) if d and not a) / sum(1 for _, d in pts if d)
        spec = sum(1 for a, (_, d) in zip(allow, pts) if not d and a) / sum(
            1 for _, d in pts if not d
        )
        ba = (sens + spec) / 2
        tie = t if higher_is_safe else -t
        if best is None or ba > best[0] + 1e-12 or (abs(ba - best[0]) <= 1e-12 and tie > best[1]):
            best = (ba, tie, t)
    return best[2]


def decide(c: dict, th: dict) -> dict:
    f = c["feat"]
    d: dict[str, str | None] = {}
    D = c["D"]
    d["check"] = "allow" if D < TA else ("alarm" if D > TR else "fallback")
    d["mlx_mean"] = "allow" if D <= 0.01 else "refuse"
    r = c.get("r")
    d["ratio"] = None if r is None else ("allow" if r <= 3 else ("alarm" if r > 15 else "fallback"))
    if f is None:
        d.update({"beff": None, "bias": None, "cos": None, "akl": None, "kmse": None})
        return d
    d["beff"] = None if th["beff"] is None else ("allow" if f["b_meas"] >= th["beff"] else "refuse")
    d["bias"] = "refuse" if f["bias_stat"] > 100 else "allow"
    d["cos"] = "allow" if f["cos_min"] >= 0.998 else "refuse"
    d["akl"] = None if th["akl"] is None else ("allow" if f["attn_kl"] <= th["akl"] else "refuse")
    d["kmse"] = (
        None if th["kmse"] is None else ("allow" if f["key_mse"] <= th["kmse"] else "refuse")
    )
    return d


def family_table(train: list[dict]) -> dict:
    tab: dict = {}
    for c in train:
        if not c["feat"]:
            continue
        f = tab.setdefault(c["family"], {"lossless": 0, "damaged": 0})
        f["damaged" if c["y"] > 0 else "lossless"] += 1
    return tab


def metrics(cells: list[dict], name: str) -> dict:
    rows = [(c["dec"][name] == "allow", c["y"]) for c in cells if c["dec"].get(name) is not None]
    n = len(rows)
    dmg = [a for a, y in rows if y > 0]
    ok = [a for a, y in rows if y == 0]
    mis = sum(dmg)
    fa = sum(1 for a in ok if not a)
    sens = (len(dmg) - mis) / len(dmg) if dmg else float("nan")
    spec = (len(ok) - fa) / len(ok) if ok else float("nan")
    mis_col = sum(1 for c in cells if c["dec"].get(name) == "allow" and c["y"] == 2)
    return {
        "n": n,
        "damaged": len(dmg),
        "misclear": mis,
        "misclear_collapse": mis_col,
        "false_alarm": fa,
        "bal_acc": (sens + spec) / 2 if dmg and ok else float("nan"),
        "cost": 5 * mis + fa,
    }


def mcnemar(cells: list[dict], a: str, b: str) -> dict:
    def right(c: dict, k: str) -> bool:
        return (c["dec"][k] == "allow") == (c["y"] == 0)

    both = [c for c in cells if c["dec"].get(a) is not None and c["dec"].get(b) is not None]
    n01 = sum(1 for c in both if right(c, a) and not right(c, b))
    n10 = sum(1 for c in both if right(c, b) and not right(c, a))
    m = n01 + n10
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(0, min(n01, n10) + 1)) / 2**m) if m else 1.0
    return {
        "n": len(both),
        f"{a}_only_right": n01,
        f"{b}_only_right": n10,
        "p_exact": p,
        "significance_claimable": len(both) >= 30,
    }


def by_model(cells: list[dict]) -> list[dict]:
    """One pseudo-cell per model: its worst label and, per predictor, 'allow' only if the predictor
    allows the majority of the model's cells it decides (review r31 S-5: the formats of one model
    are not independent)."""
    groups: dict[str, list[dict]] = {}
    for c in cells:
        groups.setdefault(c["model"], []).append(c)
    out = []
    for m, cs in groups.items():
        dec: dict[str, str | None] = {}
        for k in cs[0]["dec"]:
            vs = [c["dec"][k] for c in cs if c["dec"].get(k) is not None]
            dec[k] = (
                None
                if not vs
                else ("allow" if 2 * sum(v == "allow" for v in vs) > len(vs) else "refuse")
            )
        out.append({"model": m, "y": max(c["y"] for c in cs), "dec": dec})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--peel", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cells = load_cells(Path(a.peel), Path(a.registry))
    voids = attach(cells, Path(a.probe))
    train = [c for c in cells if c["split"] == "TRAIN-I"]
    ftab = family_table(train)
    print("TRAIN-I class x family (probed cells):", ftab)
    for fam in NEED_BOTH:
        if not (ftab.get(fam, {}).get("lossless") and ftab.get(fam, {}).get("damaged")):
            raise SystemExit(f"TRAIN-I: family {fam} lacks a probed lossless or damaged cell")
    th = {
        "beff": fit_threshold(train, "b_meas", True),
        "akl": fit_threshold(train, "attn_kl", False),
        "kmse": fit_threshold(train, "key_mse", False),
    }
    for c in cells:
        c["dec"] = decide(c, th)
    print("thresholds fitted on TRAIN-I:", th)
    tr = sorted((c["feat"]["b_meas"], c["y"], c["model"], c["cfg"]) for c in train if c["feat"])
    print("TRAIN-I b_meas order:", [(round(v, 3), y, m, f) for v, y, m, f in tr])
    lo_b = (
        max((v for v, y, _, _ in tr if y > 0 and v < th["beff"]), default=None)
        if th["beff"] is not None
        else None
    )
    sens_b = {"upper (registered)": th["beff"]}
    if lo_b is not None:
        sens_b.update({"lower": lo_b + 1e-9, "middle": (lo_b + th["beff"]) / 2})
    print(
        "b* interval of equal TRAIN-I balanced accuracy (lower end = just above the highest damaged b_meas below b*):",
        sens_b,
    )
    print(
        "counts:",
        {s: sum(1 for c in cells if c["split"] == s) for s in ("TRAIN-I", "TEST-N", "TEST-P")},
    )
    print("voids:", *voids, sep="\n  ")
    preds = ["check", "beff", "bias", "ratio", "mlx_mean", "cos", "akl", "kmse"]
    sets = {
        "TRAIN-I": train,
        "TEST-N all": [c for c in cells if c["split"] == "TEST-N"],
        "TEST-N out-of-sample + TEST-P": [
            c
            for c in cells
            if (c["split"] == "TEST-N" and not c["check_insample"]) or c["split"] == "TEST-P"
        ],
        "TEST-P": [c for c in cells if c["split"] == "TEST-P"],
    }
    probe_preds = ["beff", "bias", "cos", "akl", "kmse"]
    # G2' reads only this table: cells that every probe-based predictor and the check decide (review
    # r31 S-1)
    sets["COMMON (G2')"] = [
        c
        for c in sets["TEST-N out-of-sample + TEST-P"]
        if all(c["dec"].get(k) is not None for k in probe_preds + ["check"])
    ]
    out: dict = {"thresholds": th, "voids": voids, "tables": {}, "mcnemar": {}}
    for sname, cs in sets.items():
        print(f"\n## {sname}")
        print(
            "| predictor | n | damaged | misclear | misclear collapse | false alarm | bal. acc. | cost |"
        )
        print("|---|---|---|---|---|---|---|---|")
        out["tables"][sname] = {}
        for p in preds:
            m = metrics(cs, p)
            out["tables"][sname][p] = m
            print(
                f"| {p} | {m['n']} | {m['damaged']} | {m['misclear']} | {m['misclear_collapse']} | "
                f"{m['false_alarm']} | "
                f"{m['bal_acc']:.3f} | {m['cost']} |"
            )
        out["mcnemar"][sname] = {p: mcnemar(cs, p, "check") for p in preds if p != "check"}
        out["mcnemar"][sname + " (by model)"] = {
            p: mcnemar(by_model(cs), p, "check") for p in preds if p != "check"
        }
        print("McNemar vs check (cells):", json.dumps(out["mcnemar"][sname]))
        print("McNemar vs check (by model):", json.dumps(out["mcnemar"][sname + " (by model)"]))
    # G3': collapse cells of models without a key bias
    g3 = [
        c
        for c in cells
        if c["split"] != "TRAIN-I" and c["y"] == 2 and c["feat"] and not c["feat"]["has_bias"]
    ]
    print(
        "\nG3' no-bias collapse cells:",
        [(c["model"], c["cfg"], c["dec"]["bias"], c["dec"]["check"], c["dec"]["beff"]) for c in g3],
    )
    # G4': ordering of the middle-range models by q4_0 b_meas against their worst label
    mid = {}
    for c in cells:
        if c["model"] in G4_MODELS and c["feat"]:
            w = mid.setdefault(c["model"], {"worst": 0, "b": None})
            w["worst"] = max(w["worst"], c["y"])
            if c["cfg"] == "q4_0":
                w["b"] = c["feat"]["b_meas"]
    names = [m for m in G4_MODELS if m in mid and mid[m]["b"] is not None]
    pairs = [
        (x, y)
        for i, x in enumerate(names)
        for y in names[i + 1 :]
        if mid[x]["worst"] != mid[y]["worst"]
    ]
    good = sum(
        1 for x, y in pairs if (mid[x]["b"] < mid[y]["b"]) == (mid[x]["worst"] > mid[y]["worst"])
    )
    print(f"G4' ordered pairs: {good} / {len(pairs)}; per model:", {m: mid[m] for m in names})
    out["g4"] = {"good": good, "pairs": len(pairs), "models": {m: mid[m] for m in names}}
    # explicit verdicts of the registered predictions (review r31 S-4)
    g1 = sum(1 for c in sets["TEST-P"] if c["y"] > 0 and c["dec"]["check"] == "allow")
    cm = out["tables"]["COMMON (G2')"]
    g2 = (
        cm["beff"]["n"] > 0
        and cm["beff"]["bal_acc"] >= cm["check"]["bal_acc"] - 0.05
        and cm["beff"]["misclear_collapse"] == 0
    )
    g3_ok = bool(g3) and all(
        c["dec"]["bias"] == "allow"
        and c["dec"]["check"] != "allow"
        and c["dec"]["beff"] == "refuse"
        for c in g3
    )
    rat = {
        c["model"]: c["dec"]["ratio"] for c in cells if c["model"] in names and c["cfg"] == "q4_0"
    }
    g4_ok = bool(pairs) and good >= 0.8 * len(pairs) and all(v == "fallback" for v in rat.values())
    verdict = {
        "G1'": f"TEST-P damaged cells allowed by the check: {g1} -> {'PASS' if g1 == 0 else 'FAIL'}",
        "G2'": (
            f"common n {cm['beff']['n']}, beff BA {cm['beff']['bal_acc']:.3f} vs check "
            f"{cm['check']['bal_acc']:.3f}, beff collapse misclears {cm['beff']['misclear_collapse']} -> "
            f"{'PASS' if g2 else 'FAIL'}"
        ),
        "G3'": f"{len(g3)} no-bias collapse cells -> {'PASS' if g3_ok else 'FAIL'}",
        "G4'": (
            f"{good}/{len(pairs)} pairs (need >= {math.ceil(0.8 * len(pairs))}), ratio bands {rat} -> "
            f"{'PASS' if g4_ok else 'FAIL'}"
        ),
    }
    common = sets["COMMON (G2')"]
    for nm, t in sens_b.items():
        for c in common:
            c["dec"]["beff_s"] = "allow" if c["feat"]["b_meas"] >= t else "refuse"
        m = metrics(common, "beff_s")
        print(
            f"G2' sensitivity (descriptive): b* {nm} {t:.3f} -> COMMON beff BA {m['bal_acc']:.3f}, "
            f"collapse misclears {m['misclear_collapse']}"
        )
    for c in common:
        c["dec"].pop("beff_s", None)
    for k, v in verdict.items():
        print(f"VERDICT {k}: {v}")
    out["verdict"] = verdict
    out["train_family_table"] = ftab
    out["cells"] = [{k: v for k, v in c.items()} for c in cells]
    Path(a.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
