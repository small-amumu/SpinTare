# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""Detection baselines as decision rules on the features of `spintare probe`, `spintare offset` and
an engine D.

Each rule returns "allow", "refuse" (do not deploy the format), "fallback" or "alarm" for one
(model, format) cell:
  bias      vLLM PR #63081: refuse when the PR's key-bias statistic exceeds 100 (computed by
            `spintare probe`).
  ratio     offset ratio of the 32-element blocks (r_blk32_max of `spintare offset`): allow r <= 3,
            alarm r > 15, fallback in between.
  mlx       KL band of mlx-quant-fidelity on the engine D: allow D <= 0.01, else refuse.
  cos       SageAttention-style per-layer check: allow when the minimum attention-output cosine
            >= 0.998.
  akl / kmse / beff
            simulated attention KL, key reconstruction error and measured effective bits:
            thresholds are fitted on labelled (model, format) cells with `fit_threshold` (maximum
            balanced accuracy) and then applied unchanged to new models; the labelled cells are
            not part of this repository.
  check     the SpinTare check on the engine D: allow D < 0.1016; otherwise allow when the repaired D'
            < 0.1016, alarm when D' > 0.3577, fallback otherwise (without D', fallback).
The origin of every constant is listed in TOOLS.md (Thresholds).
Usage: python -m spintare rules --probe PROBE.json --fmt q4_0 [--basis hadamard]
       [--offset OFFSET.json] [--D 0.05] [--D-repaired 0.02] [--thresholds TH.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spintare.check.check import TAU_ALARM, TAU_ALLOW  # one definition of the check thresholds


def fit_threshold(points: list[tuple[float, bool]], higher_is_safe: bool) -> float | None:
    """points: (feature, damaged). Threshold with the best balanced accuracy; ties go to the
    strictest one."""
    if not points or all(d for _, d in points) or not any(d for _, d in points):
        return None
    best = None
    n_dam = sum(1 for _, d in points if d)
    n_ok = len(points) - n_dam
    for t, _ in sorted(points):
        allow = [(v >= t) if higher_is_safe else (v <= t) for v, _ in points]
        sens = sum(1 for a, (_, d) in zip(allow, points) if d and not a) / n_dam
        spec = sum(1 for a, (_, d) in zip(allow, points) if not d and a) / n_ok
        ba = (sens + spec) / 2
        tie = t if higher_is_safe else -t
        if best is None or ba > best[0] + 1e-12 or (abs(ba - best[0]) <= 1e-12 and tie > best[1]):
            best = (ba, tie, t)
    return best[2]


def check_rule(D: float | None, Dr: float | None) -> str | None:
    if D is None:
        return None
    if D < TAU_ALLOW or (Dr is not None and Dr < TAU_ALLOW):
        return "allow"
    return "alarm" if Dr is not None and Dr > TAU_ALARM else "fallback"


def decide(
    feat: dict, D: float | None, Dr: float | None, r: float | None, th: dict
) -> dict[str, str | None]:
    d: dict[str, str | None] = {}
    d["check"] = check_rule(D, Dr)
    d["mlx"] = None if D is None else ("allow" if D <= 0.01 else "refuse")
    d["ratio"] = None if r is None else ("allow" if r <= 3 else ("alarm" if r > 15 else "fallback"))
    d["bias"] = "refuse" if feat["bias_stat"] > 100 else "allow"
    d["cos"] = "allow" if feat["cos_min"] >= 0.998 else "refuse"
    d["beff"] = (
        None if th.get("beff") is None else ("allow" if feat["b_meas"] >= th["beff"] else "refuse")
    )
    d["akl"] = (
        None if th.get("akl") is None else ("allow" if feat["attn_kl"] <= th["akl"] else "refuse")
    )
    d["kmse"] = (
        None if th.get("kmse") is None else ("allow" if feat["key_mse"] <= th["kmse"] else "refuse")
    )
    return d


def features(probe: dict, basis: str, fmt: str) -> dict:
    s, k = probe["summary"], f"{basis}/{fmt}"
    return {
        "b_meas": s[f"{k}/b_meas_mean"],
        "attn_kl": s[f"{k}/attn_kl_mean"],
        "key_mse": s[f"{k}/key_rel_mse_mean"],
        "cos_min": s[f"{k}/out_cos_min"],
        "bias_stat": probe["weights"]["bias_stat"],
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="spintare rules")
    ap.add_argument("--probe", required=True, type=Path)
    ap.add_argument("--fmt", default="q4_0", choices=("q4_0", "q4_1", "q8_0"))
    ap.add_argument("--basis", default="hadamard", choices=("hadamard", "identity"))
    ap.add_argument(
        "--offset", type=Path, default=None, help="`spintare offset` JSON (for the ratio rule)"
    )
    ap.add_argument("--D", type=float, default=None, help="engine D of this cell (for check / mlx)")
    ap.add_argument(
        "--D-repaired", type=float, default=None, help="engine D with postmean (for check)"
    )
    ap.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help="JSON with fitted beff / akl / kmse thresholds",
    )
    a = ap.parse_args(argv)
    probe = json.loads(a.probe.read_text(encoding="utf-8"))
    r = json.loads(a.offset.read_text(encoding="utf-8"))["r_blk32_max"] if a.offset else None
    th = json.loads(a.thresholds.read_text(encoding="utf-8")) if a.thresholds else {}
    print(
        json.dumps(
            {
                "model": probe["model"],
                "fmt": a.fmt,
                "basis": a.basis,
                "decisions": decide(features(probe, a.basis, a.fmt), a.D, a.D_repaired, r, th),
            }
        )
    )


if __name__ == "__main__":
    main()
