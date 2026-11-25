# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""Analyzer of the vLLM-figure run (vpd_chain.sh): controls and panel values on one RTX PRO 5000.

Controls (any failure -> exit 2, nothing from the round is used):
  C1 every KL arm present with finite, positive D; add-back arms built (addback_impls > 0)
  C2 identity: on Llama-3.1-8B (no key bias) the PR #59674 arm equals stock INT4 exactly
  C3 PR applied: on both Qwen2.5 models the PR arm differs from stock INT4
  C4 every profiled run present with a trace, full-length outputs, add-back impls built
  C5 every throughput run present (3 repeats per arm + the stock re-time), full length, add-back
     built, out tok/s finite and positive, equal KV blocks across the repeats of an arm
Prints the panel values: (a) KL per model and arm, (b) attention-kernel ms per profiled arm and its
ratio to stock, (c) median out tok/s per arm, spread, and ratio to the matching stock arm.
Writes vpd_result.json next to the runs. Exit 0 = controls pass.
usage: python vpd_analyze.py <results/vpd>
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf1_trace as T  # noqa: E402

MODELS = ("Qwen2.5-1.5B-Instruct", "Qwen2.5-7B-Instruct", "Meta-Llama-3.1-8B-Instruct")
KL_ARMS = ("int4_stock", "int4_59674", "int4_ab_v1f", "fp8_stock", "fp8_ab_v1f")
PROF = ("int4", "fp8", "int4_ab_v0t", "int4_ab_v1f", "fp8_ab_v0t", "fp8_ab_v1f")


def load(p: Path) -> dict | None:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def pos(x: object) -> bool:
    """True for a finite, positive number (bool excluded)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def main() -> int:
    out = Path(sys.argv[1])
    (out / "vpd_result.json").unlink(missing_ok=True)
    try:
        return analyze(out)
    except Exception as e:  # any crash is a failed round, never a stale PASS
        crash = {"controls": "FAIL", "fails": [f"analyzer crashed: {e!r}"]}
        (out / "vpd_result.json").write_text(json.dumps(crash), encoding="utf-8")
        print(f"VPD CONTROLS FAIL (analyzer crashed: {e!r})")
        return 2


def analyze(out: Path) -> int:
    fails: list[str] = []
    res: dict = {
        "meta": {"script_md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest()},
        "kl": {},
        "kernel_ms": {},
        "throughput": {},
    }
    # (a) output KL
    for m in MODELS:
        row = {}
        for a in KL_ARMS:
            j = load(out / "kl" / m / f"{a}.json")
            d = None if j is None else j.get("D")
            if not pos(d):
                fails.append(f"C1 {m} {a}: D missing or unusable ({d})")
                continue
            if "_ab_" in a and not j.get("addback_impls"):
                fails.append(f"C1 {m} {a}: add-back not built")
            row[a] = d
        res["kl"][m] = row
        s, p = row.get("int4_stock"), row.get("int4_59674")
        if s is not None and p is not None:
            if m.startswith("Meta-Llama") and not (p == s):
                fails.append(f"C2 {m}: PR {p} != stock {s} (PR must be a no-op without key bias)")
            if m.startswith("Qwen2.5") and not (p != s):
                fails.append(f"C3 {m}: PR equals stock ({p}): PR path not applied")
        print(f"(a) {m:28s} " + " ".join(f"{a}={row.get(a, float('nan')):.5g}" for a in KL_ARMS))
    # (b) attention kernel time, eager profile
    for a in PROF:
        j = load(out / f"prof_{a}.json")
        tr = sorted((out / "traces" / f"prof_{a}").rglob("*.json*"))
        if j is None or not tr:
            fails.append(f"C4 prof_{a}: missing run or trace")
            continue
        if not j.get("all_full_length") or ("_ab_" in a and not j.get("addback_impls")):
            fails.append(f"C4 prof_{a}: short outputs or add-back not built")
        if len(tr) != 1:
            fails.append(f"C4 prof_{a}: {len(tr)} trace files, expected 1")
            continue
        r = T.analyze(tr[0])
        att = r["by_cat_ms"].get("attention", 0.0)
        if not (r["kernels"] > 0 and att > 0 and math.isfinite(att)):
            fails.append(f"C4 prof_{a}: trace has no attention kernels")
            continue
        res["kernel_ms"][a] = {
            "attention": att,
            "kernel_total": r["kernel_ms"],
            "other": r["by_cat_ms"].get("other", 0.0),
            "kv_write": r["by_cat_ms"].get("kv_write", 0.0),
            "kv_blocks": j.get("num_gpu_blocks"),
        }
    for a, v in res["kernel_ms"].items():
        base = res["kernel_ms"].get(a.split("_")[0], {}).get("attention")
        ratio = v["attention"] / base if base else float("nan")
        print(
            f"(b) {a:12s} attention {v['attention']:9.1f} ms x{ratio:.2f} of stock | "
            f"kv_write {v['kv_write']:.1f} other {v['other']:.1f} total {v['kernel_total']:.1f} ms"
        )
    # (c) throughput with CUDA graphs
    for a in PROF:
        rs = [load(p) for p in sorted(out.glob(f"tp_{a}_r[123].json"))]
        rs = [r for r in rs if isinstance(r, dict)]
        good = [r for r in rs if pos(r.get("out_tok_s"))]
        if (
            len(good) != 3
            or not all(r.get("all_full_length") for r in good)
            or ("_ab_" in a and not all(r.get("addback_impls") for r in good))
        ):
            fails.append(f"C5 tp_{a}: {len(good)} of 3 usable runs or controls fail")
            if not good:
                continue
        v = [r["out_tok_s"] for r in good]
        med = st.median(v)
        blocks = sorted({r.get("num_gpu_blocks") for r in good}, key=str)
        if len(blocks) != 1:  # same arm, same card: the KV regime must not change between repeats
            fails.append(f"C5 tp_{a}: KV blocks differ between repeats {blocks}")
        res["throughput"][a] = {
            "median": med,
            "runs": v,
            "spread": (max(v) - min(v)) / med,
            "kv_blocks": blocks,
        }
    for a, v in res["throughput"].items():
        base = res["throughput"].get(a.split("_")[0], {}).get("median")
        v["vs_stock"] = v["median"] / base if base else None
        print(
            f"(c) {a:12s} {v['median']:8.1f} out tok/s spread {v['spread']:.1%} "
            f"vs stock {v['vs_stock']}"
        )
    e = load(out / "tp_int4_end.json")
    if not isinstance(e, dict) or not e.get("all_full_length") or not pos(e.get("out_tok_s")):
        fails.append("C5 tp_int4_end missing or unusable")
    else:  # informational: drift of the stock arm over the round
        res["throughput"]["int4_end"] = e["out_tok_s"]
        print(f"(c) stock int4 re-time {e['out_tok_s']:.1f} (informational)")
    res["controls"] = "PASS" if not fails else "FAIL"
    res["fails"] = fails
    (out / "vpd_result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for f in fails:
        print("FAIL", f)
    print(f"VPD CONTROLS {res['controls']}")
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())
