# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Configuration check of one model from its probe output (b_top2 of the two worst layers).

b_l is the effective bits of layer l in one K format, b_meas of the probe (kvprobe_unified.py, or
kvprobe_pth.py for vLLM's int4_per_token_head). b_top2 = -log4(mean of the two largest 4^-b_l over the
layers): the energy mean of the two worst layers. A format with b_top2 <= tau_b is flagged: the model
needs centring (SpinTare) in that format; otherwise the plain 4-bit cache is kept. tau_b = 2.028 was
fitted on the 60 natural cells of the fitting set and frozen. For vLLM's 128-wide INT4 blocks the same
threshold is an extrapolation from llama.cpp's 32-wide blocks.

Usage: python config_check.py <probe.json> [--formats hadamard/q4_0,identity/q4_0,...] [--json]
Format keys are "<basis>/<format>": basis hadamard (llama.cpp's default K rotation) or identity
(no rotation), format q4_0, q4_1, q8_0 or int4pth.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

TAU_B = 2.0280476994784027  # frozen threshold (fitting set, 29 models / 60 cells)


def b_top2(bits: list[float]) -> float:
    """Energy mean of the two worst layers: -log4(mean of the two largest 4^-b)."""
    if not bits or not all(math.isfinite(b) for b in bits):
        raise ValueError("need finite per-layer bits")
    e = sorted((4.0 ** (-b) for b in bits), reverse=True)[:2]
    return -math.log(sum(e) / len(e), 4)


def layer_bits(probe: dict, key: str) -> list[float]:
    n = probe["layers_config"]
    if probe.get("layers_captured", n) != n:
        raise ValueError(f"probe captured {probe.get('layers_captured')} of {n} layers")
    return [probe["per_layer"][str(i)][key]["b_meas"] for i in range(n)]


def check(probe: dict, keys: list[str]) -> list[dict]:
    out = []
    for key in keys:
        bits = layer_bits(probe, key)
        b = b_top2(bits)
        worst = sorted(range(len(bits)), key=lambda i: bits[i])[:2]
        out.append(
            {
                "format": key,
                "b_top2": round(b, 4),
                "tau_b": TAU_B,
                "decision": "centre" if b <= TAU_B else "plain",
                "worst_layers": worst,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("probe", type=Path)
    ap.add_argument("--formats", default=None, help="comma-separated format keys (default: all)")
    ap.add_argument("--json", action="store_true", help="print JSON lines")
    a = ap.parse_args()
    probe = json.loads(a.probe.read_text(encoding="utf-8"))
    first = probe["per_layer"]["0"]
    keys = (
        a.formats.split(",")
        if a.formats
        else [k for k in first if isinstance(first[k], dict) and "b_meas" in first[k]]
    )
    for r in check(probe, keys):
        if a.json:
            print(json.dumps(r))
        else:
            print(
                f"{r['format']:<18} b_top2 {r['b_top2']:7.4f}  tau_b {TAU_B:.3f}"
                f"  -> {r['decision']}  (worst layers {r['worst_layers']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
