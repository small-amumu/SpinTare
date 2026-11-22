"""Configuration check for the vLLM INT4 per-token-head cache: b_top2 and its decision (paper sec.
4, eq. top2 / gate).

b_top2 = -log4(mean of the two largest 4^-b_l over layers), b_l = the probe's b_meas of the layer
for the format; the format is flagged (refused as is, so centring is selected) iff b_top2 <= tau_b =
2.0280476994784027 (frozen). Validation: the llama.cpp formats of the same pass (hadamard / identity
x q4_0 / q4_1) are compared with the stored probe.json of the model (paper probe, same text / seq /
skip); the pass is valid if every |difference| <= 0.01. Prints one JSON line per model and exits 1
if any validation fails. usage: python -I bt2_score.py <new probe.json> <stored probe.json> <model
tag>
"""

from __future__ import annotations

import json
import math
import sys

TAU_B = 2.0280476994784027
REF_KEYS = ("hadamard/q4_0", "identity/q4_0", "hadamard/q4_1", "identity/q4_1")


def b_top2(p: dict, key: str) -> float:
    n = p["layers_config"]
    b = [p["per_layer"][str(i)][key]["b_meas"] for i in range(n)]
    if not all(math.isfinite(x) for x in b):
        raise ValueError(f"non-finite b_meas in {key}")
    e = sorted((4.0 ** (-x) for x in b), reverse=True)
    return -math.log(sum(e[:2]) / 2, 4)


def main() -> int:
    new, old = (json.load(open(f, encoding="utf-8")) for f in sys.argv[1:3])
    tag = sys.argv[3]
    if new["layers_captured"] != new["layers_config"]:
        print(json.dumps({"model": tag, "error": "not every layer captured"}))
        return 1
    same = {k: new[k] == old[k] for k in ("seq", "skip", "order", "bos")}
    ref = {k: [round(b_top2(new, k), 4), round(b_top2(old, k), 4)] for k in REF_KEYS}
    ok = all(same.values()) and all(abs(a - b) <= 0.01 for a, b in ref.values())
    bt = b_top2(new, "identity/int4pth")
    n = new["layers_config"]
    worst = sorted(range(n), key=lambda i: new["per_layer"][str(i)]["identity/int4pth"]["b_meas"])[
        :2
    ]
    print(
        json.dumps(
            {
                "model": tag,
                "format": "int4_per_token_head (vLLM RHT, one 128-wide block)",
                "b_top2": round(bt, 4),
                "tau_b": TAU_B,
                "margin": round(bt - TAU_B, 4),
                "decision": "centre (format flagged: b_top2 <= tau_b)"
                if bt <= TAU_B
                else "fall back to the format as is (not flagged: b_top2 > tau_b)",
                "worst_layers": worst,
                "validation_ok": ok,
                "same_setup": same,
                "llamacpp_b_top2_new_vs_stored": ref,
            }
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
