# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Export per-layer effective bits (hadamard/q4_0, engine coordinates) of released GATE-PRED models
with their worst q4_0 label, from the frozen configuration-check results.
Usage: python export_layers.py > ../../../results/spintare/figure_inputs/layers.csv"""

from __future__ import annotations

import json
from pathlib import Path

DG = Path(__file__).resolve().parents[3] / "results" / "spintare" / "detection_gate"
labels: dict[str, list[int]] = {}
for c in json.load(open(DG / "gp_score.json"))["cells"]:
    if (
        c.get("tag")
        and "_inj" not in str(c["tag"])
        and "inj" not in c["model"]
        and c.get("y") is not None
    ):
        labels.setdefault(c["tag"], []).append(c["y"])
print("tag,worst_label,layer,b")
for d in sorted((DG / "results").iterdir()):
    tag = d.name
    if "_inj" in tag or tag not in labels or not (d / "probe.json").exists():
        continue
    pj = json.load(open(d / "probe.json"))
    if pj["layers_captured"] != pj["layers_config"]:
        continue
    for i in range(pj["layers_config"]):
        b = pj["per_layer"][str(i)]["hadamard/q4_0"]["b_meas"]
        print(f"{tag},{max(labels[tag])},{i},{b:.4f}")
