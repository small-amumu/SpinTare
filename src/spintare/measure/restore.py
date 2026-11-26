# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Layer-wise restoration of a quantised K cache (Hugging Face simulation).

All layers' K are quantised; then layers are put back to 16-bit to see where the output damage comes
from:
  - one at a time: D_without_l = output KL with every layer quantised except l; recovery_l = 1 -
    D_without_l / D_all;
  - cumulatively, in the order of the largest single-layer recovery: D after restoring the top 1..K
    layers.
Arms: "none" (plain format) and "postmean" (static post-RoPE mean removed before the write, its
score term q . m added back, which equals the engine's softmax). A damage-attribution diagnostic,
not a cache mode (K quantised, V exact; see spintare/sim.py). The engine-side counterpart is
`spintare restore-engine` (llama.cpp with LLAMA_KV_F16_LAYERS).
Usage: python -m spintare restore --model DIR --text EVAL.txt --calib CALIB.txt --out OUT.json
       [--fmt q4_0] [--T 2048] [--restore-top 4] [--arms none,postmean] [--basis hadamard|identity]
       [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from spintare.sim import Plan, Sim, kl_rows

ARMS = ("none", "postmean")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="spintare restore")
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True, type=Path)
    ap.add_argument("--calib", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fmt", default="q4_0", choices=("q4_0", "q4_1", "q8_0"))
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument(
        "--restore-top", type=int, default=4, help="number of layers restored cumulatively"
    )
    ap.add_argument(
        "--arms", default="none,postmean", help="comma-separated subset of none,postmean"
    )
    ap.add_argument("--basis", default="hadamard", choices=("hadamard", "identity"))
    ap.add_argument("--order", default=None, choices=("hf", "gguf"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    a = ap.parse_args(argv)
    arms = a.arms.split(",")
    if any(x not in ARMS for x in arms):
        ap.error(f"--arms must be a subset of {ARMS}")
    if a.restore_top < 0:
        ap.error("--restore-top must be >= 0")
    if a.text.resolve() == a.calib.resolve():
        ap.error("calibration and evaluation texts must differ")
    t0 = time.time()
    sim = Sim(a.model, a.device, a.dtype, a.order, a.basis)
    ids = sim.encode(a.text, a.T)
    start = a.T // 2
    means = sim.means(sim.encode(a.calib, a.T))
    ref = sim.logprobs(ids, Plan())
    every = set(range(sim.n_layer))

    def d_of(quantised: set[int], m: dict[int, torch.Tensor] | None) -> float:
        return float(kl_rows(ref, sim.logprobs(ids, Plan(a.fmt, quantised, m)), start).mean())

    res: dict = {
        "model": a.model,
        "fmt": a.fmt,
        "T": a.T,
        "basis": a.basis,
        "order": sim.engine.order,
        "arms": {},
    }
    for arm in arms:
        m = means if arm == "postmean" else None
        d_all = d_of(every, m)
        single = []
        for li in range(sim.n_layer):
            d = d_of(every - {li}, m)
            single.append(
                {
                    "layer": li,
                    "D_without": round(d, 6),
                    "recovery": round(1 - d / d_all, 4) if d_all > 0 else None,
                }
            )
        order = [x["layer"] for x in sorted(single, key=lambda x: x["D_without"])][: a.restore_top]
        cum = [
            {"restored": order[: i + 1], "D": round(d_of(every - set(order[: i + 1]), m), 6)}
            for i in range(len(order))
        ]
        res["arms"][arm] = {
            "D_all": round(d_all, 6),
            "order": order,
            "cumulative": cum,
            "single": single,
        }
        print(
            json.dumps(
                {
                    "arm": arm,
                    "D_all": round(d_all, 6),
                    "order": order,
                    "D_after_top": cum[-1]["D"] if cum else None,
                }
            ),
            flush=True,
        )
    res["sec"] = round(time.time() - t0, 1)
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
