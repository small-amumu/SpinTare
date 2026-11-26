# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Layer-wise restoration in llama.cpp: D with the K of selected layers kept in F16
(LLAMA_KV_F16_LAYERS). A damage-attribution diagnostic (which layers the damage comes from), not a
cache mode.

Uses the same measurement as `spintare check` (llama-perplexity --kl-divergence against the 16-bit K
cache, -c 2048, 4 chunks, V q8_0). Requires a llama.cpp build with the SpinTare patch, which reads
LLAMA_KV_F16_LAYERS="0,3" and keeps K of the listed layers in F16. Runs: all layers quantised, then
each layer in --layers restored alone, then the --layers set restored together. Optional --m applies
the postmean repair to the quantised layers.
Usage: python -m spintare restore-engine <llama.cpp bin dir> <model.gguf> <text> --ctk q4_0
       --layers 0,1,27 [--m m.kvpm] [--ngl 99]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from spintare.check.check import KLD

KEPT = re.compile(r"layer\s+(\d+): K kept in F16")


def run(
    a: argparse.Namespace, base: Path, ctk: str, score: bool, f16_layers: str | None
) -> float | None:
    cmd = [
        str(Path(a.bin) / "llama-perplexity"),
        "-m",
        a.gguf,
        "-f",
        a.text,
        "-c",
        "2048",
        "--chunks",
        "4",
        "-fa",
        "on",
        "-ctv",
        "q8_0",
        "-ngl",
        str(a.ngl),
        "-ctk",
        ctk,
        "--kl-divergence-base",
        str(base),
    ]
    cmd += ["--kl-divergence"] if score else []
    env = {
        k: v for k, v in os.environ.items() if k not in ("LLAMA_KV_POSTMEAN", "LLAMA_KV_F16_LAYERS")
    }
    if a.m and score:
        env["LLAMA_KV_POSTMEAN"] = a.m
    if f16_layers:
        env["LLAMA_KV_F16_LAYERS"] = f16_layers
        cmd += ["-lv", "4"]  # the per-layer confirmation is an info-level log line
    out = subprocess.run(
        cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    log = out.stdout + "\n" + out.stderr
    if out.returncode != 0:
        sys.exit(f"llama-perplexity failed ({ctk}, F16 layers {f16_layers}): {log[-500:]}")
    if f16_layers:
        kept = {int(x) for x in KEPT.findall(log)}
        missing = sorted({int(x) for x in f16_layers.split(",")} - kept)
        if not kept:
            sys.exit(
                "no listed layer was kept in F16: the llama.cpp build lacks the SpinTare patch, "
                "or every index is out of range for this model"
            )
        if missing:
            sys.exit(f"layers {missing} were not kept in F16 (index out of range for this model?)")
    if not score:
        return None
    hit = KLD.search(log)
    if hit is None:
        sys.exit(f"no 'Mean KLD' line ({ctk}, F16 layers {f16_layers}): {log[-500:]}")
    return float(hit.group(1))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="spintare restore-engine")
    ap.add_argument("bin")
    ap.add_argument("gguf")
    ap.add_argument("text")
    ap.add_argument("--ctk", default="q4_0")
    ap.add_argument("--layers", required=True, help="comma-separated layer indices to restore")
    ap.add_argument(
        "--m", default=None, help="postmean file (KVPM) applied to the quantised layers"
    )
    ap.add_argument("--ngl", type=int, default=99)
    a = ap.parse_args(argv)
    if a.m and not Path(a.m).exists():
        sys.exit(f"postmean file not found: {a.m}")
    try:
        layers = list(dict.fromkeys(int(x) for x in a.layers.split(",")))  # unique, in order
    except ValueError:
        ap.error("--layers must be comma-separated integers")
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "base.kld"
        run(a, base, "f16", False, None)
        d_all = run(a, base, a.ctk, True, None)
        single = {li: run(a, base, a.ctk, True, str(li)) for li in layers}
        d_set = run(a, base, a.ctk, True, ",".join(map(str, layers)))
    print(
        json.dumps(
            {
                "ctk": a.ctk,
                "postmean": bool(a.m),
                "D_all": d_all,
                "single": [
                    {
                        "layer": li,
                        "D": d,
                        "recovery": None if not d_all else round(1 - d / d_all, 4),
                    }
                    for li, d in single.items()
                ],
                "restored_set": layers,
                "D_set": d_set,
            }
        )
    )


if __name__ == "__main__":
    main()
