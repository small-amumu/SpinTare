# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""CLI.

  probe:  one 16-bit forward with SpinTare's kvprobe_unified.py (GPU or CPU), writes probe.json
    python -m kvsel probe --model DIR --order hf|gguf --out probe.json [--device cuda] [...]
  select: probe.json (+ optional config.json) and the deployment -> JSON configuration
    python -m kvsel select probe.json [--config config.json] --engine llama.cpp[:spintare]
        [--engine vllm[:spintare]] --ctx 32768 [--batch 1] [--gpu-kv-gib 8] [--no-rotation]
        [--model-type T] [--rope-type R|none]
Without --config the selector does not know NoPE layers or the RoPE scaling type: add-back kernels
then need --model-type / --rope-type.
Engine suffixes: ":spintare" (add-back + gate kernel), ":postmean" (static mean patch).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import evidence as ev
from .metrics import from_probe
from .select import Deployment, Engine, select


def _engine(spec: str, rotation: bool) -> Engine:
    name, *opts = spec.split(":")
    if name not in ("llama.cpp", "vllm"):
        raise SystemExit(f"unknown engine {name}")
    bad = [o for o in opts if o not in ("spintare", "postmean")]
    if bad:
        raise SystemExit(f"unknown engine option(s) {bad}")
    return Engine(name=name, rotation=rotation, spintare_kernel="spintare" in opts,  # type: ignore
                  postmean_kernel="postmean" in opts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kvsel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe", help="run SpinTare's kvprobe_unified.py (one 16-bit forward)")
    p.add_argument("--model", required=True)
    p.add_argument("--order", required=True, choices=["hf", "gguf"])
    p.add_argument("--out", required=True)
    p.add_argument("rest", nargs=argparse.REMAINDER, help="passed to kvprobe_unified.py")
    s = sub.add_parser("select")
    s.add_argument("probe", type=Path)
    s.add_argument("--config", type=Path, default=None,
                   help="HF config.json (NoPE layers, model_type, RoPE scaling)")
    s.add_argument("--model-type", default=None, help="HF model_type, if no config.json")
    s.add_argument("--rope-type", default=None,
                   help="HF rope_type ('none' = no scaling), if no config.json")
    s.add_argument("--engine", action="append", required=True)
    s.add_argument("--ctx", type=int, required=True)
    s.add_argument("--batch", type=int, default=1)
    s.add_argument("--gpu-kv-gib", type=float, default=float("inf"))
    s.add_argument("--no-rotation", action="store_true", help="llama.cpp K rotation disabled")
    a = ap.parse_args(argv)
    if a.cmd == "probe":
        cmd = [sys.executable, str(ev.KVPROBE_PY), "--model", a.model, "--order", a.order,
               "--out", a.out, *a.rest]
        return subprocess.call(cmd)
    try:
        rt = None if a.rope_type in (None, "none") else a.rope_type
        m = from_probe(a.probe, a.config, model_type=a.model_type, rope_type=rt,
                       rope_known=a.rope_type is not None)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"kvsel: {exc}") from exc
    dep = Deployment(engines=[_engine(e, not a.no_rotation) for e in a.engine], ctx_len=a.ctx,
                     batch=a.batch, gpu_kv_gib=a.gpu_kv_gib)
    print(json.dumps(select(m, dep).to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
