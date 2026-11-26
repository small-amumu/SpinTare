# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Synthetic end-to-end test of vpd_analyze.py: a passing tree, then three broken copies.

usage: python test_vpd_analyze.py   (exit 0 = every case gives the expected exit code)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ("Qwen2.5-1.5B-Instruct", "Qwen2.5-7B-Instruct", "Meta-Llama-3.1-8B-Instruct")
PROF = ("int4", "fp8", "int4_ab_v0t", "int4_ab_v1f", "fp8_ab_v0t", "fp8_ab_v1f")


def write(p: Path, obj: object) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def build(root: Path) -> None:
    for m in MODELS:
        stock = 7.0 if m.startswith("Qwen") else 0.0172
        arms = {
            "int4_stock": stock,
            "int4_59674": 0.014 if m.startswith("Qwen") else stock,
            "int4_ab_v1f": 0.010,
            "fp8_stock": 1.6,
            "fp8_ab_v1f": 0.0017,
        }
        for a, d in arms.items():
            write(root / "kl" / m / f"{a}.json", {"D": d, "addback_impls": 1 if "_ab_" in a else 0})
    for i, a in enumerate(PROF):
        ab = 1 if "_ab_" in a else 0
        write(
            root / f"prof_{a}.json",
            {"all_full_length": True, "addback_impls": ab, "num_gpu_blocks": 1000},
        )
        ev = [
            {
                "ph": "X",
                "cat": "kernel",
                "name": "unified_attention_kernel",
                "ts": 0,
                "dur": 1000.0 * (i + 1),
            },
            {"ph": "X", "cat": "kernel", "name": "cutlass_gemm", "ts": 5000, "dur": 500.0},
        ]
        write(root / "traces" / f"prof_{a}" / "t.pt.trace.json", {"traceEvents": ev})
        for r in (1, 2, 3):
            write(
                root / f"tp_{a}_r{r}.json",
                {
                    "all_full_length": True,
                    "addback_impls": ab,
                    "out_tok_s": 3000.0 - 100 * i + r,
                    "num_gpu_blocks": 1000,
                },
            )
    write(root / "tp_int4_end.json", {"all_full_length": True, "out_tok_s": 3001.0})


def run(root: Path) -> int:
    return subprocess.run(
        [sys.executable, str(HERE / "vpd_analyze.py"), str(root)], capture_output=True, text=True
    ).returncode


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    ok = True
    good = tmp / "good"
    build(good)
    rc = run(good)
    print("good tree rc", rc)
    ok &= rc == 0
    cases = {
        "llama PR != stock": ("kl/Meta-Llama-3.1-8B-Instruct/int4_59674.json", {"D": 0.0173}),
        "qwen PR == stock": ("kl/Qwen2.5-7B-Instruct/int4_59674.json", {"D": 7.0}),
        "NaN D": (
            "kl/Qwen2.5-1.5B-Instruct/fp8_ab_v1f.json",
            {"D": float("nan"), "addback_impls": 1},
        ),
        "empty trace": ("traces/prof_int4_ab_v1f/t.pt.trace.json", {"traceEvents": []}),
        "NaN tok/s": (
            "tp_int4_r1.json",
            {"all_full_length": True, "addback_impls": 0, "out_tok_s": float("nan")},
        ),
        "add-back not built": (
            "tp_fp8_ab_v0t_r3.json",
            {"all_full_length": True, "addback_impls": 0, "out_tok_s": 2500.0},
        ),
        "two traces": ("traces/prof_fp8/u.pt.trace.json", {"traceEvents": []}),
        "list json": ("prof_fp8.json", [1, 2]),
    }
    for name, (rel, obj) in cases.items():
        d = tmp / name.replace(" ", "_").replace("!", "n").replace("=", "e").replace("/", "_")
        shutil.copytree(good, d)
        write(d / rel, obj)
        rc = run(d)
        print(f"{name}: rc {rc}")
        ok &= rc == 2
    d = tmp / "missing_tp"
    shutil.copytree(good, d)
    (d / "tp_fp8_ab_v1f_r2.json").unlink()
    rc = run(d)
    print(f"missing throughput run: rc {rc}")
    ok &= rc == 2
    print("ALL OK" if ok else "TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
