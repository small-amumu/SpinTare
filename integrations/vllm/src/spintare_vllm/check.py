# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""SpinTare pre-deployment check for vLLM's low-bit KV caches.

For each candidate KV format (smallest first) the check measures D, the mean output KL against the
16-bit cache (spintare_vllm.kld: 4 chunks of 2048 tokens, second half of each chunk scored).
D < TAU_ALLOW allows the format; otherwise the postmean repair (SPINTARE_KVPM=<mean file>) is
tried under the same threshold; a repaired D' > TAU_ALARM raises an alarm. If nothing passes,
the 16-bit cache.

Difference from the llama.cpp check (recorded in the output): vLLM quantises K and V with one
dtype, so D here includes the V error, whereas the llama.cpp thresholds were fitted with a q8_0 V
cache and q4_0 / q4_1 K. The thresholds are reused unchanged and are not refitted for vLLM formats.
Each pass is a separate process (one LLM per process).
Usage: python -m spintare_vllm check --model DIR --text FILE (--m FILE | --no-postmean)
       [--formats int4_per_token_head,fp8_per_token_head] [--workdir DIR] [--util 0.85] [--graph]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

TAU_ALLOW = 0.1016  # allow threshold on D (fitted on llama.cpp cells; see the module docstring)
TAU_ALARM = 0.3577  # alarm threshold on the repaired D'
BITS = {"int4_per_token_head": 4.25, "int8_per_token_head": 8.25, "fp8_per_token_head": 8.25}
THRESHOLD_NOTE = (
    "thresholds fitted on llama.cpp (K q4_0 / q4_1, V q8_0); "
    "vLLM D includes the V error; not refitted"
)


def run_pass(
    a: argparse.Namespace, workdir: Path, kv: str, ref: bool, m: str | None
) -> dict[str, object]:
    tag = f"{kv}{'_pm' if m else ''}{'_ref' if ref else ''}"
    out = workdir / f"{tag}.json"
    cmd = (
        [
            sys.executable,
            "-m",
            "spintare_vllm.kld",
            "--model",
            a.model,
            "--kv",
            kv,
            "--refdir",
            str(workdir / "ref"),
            "--out",
            str(out),
            "--text",
            a.text,
            "--util",
            str(a.util),
        ]
        + (["--ref"] if ref else [])
        + (["--graph"] if a.graph else [])
    )
    env = {k: v for k, v in os.environ.items() if k != "SPINTARE_KVPM"}
    if m:
        env["SPINTARE_KVPM"] = m
    with (workdir / f"{tag}.log").open("w", encoding="utf-8") as log:
        rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        sys.exit(f"KL pass {tag} failed (rc {rc}); log {workdir / (tag + '.log')}")
    return json.loads(out.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="spintare_vllm check")
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--m", help="postmean file (KVPM); without it postmean is not tried")
    ap.add_argument("--no-postmean", action="store_true")
    ap.add_argument("--formats", default="int4_per_token_head,fp8_per_token_head")
    ap.add_argument("--workdir", help="keep logs and per-pass JSON here (default: temporary)")
    ap.add_argument("--util", type=float, default=0.85)
    ap.add_argument("--graph", action="store_true")
    a = ap.parse_args(argv)
    if not a.m and not a.no_postmean:
        sys.exit("pass --m <mean file> (or --no-postmean to skip the repair step)")
    if a.m and not Path(a.m).exists():
        sys.exit(f"mean file not found: {a.m}")
    formats = a.formats.split(",")
    if any(f not in BITS for f in formats):
        sys.exit(f"--formats must be a subset of {sorted(BITS)}")
    formats.sort(key=BITS.get)

    trace: list[dict[str, object]] = []
    alarms: list[str] = []

    def report(decision: str, postmean: bool) -> None:
        print(
            json.dumps(
                {
                    "decision": decision,
                    "postmean": postmean,
                    "alarms": alarms,
                    "thresholds": {"allow": TAU_ALLOW, "alarm": TAU_ALARM, "note": THRESHOLD_NOTE},
                    "trace": trace,
                }
            )
        )

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(a.workdir or tmp)
        workdir.mkdir(parents=True, exist_ok=True)
        run_pass(a, workdir, "auto", ref=True, m=None)
        for f in formats:
            r = run_pass(a, workdir, f, ref=False, m=None)
            trace.append({"format": f, "D": r["D"], "top1_same": r["top1_same"]})
            if float(r["D"]) < TAU_ALLOW:  # type: ignore[arg-type]
                report(f, False)
                return
            if a.m:
                rp = run_pass(a, workdir, f, ref=False, m=a.m)
                t = {"format": f, "postmean": True, "D": rp["D"], "top1_same": rp["top1_same"]}
                t["postmean_md5"] = rp["postmean_md5"]
                trace.append(t)
                if rp["D"] == r["D"]:
                    t["refused"] = "repair did not run (D unchanged)"
                    continue
                if float(rp["D"]) < TAU_ALLOW:  # type: ignore[arg-type]
                    report(f, True)
                    return
                if float(rp["D"]) > TAU_ALARM:  # type: ignore[arg-type]
                    t["alarm"] = True
                    alarms.append(f)
                    print(f"ALARM: {f} with postmean has D' = {rp['D']:.4f}", file=sys.stderr)
        report("auto", False)


if __name__ == "__main__":
    main()
