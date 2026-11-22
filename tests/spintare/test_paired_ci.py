# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""paired-ci on synthetic LongBench result files."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

TASKS = ["2wikimqa", "hotpotqa", "musique", "passage_retrieval_en"]
ROOT = Path(__file__).resolve().parents[2]  # run from the repository root, installed or not


def write(d: Path, tag: str, scores: list[float], loop: bool = False, shift: int = 0) -> None:
    for t in TASKS:
        recs = [{"id": f"q{i + shift}", "score": s, "loop": loop} for i, s in enumerate(scores)]
        (d / f"{t}_lc_{tag}.json").write_text(json.dumps({"records": recs}), encoding="utf-8")


def run(d: Path, quant: str, b: str = "2000") -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "spintare", "paired-ci", str(d), "f16", quant, "--B", b]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


@pytest.fixture()
def results(tmp_path: Path) -> Path:
    base = [(i % 10) / 10 for i in range(40)]
    write(tmp_path, "f16", base)
    write(tmp_path, "same", base)
    write(tmp_path, "bad", [s * 0.3 for s in base], loop=True)
    write(tmp_path, "shift", base, shift=1)
    write(tmp_path, "nan", [float("nan")] + base[1:])
    return tmp_path


def test_verdicts(results: Path) -> None:
    same = json.loads(run(results, "same").stdout)
    assert same["verdict"] == "lossless" and same["ci95"] == [0.0, 0.0]
    bad = json.loads(run(results, "bad").stdout)
    assert bad["verdict"] == "collapse" and bad["loop_rate"] == 1.0


def test_refusals(results: Path) -> None:
    assert "cover different questions" in run(results, "shift").stderr
    assert "fractions in [0, 1]" in run(results, "nan").stderr
    assert "--B must be >= 1000" in run(results, "same", b="10").stderr
    assert "not found" in run(results, "missing").stderr


def test_unequal_task_sizes(tmp_path: Path) -> None:
    for tag in ("f16", "q"):
        for k, t in enumerate(TASKS):
            recs = [{"id": f"q{i}", "score": 0.5, "loop": False} for i in range(10 + 5 * k)]
            (tmp_path / f"{t}_lc_{tag}.json").write_text(json.dumps({"records": recs}))
    out = json.loads(run(tmp_path, "q").stdout)
    assert out["n_per_task"] == [10, 15, 20, 25] and out["verdict"] == "lossless"
