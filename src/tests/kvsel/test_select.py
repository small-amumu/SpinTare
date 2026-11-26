# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Unit tests of kvsel on synthetic metrics (no probe files needed except config_check.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kvsel.metrics import ModelMetrics  # noqa: E402
from kvsel.select import Deployment, Engine, select  # noqa: E402


def mk(b: float, d: int = 128, kvh: int = 8, heads: int = 32, rot: int | None = None,
       nope: list[int] | None = None, model_type: str | None = "qwen2",
       rope_type: str | None = None, known: bool = True) -> ModelMetrics:
    keys = ["hadamard/q4_0", "hadamard/q4_1", "hadamard/q8_0", "identity/q4_0", "identity/q4_1",
            "identity/q8_0"]
    bt2 = {k: (b if "q8" not in k else 6.0) for k in keys}
    return ModelMetrics("m", 32, d, heads, kvh, rot or d, False, False, 0.0, "hf", bt2,
                        {k: [0, 1] for k in keys}, nope_layers=nope or [], model_type=model_type,
                        rope_type=rope_type, config_read=known, rope_known=known)


def test_spintare_kernel_only_on_wired_graphs() -> None:
    dep = Deployment([Engine("llama.cpp", spintare_kernel=True)], 32768)
    for mt in ("qwen2", "qwen3", "gpt_neox"):
        assert select(mk(1.0, model_type=mt), dep).chosen.centring == "spintare"  # type: ignore
    for mt in ("llama", "glm", "smollm3", None):  # not wired, or unknown without a config
        c = select(mk(1.0, model_type=mt), dep).chosen
        assert c is not None and c.k_format == "q8_0", mt


def test_spintare_kernel_refuses_yarn_but_vllm_accepts_llama3() -> None:
    lc = Deployment([Engine("llama.cpp", spintare_kernel=True)], 32768)
    c = select(mk(1.0, model_type="qwen3", rope_type="yarn"), lc).chosen
    assert c is not None and c.k_format == "q8_0"
    vl = Deployment([Engine("vllm", spintare_kernel=True)], 32768)
    c = select(mk(1.0, model_type="llama", rope_type="llama3"), vl).chosen
    assert c is not None and c.centring == "spintare"
    c = select(mk(1.0, model_type="qwen2", rope_type="longrope"), vl).chosen
    assert c is not None and c.k_format == "bf16"
    c = select(mk(1.0, model_type=None, known=False), vl).chosen  # RoPE scaling unknown
    assert c is not None and c.k_format == "bf16"


def test_postmean_only_on_wired_graphs() -> None:
    e = Engine("llama.cpp", postmean_kernel=True)
    c = select(mk(1.0, model_type="llama"), Deployment([e], 2048)).chosen
    assert c is not None and c.centring == "postmean"
    c = select(mk(1.0, model_type="gemma2"), Deployment([e], 2048)).chosen
    assert c is not None and c.k_format == "q8_0"


def test_safe_model_runs_q4_0_plain() -> None:
    s = select(mk(3.0), Deployment([Engine("llama.cpp")], 32768))
    assert s.chosen is not None and s.chosen.k_format == "q4_0" and s.chosen.centring == "none"


def test_flagged_without_kernel_falls_back_to_q8() -> None:
    s = select(mk(1.0), Deployment([Engine("llama.cpp")], 32768))
    assert s.chosen is not None and s.chosen.k_format == "q8_0"
    assert {r["k_format"] for r in s.rejected} == {"q4_0", "q4_1"}


def test_flagged_with_kernel_uses_spintare_and_gate() -> None:
    s = select(mk(1.0), Deployment([Engine("llama.cpp", spintare_kernel=True)], 32768))
    c = s.chosen
    assert c is not None and c.k_format == "q4_0" and c.centring == "spintare"
    assert c.exemption_gate is True and c.checks_required


def test_postmean_only_inside_calibration_window() -> None:
    e = Engine("llama.cpp", postmean_kernel=True)
    assert select(mk(1.0), Deployment([e], 2048)).chosen.centring == "postmean"  # type: ignore
    assert select(mk(1.0), Deployment([e], 8192)).chosen.k_format == "q8_0"  # type: ignore


def test_vllm_flagged_without_kernel_falls_back_to_bf16() -> None:
    s = select(mk(1.0), Deployment([Engine("vllm")], 32768))
    assert s.chosen is not None and s.chosen.k_format == "bf16"
    assert any(r.rule == "C1" for r in s.chosen.reasons)


def test_unknown_config_is_not_covered() -> None:
    # no config: llama.cpp add-back needs the RoPE type
    lc = Deployment([Engine("llama.cpp", spintare_kernel=True)], 32768)
    c = select(mk(1.0, model_type="qwen2", known=False), lc).chosen
    assert c is not None and c.k_format == "q8_0"


def test_postmean_reasons_do_not_carry_the_addback_refusal() -> None:
    e = Engine("llama.cpp", spintare_kernel=True, postmean_kernel=True)
    c = select(mk(1.0, model_type="llama"), Deployment([e], 2048)).chosen
    assert c is not None and c.centring == "postmean"
    assert not any(r.rule == "C13" for r in c.reasons)


def test_window_longer_than_context_is_not_sliding() -> None:
    from kvsel.metrics import _arch_flags
    full = _arch_flags({"sliding_window": 262144, "max_position_embeddings": 131072}, 4)
    real = _arch_flags({"sliding_window": 4096, "max_position_embeddings": 8192}, 4)
    assert full.get("sliding_window_layers", 0) == 0 and real["sliding_window_layers"] == -1


def test_vllm_coverage_matches_calib_mu() -> None:
    import ast

    from kvsel import evidence as ev
    src = (ev.SPINTARE_REPO / "integrations" / "vllm" / "src" / "spintare_vllm" /
           "calib_mu.py").read_text(encoding="utf-8")
    neox = next((ast.literal_eval(n.value) for n in ast.parse(src).body
                 if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "NEOX_TYPES"
                 or isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "NEOX_TYPES"),
                None)
    assert neox is not None, "NEOX_TYPES not found in calib_mu.py"
    assert ev.KERNEL_MODEL_TYPES[("vllm", "spintare")] == frozenset(neox)
    vl = Deployment([Engine("vllm", spintare_kernel=True)], 32768)
    c = select(mk(1.0, model_type="glm"), vl).chosen
    assert c is not None and c.k_format == "bf16"


def test_llamacpp_decode_cost_matches_timing_file() -> None:
    import json

    from kvsel import evidence as ev
    f = ev.SPINTARE_REPO / "experiments" / "results" / "spintare" / "decode_overhead" / "res_sg2_time_agg.json"
    d = json.loads(f.read_text(encoding="utf-8"))
    assert len(d["models"]) == 2  # the F9 reason text says "2 models"
    v = [x for m in d["models"].values() for x in m["overhead_pct"]["k4v8mu_gate"].values()]
    lo, hi = ev.COST["llamacpp_spintare_decode"]
    assert abs(lo - min(v) / 100) < 5e-4 and abs(hi - max(v) / 100) < 5e-4


def test_llamacpp_cost_is_evidence_only_where_timed() -> None:
    dep = Deployment([Engine("llama.cpp", spintare_kernel=True)], 32768)
    s = select(mk(1.0, model_type="qwen3"), dep)
    by_fmt = {c.k_format: c for c in [s.chosen, *s.alternatives] if c is not None}
    f9 = {k: [r for r in c.reasons if r.rule == "F9"][0] for k, c in by_fmt.items()
          if c.centring == "spintare"}
    assert f9["q4_0"].status == "evidence" and f9["q4_1"].status == "to_test"
