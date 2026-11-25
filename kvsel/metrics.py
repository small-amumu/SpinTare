# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""Model metrics for the selector, read from one 16-bit probe pass.

The probe is SpinTare's kvprobe_unified.py (one forward, no quantisation in the model): it writes
per-layer effective bits b_meas for each (basis, format) in engine coordinates, the layer geometry
and the key-bias statistic. This module only reads that JSON (and optionally the HF config.json for
architecture flags the probe does not record); b_top2 is SpinTare's own function, imported
unchanged.
"""

from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from . import evidence as ev

_CC: ModuleType | None = None


def config_check() -> ModuleType:
    """Import SpinTare's config_check.py (b_top2, layer_bits, TAU_B) from the repository."""
    global _CC
    if _CC is None:
        spec = importlib.util.spec_from_file_location("spintare_config_check", ev.CONFIG_CHECK_PY)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {ev.CONFIG_CHECK_PY}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CC = mod
    return _CC


@dataclass
class ModelMetrics:
    name: str
    n_layers: int
    head_dim: int
    n_heads: int
    n_kv_heads: int
    rot_dim: int
    interleaved_rope: bool
    has_k_bias: bool
    bias_stat: float
    channel_order: str  # "hf" or "gguf" (llama.cpp permutes Llama-family q/k)
    b_top2: dict[str, float]  # "<basis>/<format>" -> b_top2
    worst_layers: dict[str, list[int]]
    nope_layers: list[int] = field(default_factory=list)
    sliding_window_layers: int = 0
    mla: bool = False
    qk_norm: bool = False
    params_b: float = 0.0
    # HF config "model_type" (selects the llama.cpp graph a patch must be wired into) and RoPE
    # scaling type (rope_scaling / rope_parameters "rope_type"); None model_type = unknown
    model_type: str | None = None
    rope_type: str | None = None  # None = no scaling, but only meaningful when rope_known
    # what is known beyond the probe: config_read = the architecture facts (NoPE / sliding-window
    # layers, RoPE scaling) come from a config.json or an equivalent table; rope_known = rope_type
    # is a known value (config read, or given explicitly)
    config_read: bool = False
    rope_known: bool = False

    @property
    def gqa_ratio(self) -> int:
        return self.n_heads // self.n_kv_heads

    @property
    def partial_rope(self) -> bool:
        return 0 < self.rot_dim < self.head_dim


def _layer_geometry(probe: dict[str, Any]) -> tuple[int, int, int]:
    """(head_dim, n_heads, n_kv_heads); every layer must agree (else the probe is not usable)."""
    geo = {(v["d"], v["H"], v["KVH"]) for v in probe["per_layer"].values()}
    if len(geo) != 1:
        raise ValueError(f"layers disagree on geometry: {sorted(geo)}")
    return next(iter(geo))


def _arch_flags(cfg: dict[str, Any], n_layers: int) -> dict[str, Any]:
    """Architecture flags from an HF config.json that the probe does not record."""
    out: dict[str, Any] = {}
    nrl = cfg.get("no_rope_layers")  # SmolLM3: 1 = RoPE, 0 = NoPE, one entry per layer
    if isinstance(nrl, list):
        if len(nrl) != n_layers:
            raise ValueError(f"no_rope_layers has {len(nrl)} entries for {n_layers} layers")
        out["nope_layers"] = [i for i, v in enumerate(nrl) if not v]
    lt = cfg.get("layer_types")  # Gemma-3 / Qwen3-Next style per-layer attention type
    if isinstance(lt, list):
        out["sliding_window_layers"] = sum(1 for t in lt if "sliding" in str(t))
    elif cfg.get("sliding_window") and cfg.get("use_sliding_window", True) and not (
            cfg.get("max_position_embeddings")
            and cfg["sliding_window"] >= cfg["max_position_embeddings"]):
        # a window at least as long as the maximum context is full attention (Phi-4-mini 262144)
        # HF semantics: Gemma-2 / Mistral-v0.1 set sliding_window without a flag and use it;
        # Qwen2 / Qwen3 set use_sliding_window = false explicitly
        out["sliding_window_layers"] = -1  # present, layer split unknown
    out["mla"] = bool(cfg.get("kv_lora_rank"))
    archs = cfg.get("architectures") or []
    out["qk_norm"] = bool(cfg.get("use_qk_norm")) or any(
        str(a).startswith("Qwen3ForCausalLM") for a in archs)  # reported only, not a rule input
    out["model_type"] = cfg.get("model_type")
    # rope_scaling (legacy) takes precedence over rope_parameters (transformers 5.x); a scaling
    # dict whose type cannot be read is "unparsed", which no kernel accepts
    rs = cfg.get("rope_scaling") or cfg.get("rope_parameters") or {}
    rt = None
    if isinstance(rs, dict):
        rt = rs.get("rope_type") or rs.get("type")
        if rt is None and set(rs) - {"rope_theta", "partial_rotary_factor"}:
            rt = "unparsed"
    elif rs:
        rt = "unparsed"
    out["rope_type"] = rt
    return out


def from_probe(
    probe_path: Path, config_path: Path | None = None, name: str | None = None,
    model_type: str | None = None, rope_type: str | None = None, rope_known: bool = False,
    arch_known: bool = False, config: dict[str, Any] | None = None,
) -> ModelMetrics:
    """model_type / rope_type: explicit overrides of the config values. rope_known=True states that
    rope_type (possibly None = no scaling) is known without a config; arch_known=True states that
    NoPE / sliding-window facts were supplied by the caller (e.g. a model-card table). config: an
    already-loaded config.json dict, used like config_path."""
    if config is not None and config_path is not None:
        raise ValueError("pass config_path or config, not both")
    probe = json.loads(Path(probe_path).read_text(encoding="utf-8"))
    cc = config_check()
    n = probe["layers_config"]
    if probe.get("layers_captured", n) != n:
        raise ValueError(f"probe captured {probe.get('layers_captured')} of {n} layers")
    d, h, kvh = _layer_geometry(probe)
    first = probe["per_layer"]["0"]
    keys = [k for k in first if isinstance(first[k], dict) and "b_meas" in first[k]]
    bt2: dict[str, float] = {}
    worst: dict[str, list[int]] = {}
    for key in keys:
        # unrounded value (config_check.check rounds to 4 decimals for printing)
        bits = cc.layer_bits(probe, key)
        b = float(cc.b_top2(bits))
        if not math.isfinite(b):
            raise ValueError(f"non-finite b_top2 for {key}")
        bt2[key] = b
        worst[key] = sorted(range(len(bits)), key=lambda i: bits[i])[:2]
    w = probe["weights"]
    flags: dict[str, Any] = {}
    if config_path is not None:
        if not Path(config_path).exists():
            raise FileNotFoundError(f"config.json not found: {config_path}")
        flags = _arch_flags(json.loads(Path(config_path).read_text(encoding="utf-8")), n)
    elif config is not None:
        flags = _arch_flags(config, n)
    if model_type is not None:
        flags["model_type"] = model_type
    if rope_type is not None:
        flags["rope_type"] = rope_type
    have_cfg = config_path is not None or config is not None
    flags["config_read"] = have_cfg or arch_known
    flags["rope_known"] = have_cfg or rope_known or rope_type is not None
    return ModelMetrics(
        name=name or Path(str(probe.get("model", probe_path))).name,
        n_layers=n,
        head_dim=d,
        n_heads=h,
        n_kv_heads=kvh,
        rot_dim=int(w.get("rot_dim") or d),
        interleaved_rope=bool(w.get("interleaved")),
        has_k_bias=bool(w.get("has_k_bias")),
        bias_stat=float(w.get("bias_stat") or 0.0),
        channel_order=str(probe.get("order", "hf")),
        b_top2=bt2,
        worst_layers=worst,
        params_b=float(probe.get("params_b") or 0.0),
        **flags,
    )
