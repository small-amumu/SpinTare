# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Probe -> configuration selector for a SpinTare K cache.

One quantised K/V format on the GPU; the configuration check decides per K format whether it runs
plain, with rotary centring + add-back + exemption gate, with the static post-RoPE mean, or is
refused (then the 8-bit / 16-bit fallback).

Rule ids (F*, C*) match docs/kvsel/README.md; each Reason says whether a rule rests on a
measurement or on a fact read from the engine / patch source ("evidence"), needs a per-model engine
run before serving ("check"), or has neither ("to_test").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from . import evidence as ev
from .metrics import ModelMetrics, config_check

GIB = 1024**3
Status = Literal["evidence", "check", "to_test"]


@dataclass
class Engine:
    name: Literal["llama.cpp", "vllm"]
    rotation: bool = True  # llama.cpp K-cache Hadamard (upstream default on)
    spintare_kernel: bool = False  # rotary add-back + exemption gate in the attention kernel
    postmean_kernel: bool = False  # static post-RoPE mean patch (llama.cpp)


@dataclass
class Deployment:
    engines: list[Engine]
    ctx_len: int
    batch: int = 1
    gpu_kv_gib: float = float("inf")  # GPU memory left for the KV cache after weights


@dataclass
class Reason:
    rule: str
    text: str
    status: Status


@dataclass
class Config:
    engine: str
    k_format: str
    v_format: str
    centring: Literal["none", "spintare", "postmean"]
    exemption_gate: bool | None
    basis: str
    gpu_kv_gib: float
    checks_required: list[str] = field(default_factory=list)
    reasons: list[Reason] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Selection:
    model: str
    chosen: Config | None
    alternatives: list[Config]
    rejected: list[dict[str, Any]]
    to_test: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "chosen": self.chosen.to_dict() if self.chosen else None,
            "alternatives": [c.to_dict() for c in self.alternatives],
            "rejected": self.rejected,
            "to_test": self.to_test,
        }


# ---------------------------------------------------------------- byte models
def flat_kv_gib(m: ModelMetrics, k_bpw: float, v_bpw: float, dep: Deployment) -> float:
    per_tok = m.n_layers * m.n_kv_heads * m.head_dim * (k_bpw + v_bpw) / 8.0
    return per_tok * dep.ctx_len * dep.batch / GIB


# ---------------------------------------------------------------- formats and rules
def _flat_formats(e: Engine, m: ModelMetrics) -> list[tuple[str, str, float, float]]:
    """(K format, probe key for b_top2, K bits/elem, V bits/elem); select() orders by GPU bytes."""
    if e.name == "llama.cpp":
        basis = "hadamard" if e.rotation else "identity"
        v = ev.FLAT_BPW["q8_0"]  # rule F7: V stays q8_0 (every SpinTare run held V at 8 bits)
        return [(f, f"{basis}/{f}", ev.FLAT_BPW[f], v) for f in ("q4_0", "q4_1", "q8_0")]
    if e.name == "vllm":  # vLLM quantises K and V with the same --kv-cache-dtype
        i4 = ev.pth_bpw(4, m.head_dim, 1)
        f8 = ev.pth_bpw(8, m.head_dim, 1)
        # bf16 is the vLLM fallback (the integer 8-bit fallback has no vLLM equivalent:
        # FP8 itself fails on Qwen2.5-7B, D 3.97)
        return [("int4_per_token_head", "identity/int4pth", i4, i4),
                ("fp8_per_token_head", "identity/fp8pth", f8, f8),
                ("bf16", "-", 16.0, 16.0)]
    return []


def _assess_flat(
    m: ModelMetrics, e: Engine, fmt: str, key: str, dep: Deployment, tau: float
) -> tuple[str, list[Reason]]:
    """-> ("plain" | "spintare" | "postmean" | "refuse", reasons)."""
    rs: list[Reason] = []
    if fmt == "q8_0":
        rs.append(Reason("F6", "8-bit K is the fallback of the configuration check", "evidence"))
        return "plain", rs
    if fmt == "bf16":
        rs.append(Reason("F6", "16-bit cache: vLLM fallback (no quantisation)", "evidence"))
        return "plain", rs
    proxy = None
    if key not in m.b_top2 and "hadamard/q4_0" not in m.b_top2:
        rs.append(Reason("F8", f"probe has neither {key} nor hadamard/q4_0", "to_test"))
        return "refuse", rs
    if key not in m.b_top2:
        # vLLM formats: the census probe has no int4pth / fp8 bits for most models; use the
        # llama.cpp q4_0 cell in the Hadamard basis as a proxy (rule F8, no fitted threshold)
        proxy = "hadamard/q4_0"
        rs.append(Reason("F8", f"no b_top2 for {key}; proxy {proxy} with tau_b of 32-wide blocks",
                         "to_test"))
    b = m.b_top2[proxy or key]
    if b > tau:
        rs.append(Reason("F1", f"b_top2 {b:.3f} > tau_b {tau:.3f}: run {fmt} plain, no centring",
                         "evidence" if proxy is None else "to_test"))
        rs.append(Reason("F5", "centring not enabled on an unflagged format: on Qwen3-4B / 8B it "
                         "lowered KL but gave no RULER difference and cost decode time (2 models); "
                         "forced centring raised KL 1.20-1.58x on low-offset models", "evidence"))
        return "plain", rs
    rs.append(Reason("F1", f"b_top2 {b:.3f} <= tau_b {tau:.3f}: {fmt} flagged (offset-dominated)",
                     "evidence" if proxy is None else "to_test"))
    odd = [n for n, flag in (("partial RoPE", m.partial_rope), ("interleaved RoPE",
           m.interleaved_rope), ("NoPE layers", bool(m.nope_layers))) if flag]
    if e.spintare_kernel:
        why = _kernel_gap(m, e.name, "spintare")
        if odd:
            # X5: the add-back is right when computed with the model's own RoPE, but no kernel
            # implements partial / interleaved / NoPE handling yet
            why.append(Reason("C6", f"no add-back kernel handles {', '.join(odd)} (X5: the formula "
                              "holds with the model's own RoPE; the kernel does not exist)",
                              "evidence"))
        if not why:
            rs.append(Reason("F2", "rotary centring with exact add-back; confirm engine KL "
                             f"D_f < {ev.TAU_ALLOW} before serving", "check"))
            rs.append(Reason("F3", "exemption gate on with centring (79/80 cells; no gate is worse "
                             "than no repair on low-offset rows)", "evidence"))
            return "spintare", rs
        refused = why
    else:
        refused = []
    if e.postmean_kernel:
        why = _kernel_gap(m, e.name, "postmean")
        if odd:
            why.append(Reason("C6", f"static mean not validated with {', '.join(odd)}", "to_test"))
        if dep.ctx_len > ev.CALIB_TOKENS:
            why.append(Reason("F4", f"static mean refused: ctx {dep.ctx_len} > calibration "
                              f"{ev.CALIB_TOKENS} (KL grows 163x at 4-8K)", "evidence"))
        if not why:
            rs.append(Reason("F4", "static post-RoPE mean only inside its 2K calibration window",
                             "check"))
            return "postmean", rs  # the add-back refusal is not part of this config's reasons
        refused += why
    return "refuse", rs + refused


def _kernel_gap(m: ModelMetrics, engine: str, repair: str) -> list[Reason]:
    """Reasons the engine's repair kernel does not cover this model (empty list = covered).
    Coverage is per (engine, repair): the model graphs a patch is wired into and the RoPE scaling
    types its decode path accepts (rule C13; evidence.KERNEL_*). Unknown model_type on an engine
    whose patch is per-graph is treated as not covered, and so is an unknown RoPE scaling type.
    Status "evidence" here means a fact read from the patch source (not a measurement)."""
    out: list[Reason] = []
    types = ev.KERNEL_MODEL_TYPES.get((engine, repair))
    if types is not None:
        if m.model_type is None:
            out.append(Reason("C13", f"{engine} {repair} patch is wired per model graph; "
                              "model_type "
                              "unknown (pass --config)", "to_test"))
        elif m.model_type not in types:
            out.append(Reason("C13", f"{engine} {repair} patch is not wired into the "
                              f"'{m.model_type}' graph (covers {', '.join(sorted(types))})",
                              "evidence"))
    ropes = ev.KERNEL_ROPE_TYPES.get((engine, repair))
    if ropes is not None and not m.rope_known:
        out.append(Reason("C13", f"{engine} {repair}: RoPE scaling unknown (pass --config or "
                          "--rope-type)", "to_test"))
    elif ropes is not None and m.rope_type not in ropes:
        out.append(Reason("C13", f"{engine} {repair} kernel does not handle RoPE scaling "
                          f"'{m.rope_type}'", "evidence"))
    return out


def _flat_configs(m: ModelMetrics, e: Engine, dep: Deployment, tau: float) -> tuple[
        list[Config], list[dict[str, Any]]]:
    ok: list[Config] = []
    rejected: list[dict[str, Any]] = []
    basis = {"llama.cpp": "hadamard" if e.rotation else "identity", "vllm": "rht"}.get(e.name, "-")
    for fmt, key, kb, vb in _flat_formats(e, m):
        kind, rs = _assess_flat(m, e, fmt, key, dep, tau)
        vfmt = "q8_0" if e.name == "llama.cpp" else fmt
        if kind == "refuse":
            rejected.append({"engine": e.name, "k_format": fmt, "why": [asdict(r) for r in rs]})
            continue
        if e.name == "vllm" and fmt.startswith("fp8") and kind == "plain":
            rs.append(Reason("F8", "FP8 has no fitted b_top2; it follows the INT4 proxy decision "
                             "(Qwen2.5-7B FP8 fails too: llama.cpp-check D 3.97, vLLM KL 4.86)",
                             "to_test"))
        if e.name == "vllm" and kind == "spintare":
            cost = ev.COST["vllm_fp8_addback_tput" if fmt.startswith("fp8") else
                           "vllm_int4_addback_tput"]
            rs.append(Reason("F9", f"add-back throughput {cost:.3f}x of stock (batch)", "evidence"))
        if e.name == "llama.cpp" and kind == "spintare":
            lo, hi = ev.COST["llamacpp_spintare_decode"]
            timed = (fmt == ev.LLAMACPP_COST_TIMED_K
                     and m.model_type in ev.LLAMACPP_COST_TIMED_GRAPHS)
            rs.append(Reason("F9", f"decode +{lo:.1%}..+{hi:.1%} per token with add-back + gate "
                             "(K q4_0 / V q8_0, Qwen2.5-7B and Qwen3-1.7B, RTX 5090 D, batch 1)"
                             + ("" if timed else f"; not timed for K {fmt} on '{m.model_type}'"),
                             "evidence" if timed else "to_test"))
        gib = flat_kv_gib(m, kb, vb, dep)
        cfg = Config(
            engine=e.name, k_format=fmt, v_format=vfmt,
            centring="spintare" if kind == "spintare" else ("postmean" if kind == "postmean"
                                                            else "none"),
            exemption_gate=True if kind == "spintare" else None,
            basis=basis, gpu_kv_gib=gib, reasons=rs,
            checks_required=([f"spintare check: engine KL D_f < {ev.TAU_ALLOW}"]
                             if kind in ("spintare", "postmean") else []),
        )
        if gib > dep.gpu_kv_gib:
            rejected.append({"engine": e.name, "k_format": fmt,
                             "why": f"KV {gib:.2f} GiB > GPU budget {dep.gpu_kv_gib:.2f} GiB"})
            continue
        ok.append(cfg)
    return ok, rejected


# ---------------------------------------------------------------- top level
def select(m: ModelMetrics, dep: Deployment) -> Selection:
    tau = float(config_check().TAU_B)
    ok: list[Config] = []
    rejected: list[dict[str, Any]] = []
    if m.mla:
        rejected.append({"mode": "all", "why": "MLA latent cache: not covered"})
    else:
        for e in dep.engines:
            if e.name not in ("llama.cpp", "vllm"):
                raise ValueError(f"unknown engine {e.name!r}")
            c, r = _flat_configs(m, e, dep, tau)
            ok += c
            rejected += r
    # rule C1: the safe config with the fewest GPU bytes that fits the budget.
    # sorted() is stable: equal bytes keep the engine / format order.
    order = sorted(ok, key=lambda c: c.gpu_kv_gib)
    chosen = order[0] if order else None
    if chosen is not None:
        rank = {"evidence": 0, "check": 1, "to_test": 2}
        weakest: Status = "evidence"
        for r in chosen.reasons:  # C1 is only as strong as the weakest rule behind the choice
            if rank[r.status] > rank[weakest]:
                weakest = r.status
        chosen.reasons.insert(0, Reason("C1", "first safe config by GPU bytes that fits the "
                                        "budget", weakest))
    to_test = sorted({f"{r.rule}: {r.text}" for c in order for r in c.reasons
                      if r.status == "to_test"})
    return Selection(m.name, chosen, order[1:], rejected, to_test)
