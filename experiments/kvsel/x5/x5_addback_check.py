# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""X5: rotary centring with exact add-back on partial / interleaved RoPE and NoPE layers (CPU only).

One bf16 forward on a calibration text (pre-RoPE key mean mu per layer and KV head, as SpinTare's
calibration: all tokens, tokenizer defaults) and one on the evaluation text (pre-RoPE q / k, the
model's own cos / sin, and the eager attention weights of the last --nq queries). Analysis in float32.

Rotations
  R_true    the model's own apply_rotary_pos_emb with the captured cos / sin (identity on layers that
            skip RoPE). This is what the model computes.
  R_param   a parametric rotation rebuilt from the model's rotary module (inv_freq, rotary width, pair
            layout, attention scaling); with every factor right it must match R_true up to the bf16
            rounding of the model's cos / sin (the floor). Variants with ONE factor wrong (whole-head
            width, NeoX vs interleaved pairing, no scaling) and with all wrong (whole-head NeoX, no
            scaling) model a hypothetical kernel that does not read the model's RoPE settings. On layers
            that skip RoPE, "rotated" applies whole-head NeoX to mu although the model does not rotate.
            These are not the existing llama.cpp patch: it is wired only into qwen2 / qwen3 / gptneox,
            and its gated decode path asserts whole-head NeoX (otherwise it adds R(p)mu to keys with
            ggml_rope_ext, i.e. the model's own settings).
Arms (K only, V exact, causal, last --nq queries; g = 1[z < 0.5], z = 1 - k.mu / |mu|^2, SpinTare)
  exact             R_true(k)
  plain             Q(R_true(k))
  ab_gate           Q(R_true(k - g mu)) + g q.R_true(p) mu
  ab_nogate         Q(R_true(k - mu)) + q.R_true(p) mu
  ab_gate_mu_intext ab_gate with mu from the evaluation text itself
  ab_gate_mu_other  ab_gate with mu of another layer ((i + 1) mod n_layers): control for "any vector"
  centre_noadd      Q(R_true(k - g mu)), no add-back: control for the add-back term
  ab_gate_<variant> ab_gate whose add-back uses an R_param variant
Q = ggml quantise-dequantise (SpinTare ggml_ref.quant, q4_0 / q4_1) in engine coordinates (--order
channel order, then whole-head Hadamard). Unquantised checks (prefix id_): ab_gate (equal to exact by
linearity: a sanity check, not a finding), every R_param variant, and the eager-weight reference.
Usage: python x5_addback_check.py --model DIR --order hf|gguf --calib FILE --text FILE --out JSON
           [--seq 2048] [--nq 256] [--threads 16] [--focus 21,25] [--random-layers N]
       python x5_addback_check.py --selftest
--random-layers N: smoke mode, the model's config with N layers and random weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ggml_ref import gguf_perm, hadamard, n_rot_k, quant  # noqa: E402

Rot = Callable[[torch.Tensor], torch.Tensor]  # [heads, L, d] -> same, positions 0..L-1
FMTS = ("q4_0", "q4_1")
VARIANTS = ("param_true", "naive_freq", "naive_width", "naive_pairing", "naive_scale", "naive_all")


# ---------------------------------------------------------------- rotations
@dataclass
class RopeSpec:
    inv_freq: torch.Tensor  # [rot / 2], float64
    interleaved: bool
    scale: float


def param_rot(spec: RopeSpec, d: int, pos: torch.Tensor) -> Rot:
    """Rotate the first 2 * len(inv_freq) dims (pass the rest), NeoX (j, j + rot/2) or interleaved
    (2j, 2j + 1) pairs, cos / sin multiplied by spec.scale."""
    rot = 2 * spec.inv_freq.numel()
    assert 0 < rot <= d
    ang = pos.to(torch.float64)[:, None] * spec.inv_freq[None, :]
    c, s = ang.cos() * spec.scale, ang.sin() * spec.scale
    if spec.interleaved:
        c, s = c.repeat_interleave(2, -1).float(), s.repeat_interleave(2, -1).float()
    else:
        c, s = torch.cat([c, c], -1).float(), torch.cat([s, s], -1).float()

    def f(x: torch.Tensor) -> torch.Tensor:
        xr, xp = x[..., :rot], x[..., rot:]
        if spec.interleaved:
            rh = torch.stack((-xr[..., 1::2], xr[..., 0::2]), -1).flatten(-2)
        else:
            h = rot // 2
            rh = torch.cat([-xr[..., h:], xr[..., :h]], -1)
        return torch.cat([xr * c + rh * s, xp], -1)

    return f


def variants(true: RopeSpec, d: int, theta: float) -> dict[str, RopeSpec]:
    full = theta ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    rot = 2 * true.inv_freq.numel()
    plain_freq = theta ** (-torch.arange(0, rot, 2, dtype=torch.float64) / rot)
    return {
        "param_true": true,
        # same width / pairing / scale, frequencies without per-dimension factors (LongRoPE etc.)
        "naive_freq": RopeSpec(plain_freq, true.interleaved, true.scale),
        "naive_width": RopeSpec(full, true.interleaved, true.scale),
        "naive_pairing": RopeSpec(true.inv_freq, not true.interleaved, true.scale),
        "naive_scale": RopeSpec(true.inv_freq, true.interleaved, 1.0),
        "naive_all": RopeSpec(full, False, 1.0),
    }


# ---------------------------------------------------------------- numerics
def to_engine(x: torch.Tensor, order: str, had: torch.Tensor) -> torch.Tensor:
    if order == "gguf":
        x = x[..., gguf_perm(x.shape[-1])]
    return x @ had


def scores(q: torch.Tensor, k: torch.Tensor, group: int, scale: float) -> torch.Tensor:
    """q [H, nq, d], k [kvh, L, d] -> [H, nq, L] (query head h uses KV head h // group)."""
    return torch.einsum("hqd,hld->hql", q, k.repeat_interleave(group, 0)) * scale


def bias(q: torch.Tensor, rmu: torch.Tensor, g: torch.Tensor, group: int, scale: float
         ) -> torch.Tensor:
    """g * q . R(p) mu per (head, query, key): rmu [kvh, L, d], g [kvh, L]."""
    return scores(q, rmu, group, scale) * g.repeat_interleave(group, 0)[:, None, :]


def compare(s_ex: torch.Tensor, s: torch.Tensor, qpos: torch.Tensor) -> dict[str, float]:
    """KL(p_exact || p_arm) and the row-centred (softmax-invariant) relative score error."""
    L = s_ex.shape[-1]
    valid = torch.arange(L)[None, :] <= qpos[:, None]  # [nq, L]
    neg = torch.finfo(torch.float32).min
    a = s_ex.masked_fill(~valid, neg).log_softmax(-1)
    b = s.masked_fill(~valid, neg).log_softmax(-1)
    kl = (a.exp() * (a - b)).masked_fill(~valid, 0.0).sum(-1)
    m = valid.float()
    cnt = m.sum(-1, keepdim=True).clamp_min(1.0)

    def centre(x: torch.Tensor) -> torch.Tensor:
        return (x - (x * m).sum(-1, keepdim=True) / cnt) * m

    err = centre(s - s_ex)
    return {"kl": float(kl.mean()), "rel_err": float(err.norm() / centre(s_ex).norm().clamp_min(
        1e-30)), "max_abs": float(err.abs().max())}


def random_like(mu: torch.Tensor, seed: int) -> torch.Tensor:
    """Norm-matched random direction per KV head (control: a vector that is not the offset)."""
    gen = torch.Generator().manual_seed(seed)
    r = torch.randn(mu.shape, generator=gen)
    return r / r.norm(dim=-1, keepdim=True) * mu.norm(dim=-1, keepdim=True)


def analyse_layer(kpre: torch.Tensor, mu: torch.Tensor, mu_in: torch.Tensor, mu_other: torch.Tensor,
                  qpost: torch.Tensor, rtrue: Rot, rvar: dict[str, Rot], order: str, group: int,
                  scale: float, qpos: torch.Tensor, seed: int = 0) -> dict[str, Any]:
    d, L = kpre.shape[-1], kpre.shape[1]
    had = hadamard(d, kpre.device)
    k_ex = rtrue(kpre)
    s_ex = scores(qpost, k_ex, group, scale)

    def gate(m: torch.Tensor) -> torch.Tensor:
        z = 1.0 - (kpre * m[:, None, :]).sum(-1) / (m * m).sum(-1, keepdim=True).clamp_min(1e-30)
        return (z < 0.5).float()

    def write(m: torch.Tensor, g: torch.Tensor) -> torch.Tensor:  # R_true(k - g mu)
        return rtrue(kpre - g[..., None] * m[:, None, :])

    def rmu(r: Rot, m: torch.Tensor) -> torch.Tensor:
        return r(m[:, None, :].expand(-1, L, -1))

    g = gate(mu)
    kc = write(mu, g)
    b_true = bias(qpost, rmu(rtrue, mu), g, group, scale)
    b_var = {v: bias(qpost, rmu(r, mu), g, group, scale) for v, r in rvar.items()}
    out: dict[str, Any] = {
        "offset_energy_share": float((mu * mu).sum(-1).mean() / (kpre * kpre).sum(-1).mean()),
        "gate_exempt_frac": float(1.0 - g.mean()),
        "id_ab_gate": compare(s_ex, scores(qpost, kc, group, scale) + b_true, qpos),
    }
    s_kc_raw = scores(qpost, kc, group, scale)
    for v, b in b_var.items():
        out[f"id_{v}"] = compare(s_ex, s_kc_raw + b, qpos)
    del s_kc_raw
    qe = to_engine(qpost, order, had)

    def sq(k: torch.Tensor, fmt: str) -> torch.Tensor:
        return scores(qe, quant(to_engine(k, order, had), fmt), group, scale)

    mu_rnd = random_like(mu, seed)
    g_in, g_ot, g_rn = gate(mu_in), gate(mu_other), gate(mu_rnd)
    out["gate_exempt_frac_mu_random"] = float(1.0 - g_rn.mean())
    out["gate_exempt_frac_mu_other"] = float(1.0 - g_ot.mean())
    ones = torch.ones_like(g)
    for fmt in FMTS:
        s_kc = sq(kc, fmt)
        arms = {
            "ab_gate_mu_random": sq(write(mu_rnd, g_rn), fmt) + bias(qpost, rmu(rtrue, mu_rnd),
                                                                      g_rn, group, scale),
            # ungated: every row shifted by the random vector (the gated arm mostly exempts rows)
            "ab_nogate_mu_random": sq(write(mu_rnd, ones), fmt) + bias(qpost, rmu(rtrue, mu_rnd),
                                                                        ones, group, scale),
            "plain": sq(k_ex, fmt),
            "ab_gate": s_kc + b_true,
            "ab_nogate": sq(write(mu, ones), fmt) + bias(qpost, rmu(rtrue, mu), ones, group,
                                                         scale),
            "ab_gate_mu_intext": sq(write(mu_in, g_in), fmt) + bias(qpost, rmu(rtrue, mu_in), g_in,
                                                                     group, scale),
            "ab_gate_mu_other": sq(write(mu_other, g_ot), fmt) + bias(qpost, rmu(rtrue, mu_other),
                                                                       g_ot, group, scale),
            "centre_noadd": s_kc,
        }
        res = {k: compare(s_ex, v, qpos) for k, v in arms.items()}
        del arms
        for v, b in b_var.items():
            res[f"ab_gate_{v}"] = compare(s_ex, s_kc + b, qpos)
        out[fmt] = res
    return out


# ---------------------------------------------------------------- capture
class Capture:
    """Per decoder layer: pre-RoPE q (last nq) / k and cos / sin through the modeling module's
    apply_rotary_pos_emb (RoPE layers), q / k from q_proj / k_proj or qkv_proj hooks (every layer;
    the only source on layers that skip RoPE), and the eager attention weights of the last nq rows."""

    def __init__(self, model: torch.nn.Module, nq: int, nh: int, kvh: int, d: int,
                 keep_attw: bool) -> None:
        self.nq, self.nh, self.kvh, self.d = nq, nh, kvh, d
        self.keep_attw = keep_attw
        self.cur = -1
        self.order: list[int] = []
        self.rope: dict[int, dict[str, torch.Tensor]] = {}
        self.proj: dict[int, dict[str, torch.Tensor]] = {}
        self.attw: dict[int, torch.Tensor] = {}
        self.mod = sys.modules[model.__class__.__module__]
        self.orig = self.mod.apply_rotary_pos_emb
        self.orig_eager = self.mod.eager_attention_forward
        self.handles: list[Any] = []
        try:
            for i, layer in enumerate(model.model.layers):
                at = layer.self_attn
                self.handles.append(at.register_forward_pre_hook(self._enter(i)))
                for name in ("q_proj", "k_proj", "qkv_proj"):
                    if hasattr(at, name):
                        self.handles.append(
                            getattr(at, name).register_forward_hook(self._proj(i, name)))
        except Exception:
            self.close()
            raise
        self.mod.apply_rotary_pos_emb = self._wrap
        self.mod.eager_attention_forward = self._eager

    def _enter(self, i: int) -> Callable[..., None]:
        def f(module: torch.nn.Module, args: Any) -> None:
            self.cur = i
            self.order.append(i)
        return f

    def _proj(self, i: int, name: str) -> Callable[..., None]:
        nq, nh, kvh, d = self.nq, self.nh, self.kvh, self.d

        def f(module: torch.nn.Module, args: Any, out: torch.Tensor) -> None:
            o = out.detach()[0]  # [L, width], bf16
            p = self.proj.setdefault(i, {})
            if name == "qkv_proj":  # Phi3 layout: q | k | v
                p["q"] = o[-nq:, : nh * d].float().clone()
                p["k"] = o[:, nh * d: (nh + kvh) * d].float().clone()
            elif name == "q_proj":
                p["q"] = o[-nq:].float().clone()
            else:
                p["k"] = o.float().clone()
        return f

    def _wrap(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
              *a: Any, **kw: Any) -> Any:
        if self.cur in self.rope:
            raise RuntimeError(f"apply_rotary_pos_emb called twice in layer {self.cur}")
        res = self.orig(q, k, cos, sin, *a, **kw)
        self.rope[self.cur] = {"q": q.detach()[0, :, -self.nq:].float().clone(),
                               "k": k.detach()[0].float().clone(),
                               "cos": cos.detach().float().clone(),
                               "sin": sin.detach().float().clone()}
        return res

    def _eager(self, module: Any, query: Any, key: Any, value: Any, *a: Any, **kw: Any) -> Any:
        out = self.orig_eager(module, query, key, value, *a, **kw)
        if self.cur in self.attw:
            raise RuntimeError(f"eager attention called twice in layer {self.cur}")
        self.attw[self.cur] = (out[1].detach()[0, :, -self.nq:].clone() if self.keep_attw
                               else torch.empty(0))  # [H, nq, L], model dtype
        return out

    def close(self) -> None:
        self.mod.apply_rotary_pos_emb = self.orig
        self.mod.eager_attention_forward = self.orig_eager
        for h in self.handles:
            h.remove()
        self.handles = []

    def qk(self, i: int) -> tuple[torch.Tensor, torch.Tensor, bool, float | None]:
        """(q_pre last nq [H, nq, d], k_pre [kvh, L, d], uses_rope, max |rope input - proj|)."""
        p = self.proj.get(i, {})
        pq = p["q"].view(-1, self.nh, self.d).transpose(0, 1) if "q" in p else None
        pk = p["k"].view(-1, self.kvh, self.d).transpose(0, 1) if "k" in p else None
        if i in self.rope:
            r = self.rope[i]
            dev = None
            if pq is not None and pk is not None:
                dev = max(float((pq - r["q"]).abs().max()), float((pk - r["k"]).abs().max()))
            return r["q"], r["k"], True, dev
        if pq is None or pk is None:
            raise RuntimeError(f"layer {i}: neither a RoPE call nor q / k projections captured")
        return pq, pk, False, None


def run_text(model: torch.nn.Module, tok: Any, path: Path, seq: int, nq: int, dims: tuple[int, int,
             int], keep_attw: bool) -> tuple[Capture, list[int]]:
    ids = tok(path.read_text(encoding="utf-8", errors="replace")).input_ids[:seq]
    if len(ids) < seq:
        raise SystemExit(f"{path} has only {len(ids)} tokens")
    cap = Capture(model, nq, *dims, keep_attw=keep_attw)
    try:
        with torch.no_grad():
            model(torch.tensor([ids]), use_cache=False, logits_to_keep=1)
    finally:
        cap.close()
    nl = len(model.model.layers)
    if cap.order != list(range(nl)):
        raise RuntimeError(f"attention order {cap.order[:8]}... is not 0..{nl - 1}")
    if sorted(cap.attw) != list(range(nl)):  # the eager path must have run in every layer
        raise RuntimeError(f"eager attention captured in {len(cap.attw)} of {nl} layers")
    return cap, ids[:4]


def rope_spec(model: torch.nn.Module, d: int) -> RopeSpec:
    """Read the rotation from the model: inv_freq and scaling from its rotary module, the pair layout
    from its own rotate_half (interleaved pairs (2j, 2j + 1) vs NeoX halves)."""
    re = model.model.rotary_emb
    rh = sys.modules[model.__class__.__module__].rotate_half
    x = torch.arange(1.0, 9.0)
    if torch.equal(rh(x), torch.stack((-x[1::2], x[0::2]), -1).flatten()):
        inter = True
    elif torch.equal(rh(x), torch.cat([-x[4:], x[:4]])):
        inter = False
    else:
        raise RuntimeError("unknown rotate_half layout")
    return RopeSpec(re.inv_freq.detach().to(torch.float64).cpu(), inter,
                    float(getattr(re, "attention_scaling", 1.0)))


# ---------------------------------------------------------------- main
def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def clean(x: Any) -> Any:
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


def main_model(a: argparse.Namespace) -> None:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import transformers

    out_path = Path(a.out)
    for p in (Path(a.calib), Path(a.text), out_path.parent):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    if not 1 <= a.nq <= a.seq:
        raise SystemExit("need 1 <= nq <= seq")
    # every check that needs only the config runs before the weights are loaded
    cfg0 = AutoConfig.from_pretrained(a.model)
    nl0 = a.random_layers or cfg0.num_hidden_layers
    if a.random_layers > cfg0.num_hidden_layers:
        raise SystemExit("--random-layers exceeds the model's layer count")
    d0 = getattr(cfg0, "head_dim", None) or cfg0.hidden_size // cfg0.num_attention_heads
    if n_rot_k(d0) != d0:
        raise SystemExit(f"head_dim {d0}: llama.cpp would not rotate the whole head")
    if any(not 0 <= f < nl0 for f in a.focus):
        raise SystemExit(f"focus layers {a.focus} outside 0..{nl0 - 1}")
    omax = getattr(cfg0, "original_max_position_embeddings", None) or (
        getattr(cfg0, "rope_parameters", None) or {}).get("original_max_position_embeddings")
    if omax and a.seq > omax:
        raise SystemExit("seq beyond original_max_position_embeddings switches LongRoPE factors")
    torch.set_num_threads(a.threads)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    if a.random_layers:
        cfg = cfg0
        cfg.num_hidden_layers = a.random_layers
        for key in ("no_rope_layers", "layer_types"):
            if isinstance(getattr(cfg, key, None), list):
                setattr(cfg, key, getattr(cfg, key)[: a.random_layers])
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16,
                                                 attn_implementation="eager").eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                     attn_implementation="eager").eval()
    prm = next(model.parameters())
    if prm.device.type != "cpu" or prm.dtype != torch.bfloat16:
        raise SystemExit(f"model on {prm.device} / {prm.dtype}, expected cpu / bfloat16")
    mod = sys.modules[model.__class__.__module__]
    if not all(hasattr(mod, n) for n in ("apply_rotary_pos_emb", "eager_attention_forward",
                                         "rotate_half")):
        raise SystemExit(f"{mod.__name__} lacks the functions this check patches")
    cfg = model.config
    nl, nh, kvh = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
    d = d0
    rp = getattr(cfg, "rope_parameters", None) or {}
    theta = float(getattr(cfg, "rope_theta", None) or rp["rope_theta"])
    group = nh // kvh
    spec = rope_spec(model, d)
    cal, cal_head = run_text(model, tok, Path(a.calib), a.seq, a.nq, (nh, kvh, d), False)
    mus = {i: cal.qk(i)[1].mean(1) for i in range(nl)}  # [kvh, d]
    del cal
    ev, ev_head = run_text(model, tok, Path(a.text), a.seq, a.nq, (nh, kvh, d), True)
    L = a.seq
    pos = torch.arange(L)
    qpos = pos[-a.nq:]
    rvar_all = {v: param_rot(s, d, pos) for v, s in variants(spec, d, theta).items()}
    res: dict[str, Any] = {
        "model": a.model, "random_layers": a.random_layers, "order": a.order, "seq": L,
        "nq": a.nq, "layers": nl, "heads": nh, "kv_heads": kvh, "head_dim": d,
        "rope_theta": theta, "rot_dim": 2 * spec.inv_freq.numel(),
        "interleaved": spec.interleaved, "attention_scaling": spec.scale, "focus": a.focus,
        "first_ids_calib": cal_head, "first_ids_text": ev_head,
        "script_md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
        "env": {"torch": torch.__version__, "transformers": transformers.__version__,
                "threads": torch.get_num_threads(), "machine": platform.machine()},
        "per_layer": {}}
    for i in range(nl):
        qpre, kpre, uses_rope, proj_dev = ev.qk(i)
        scale = float(model.model.layers[i].self_attn.scaling)
        if uses_rope:
            r = ev.rope[i]
            cos, sin, orig = r["cos"], r["sin"], ev.orig

            def rtrue(x: torch.Tensor, cos: torch.Tensor = cos, sin: torch.Tensor = sin,
                      orig: Any = orig) -> torch.Tensor:
                return orig(x[None], x[None], cos, sin)[1][0]

            qpost = orig(qpre[None], qpre[None], cos[:, -a.nq:], sin[:, -a.nq:])[0][0]
            rvar = rvar_all
        else:
            def rtrue(x: torch.Tensor) -> torch.Tensor:
                return x

            qpost = qpre
            rvar = {"param_true": rtrue, "naive_rotated": rvar_all["naive_all"]}
        out = analyse_layer(kpre, mus[i], kpre.mean(1), mus[(i + 1) % nl], qpost, rtrue, rvar,
                            a.order, group, scale, qpos, seed=1000 + i)
        # reference: my exact scores vs the model's own eager attention weights (non-circular).
        # The model's weights are bf16 (softmax in fp32, then cast), so report the max abs
        # difference next to the bf16-rounded version of mine, and the KL(eager || mine), which is
        # read against the q4 KL of the same layer.
        w = ev.attw[i].float()
        s_ex = scores(qpost, rtrue(kpre), group, scale)
        valid = torch.arange(L)[None, :] <= qpos[:, None]
        p_me = s_ex.masked_fill(~valid, torch.finfo(torch.float32).min).softmax(-1)
        lw = w.clamp_min(1e-30).log()
        kl_ref = (w * (lw - p_me.clamp_min(1e-30).log())).masked_fill(~valid, 0.0).sum(-1)
        out["eager_ref"] = {
            "max_abs": float((p_me - w).abs().max()),
            "max_abs_vs_bf16_rounded_mine": float((p_me.bfloat16().float() - w).abs().max()),
            "kl": float(kl_ref.mean()),
        }
        out.update({"uses_rope": uses_rope, "proj_vs_rope_input_max_abs": proj_dev})
        res["per_layer"][str(i)] = clean(out)
        if i % 4 == 3 or i == nl - 1:
            write_json(out_path, res)
    res["seconds"] = round(time.time() - t0, 1)
    res["summary"] = clean(summarise(res))
    write_json(out_path, res)
    print(json.dumps(res["summary"], indent=1))


def summarise(res: dict[str, Any]) -> dict[str, Any]:
    pl = res["per_layer"]
    med = statistics.median

    def group_stats(keys: list[str]) -> dict[str, Any]:
        if not keys:
            return {}
        s: dict[str, Any] = {"layers": len(keys)}
        ids = sorted({k for i in keys for k in pl[i] if k.startswith("id_")})
        for k in ids:
            s[k] = {"kl_median": med(pl[i][k]["kl"] for i in keys),
                    "kl_max": max(pl[i][k]["kl"] for i in keys),
                    "rel_err_max": max(pl[i][k]["rel_err"] for i in keys)}
        for fmt in FMTS:
            plain = {i: pl[i][fmt]["plain"]["kl"] for i in keys}
            s[f"{fmt}/plain_kl"] = {"median": med(plain.values()), "max": max(plain.values())}
            tiny = [i for i in keys if plain[i] < 1e-9]
            s[f"{fmt}/plain_kl_below_1e-9"] = len(tiny)
            for arm in sorted(pl[keys[0]][fmt]):
                if arm == "plain":
                    continue
                r = [pl[i][fmt][arm]["kl"] / plain[i] for i in keys if i not in tiny]
                if r:
                    s[f"{fmt}/{arm}_over_plain"] = {"median": med(r), "min": min(r), "max": max(r),
                                                    "lower": sum(x < 1 for x in r), "n": len(r)}
        return s

    rope = [i for i, v in pl.items() if v["uses_rope"]]
    nope = [i for i, v in pl.items() if not v["uses_rope"]]
    eag = [v["eager_ref"] for v in pl.values()]
    prj = [v["proj_vs_rope_input_max_abs"] for v in pl.values()
           if v.get("proj_vs_rope_input_max_abs") is not None]
    tiny = {fmt: [int(i) for i, v in pl.items() if v[fmt]["plain"]["kl"] < 1e-9] for fmt in FMTS}
    return {
        "rope_layers": group_stats(rope), "nope_layers": group_stats(nope),
        "nope_layer_ids": [int(i) for i in nope], "plain_kl_below_1e-9_layers": tiny,
        "eager_ref": {"max_abs_max": max(e["max_abs"] for e in eag),
                      "max_abs_vs_bf16_rounded_max": max(e["max_abs_vs_bf16_rounded_mine"]
                                                         for e in eag),
                      "kl_median": med(e["kl"] for e in eag), "kl_max": max(e["kl"] for e in eag),
                      "kl_over_q4_0_plain_kl_max": max(
                          v["eager_ref"]["kl"] / max(v["q4_0"]["plain"]["kl"], 1e-30)
                          for v in pl.values()),
                      "layers": len(eag)},
        "proj_vs_rope_input_max_abs_max": max(prj) if prj else None,
        "focus": {i: {fmt: {arm: pl[str(i)][fmt][arm]["kl"] for arm in pl[str(i)][fmt]}
                      for fmt in FMTS} for i in res["focus"]},
    }


# ---------------------------------------------------------------- self-test
def selftest() -> None:
    torch.manual_seed(0)
    d, kvh, H, L, nq = 128, 2, 4, 96, 8
    pos = torch.arange(L)
    qpos = pos[-nq:]
    from transformers.models.glm import modeling_glm
    from transformers.models.phi3 import modeling_phi3

    def hf_rot(fn: Any, inv: torch.Tensor, scale: float) -> Rot:
        """The HF function with cos / sin built as the rotary modules build them."""
        ang = pos.to(torch.float64)[:, None] * inv[None, :]
        emb = torch.cat([ang, ang], -1)
        cos, sin = (emb.cos() * scale).float()[None], (emb.sin() * scale).float()[None]
        return lambda x: fn(x[None], x[None], cos, sin)[1][0]

    theta = 10000.0
    cases = {  # name: (HF function, rotary width, interleaved, scale, per-dimension factors)
        "neox_full": (modeling_phi3.apply_rotary_pos_emb, d, False, 1.0, False),
        "phi_partial_scaled": (modeling_phi3.apply_rotary_pos_emb, 96, False, 1.19, False),
        "phi_partial_longrope": (modeling_phi3.apply_rotary_pos_emb, 96, False, 1.19, True),
        "glm_partial_interleaved": (modeling_glm.apply_rotary_pos_emb, 64, True, 1.0, False),
    }
    kpre = torch.randn(kvh, L, d) + 6.0 * torch.randn(kvh, 1, d)  # large shared offset
    kpre[:, 0] = 0.1 * torch.randn(kvh, d)  # sink-like row without the offset
    mu = kpre[:, 1:].mean(1)
    for name, (fn, rot, inter, sc, fac) in cases.items():
        inv = theta ** (-torch.arange(0, rot, 2, dtype=torch.float64) / rot)
        if fac:  # LongRoPE-style ext factors: inv_freq / factor per dimension
            inv = inv / torch.linspace(1.0, 8.0, rot // 2, dtype=torch.float64)
        rtrue = hf_rot(fn, inv, sc)
        spec = RopeSpec(inv, inter, sc)
        rvar = {v: param_rot(s, d, pos) for v, s in variants(spec, d, theta).items()}
        x = torch.randn(kvh, L, d)
        dev = float((rvar["param_true"](x) - rtrue(x)).norm() / rtrue(x).norm())
        assert dev < 1e-5, (name, "param_true vs HF", dev)
        qpost = hf_rot(fn, inv, sc)(torch.randn(H, L, d))[:, -nq:]
        out = analyse_layer(kpre, mu, kpre.mean(1), mu.flip(0), qpost, rtrue, rvar, "hf", H // kvh,
                            d ** -0.5, qpos)
        assert out["id_ab_gate"]["rel_err"] < 1e-5, (name, out["id_ab_gate"])
        assert out["id_param_true"]["rel_err"] < 1e-5, (name, out["id_param_true"])
        for v in VARIANTS[1:]:
            differs = {"naive_freq": fac, "naive_width": rot != d or fac, "naive_pairing": True,
                       "naive_scale": sc != 1.0,
                       "naive_all": rot != d or inter or sc != 1.0 or fac}[v]
            err = out[f"id_{v}"]["rel_err"]
            assert (err > 1e-3) == differs, (name, v, err, differs)
        for fmt in FMTS:
            assert out[fmt]["ab_gate_param_true"]["kl"] == out[fmt]["ab_gate"]["kl"] or abs(
                out[fmt]["ab_gate_param_true"]["kl"] - out[fmt]["ab_gate"]["kl"]) < 1e-6
            assert out[fmt]["centre_noadd"]["kl"] > out[fmt]["ab_gate"]["kl"], (name, fmt)
        assert out["gate_exempt_frac"] > 0, "the sink-like row must be exempted"
    # NoPE layer: identity rotation; a kernel that rotates mu anyway must show an error
    rnope = {"param_true": (lambda x: x),
             "naive_rotated": param_rot(RopeSpec(theta ** (-torch.arange(0, d, 2,
                                        dtype=torch.float64) / d), False, 1.0), d, pos)}
    q = torch.randn(H, nq, d)
    out = analyse_layer(kpre, mu, kpre.mean(1), mu.flip(0), q, lambda x: x, rnope, "gguf",
                        H // kvh, d ** -0.5, qpos)
    assert out["id_ab_gate"]["rel_err"] < 1e-5 and out["id_naive_rotated"]["rel_err"] > 1e-2
    had = hadamard(d, torch.device("cpu"))
    x, y = torch.randn(3, 5, d), torch.randn(3, 7, d)
    for order in ("hf", "gguf"):
        lhs = to_engine(x, order, had) @ to_engine(y, order, had).transpose(-1, -2)
        assert torch.allclose(lhs, x @ y.transpose(-1, -2), atol=1e-4)
    w = torch.randn(H, nq, L)
    c = compare(w, w + 3.0, qpos)
    assert c["kl"] < 1e-6 and c["rel_err"] < 1e-6, "softmax shift invariance of both metrics"
    assert compare(w, torch.zeros_like(w), qpos)["kl"] > 0
    print("selftest ok")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--order", choices=["hf", "gguf"])
    ap.add_argument("--calib")
    ap.add_argument("--text")
    ap.add_argument("--out")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--nq", type=int, default=256)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--random-layers", type=int, default=0)
    ap.add_argument("--focus", type=lambda s: [int(x) for x in s.split(",") if x], default=[])
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if not all((a.model, a.order, a.calib, a.text, a.out)):
        ap.error("--model --order --calib --text --out are required")
    if not 1 <= a.threads <= 32:
        ap.error("--threads must be in 1..32 on the shared host")
    main_model(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
