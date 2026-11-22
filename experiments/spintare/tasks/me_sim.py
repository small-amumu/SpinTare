"""ME1-ME7 (PREDICTIONS.md "BL / AB / ME design" 15:06:53Z and "ME amendment 1" 15:28:58Z): the
collapse mechanism along attention, simulated in Hugging Face with the same hook as b4_sim.py (wrap
apply_rotary_pos_emb).

Text: calibration = first 1024 tokens, evaluation = next 1024; output KL D over the last 256
positions. Format: llama.cpp q4_0 (blocks of 32, ggml rule), with the engine's K rotation when rot
is on: an orthonormal Walsh-Hadamard of size n_rot_k (64 doubled while it divides the head dim: 128
for head dim 128), applied blockwise over the head dimension (quantize in the rotated basis, rotate
back), as in llama_kv_cache (llama.cpp 3bb63cb).
Parts: --parts stats,layers,prop,inject  (inject = ME5 / ME6, only meaningful on Llama-3.1-8B / Mistral-7B).
q4_0 follows ggml exactly (d = signed max / -8). Evaluation tokens run at positions 0..1023 (as in
b4_sim.py). Models whose attention passes only a slice of the head to apply_rotary_pos_emb
(StableLM-2) are rejected. Usage: python me_sim.py <hf dir> <text> <out.json> [--parts ...] [--bf16]
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ggml primitives come from the library (spintare/ggml_ref.py); KVREL_ROOT (scripts/env.sh) or the
# repository root of this file locates it, also for copies staged under $KVREL_WORK
sys.path.insert(0, os.environ.get("KVREL_ROOT") or str(Path(__file__).resolve().parents[3]))
from spintare.ggml_ref import hadamard, n_rot_k, q4_0_div  # noqa: E402

C_LEN, T_LEN, NQ = 1024, 1024, 256
LOW, HIGH = math.pi / 4, 16 * math.pi


def arg(name: str, default: str) -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


H64: torch.Tensor | None = (
    None  # the engine's K rotation (name kept); size NROT, see n_rot_k() below
)
NROT = 64


def rot(x: torch.Tensor) -> torch.Tensor:
    s = x.shape
    return (x.reshape(*s[:-1], s[-1] // NROT, NROT) @ H64).reshape(s)


def unrot(x: torch.Tensor) -> torch.Tensor:
    s = x.shape
    return (x.reshape(*s[:-1], s[-1] // NROT, NROT) @ H64.T).reshape(s)


def quant(x: torch.Tensor, use_rot: bool) -> torch.Tensor:
    xf = x.float()
    y = unrot(q4_0_div(rot(xf))) if use_rot else q4_0_div(xf)
    return y.to(x.dtype)


def block_stats(x: torch.Tensor, c: torch.Tensor) -> tuple[float, float]:
    """median content-RMS / step and median crest factor over all q4_0 blocks; x, c: [..., D]
    FP32."""
    xb = x.reshape(-1, 32)
    cb = c.reshape(-1, 32)
    amax = xb.abs().amax(-1).clamp(min=1e-12)
    ratio = cb.pow(2).mean(-1).sqrt() / (amax / 8)
    crest = amax / xb.pow(2).mean(-1).sqrt().clamp(min=1e-12)
    return float(ratio.median()), float(crest.median())


@torch.no_grad()
def main() -> None:
    global H64, NROT
    model_dir, text, out = (
        sys.argv[1],
        Path(sys.argv[2]).read_text(encoding="utf-8"),
        Path(sys.argv[3]),
    )
    parts = arg("--parts", "stats,layers,prop").split(",")
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.bfloat16 if "--bf16" in sys.argv else torch.float32
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(dev).eval()
    cfg = model.config
    n_layer = cfg.num_hidden_layers
    base_mod = model.gpt_neox if hasattr(model, "gpt_neox") else model.model
    attn0 = (
        base_mod.layers[0].attention if hasattr(model, "gpt_neox") else base_mod.layers[0].self_attn
    )
    mod = sys.modules[type(attn0).__module__]
    orig = mod.apply_rotary_pos_emb
    inv_freq = (
        base_mod.rotary_emb.inv_freq.float().cpu()
    )  # actual rotation frequencies (incl. rope scaling)
    rscale = float(getattr(base_mod.rotary_emb, "attention_scaling", 1.0))
    assert abs(rscale - 1.0) < 1e-6, f"rope attention_scaling {rscale} != 1 would scale R(p) mu"
    HD = int(getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
    assert HD % 64 == 0, f"head dim {HD}: llama.cpp applies no K rotation"
    NROT = n_rot_k(HD)
    H64 = hadamard(NROT, dev)
    S: dict = {
        "i": 0,
        "mode": "base",
        "layers": None,
        "rot": True,
        "pm": False,
        "pos": None,
        "mu": {},
        "m": {},
        "cbar": {},
        "stats": {},
        "off": {},
    }

    def wrapped(q: torch.Tensor, k: torch.Tensor, *a, **kw):
        l = S["i"] % n_layer
        S["i"] += 1
        if (
            k.shape[-1] != HD
        ):  # the model rotates a slice of the head (e.g. StableLM-2): hook cannot see full K
            raise RuntimeError(
                f"apply_rotary_pos_emb sees {k.shape[-1]} of {HD} head dims; unsupported model"
            )
        if S["off"]:
            k = k + S["off"][l].to(k.dtype)
        qo, ko = orig(q, k, *a, **kw)
        mode = S["mode"]
        if mode == "calib":
            kf = k.float()
            mu = kf.mean(2, keepdim=True)
            S["mu"][l] = mu
            S["m"][l] = ko.float().mean(2, keepdim=True)
            S["cbar"][l] = (kf - mu).norm(dim=-1).mean(2)[0]  # [H]
            _, rmu = orig(q, mu.expand_as(k), *a, **kw)
            S["stat_share"] = S.get("stat_share", {})
            S["stat_share"][l] = (float(rmu.float().mean(2).pow(2).sum()), float(mu.pow(2).sum()))
            return qo, ko
        if mode == "stats":
            x = ko.float()
            _, rmu = orig(q, S["mu"][l].expand_as(k), *a, **kw)
            c = x - rmu.float()
            rec = {}
            for name, use_rot in (("on", True), ("off", False)):
                xx, cc = (rot(x), rot(c)) if use_rot else (x, c)
                rec[f"ratio_{name}"], rec[f"crest_{name}"] = block_stats(xx[0], cc[0])
                xp = x - S["m"][l]
                xp = rot(xp) if use_rot else xp
                rec[f"crest_pm_{name}"] = block_stats(xp[0], xp[0])[1]
            kf = k.float()
            r_h = S["mu"][l][0, :, 0].norm(dim=-1) / (kf - S["mu"][l]).norm(dim=-1).mean(2)[0]
            rec["r"] = float(r_h.max())
            # ME2: score SNR and attention KL, same hidden states, rot on
            e = quant(ko, True).float() - x
            qq = qo.float()[0, :, -NQ:]  # [Hq, NQ, D]
            g = qq.shape[0] // x.shape[1]
            xk = x[0].repeat_interleave(g, 0)  # [Hq, T, D]
            ek = e[0].repeat_interleave(g, 0)
            sc = qq @ xk.transpose(1, 2) / math.sqrt(x.shape[-1])  # [Hq, NQ, T]
            ns = qq @ ek.transpose(1, 2) / math.sqrt(x.shape[-1])
            T = x.shape[2]
            mask = (
                torch.arange(T, device=dev)[None, :] <= torch.arange(T - NQ, T, device=dev)[:, None]
            )
            mask = mask[None].expand_as(sc)
            big = torch.tensor(float("-inf"), device=dev)
            s1 = torch.where(mask, sc, big)
            s2 = torch.where(mask, sc + ns, big)
            p1, lp1, lp2 = s1.softmax(-1), s1.log_softmax(-1), s2.log_softmax(-1)
            rec["attn_kl"] = float(
                torch.where(mask, p1 * (lp1 - lp2), torch.zeros_like(p1)).sum(-1).mean()
            )
            p2 = s2.softmax(-1)
            ent = lambda pp, lp: torch.where(mask, -pp * lp, torch.zeros_like(pp)).sum(-1).mean()
            rec["attn_entropy"], rec["attn_entropy_q"] = float(ent(p1, lp1)), float(ent(p2, lp2))
            cnt = mask.sum(-1).clamp(min=2)

            def std(v: torch.Tensor) -> torch.Tensor:
                v0 = torch.where(mask, v, torch.zeros_like(v))
                mean = v0.sum(-1) / cnt
                return (
                    torch.where(mask, (v - mean[..., None]) ** 2, torch.zeros_like(v)).sum(-1)
                    / (cnt - 1)
                ).sqrt()

            rec["score_snr"] = float((std(sc) / std(ns).clamp(min=1e-12)).median())
            S["stats"][l] = rec
            return qo, ko
        if mode == "quant":
            if S["layers"] is not None and l not in S["layers"]:
                return qo, ko
            x = ko - S["m"][l].to(ko.dtype) if S["pm"] else ko
            y = quant(x, S["rot"])
            if S["pos"] is not None:
                keep = torch.zeros(y.shape[2], dtype=torch.bool, device=dev)
                keep[S["pos"]] = True
                y = torch.where(keep[None, None, :, None], y, x)
            return qo, y
        return qo, ko

    mod.apply_rotary_pos_emb = wrapped
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    calib, ev = ids[:, :C_LEN], ids[:, C_LEN : C_LEN + T_LEN]
    assert calib.shape[1] == C_LEN and ev.shape[1] == T_LEN

    def fwd(mode: str, x: torch.Tensor, hidden: bool = False, **kw):
        S.update({"i": 0, "mode": mode, "layers": None, "rot": True, "pm": False, "pos": None})
        S.update(kw)
        o = model(x, output_hidden_states=hidden)
        assert S["i"] == n_layer, f"rotary hook ran {S['i']} times for {n_layer} layers"
        lp = torch.log_softmax(o.logits[0, -NQ:].float(), -1)
        return (lp, o.hidden_states) if hidden else lp

    def kl(base: torch.Tensor, lp: torch.Tensor) -> float:
        return round(float((base.exp() * (base - lp)).sum(-1).mean()), 6)

    t0 = time.time()
    hd_cfg = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    res: dict = {
        "model": model_dir,
        "n_layer": n_layer,
        "head_dim": int(hd_cfg),
        "dtype": str(dtype),
        "q4_0": "ggml (signed max / -8)",
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "n_rot_k": NROT,
    }
    fwd("calib", calib)
    ss = S["stat_share"]
    res["static_share"] = round(
        sum(v[0] for v in ss.values()) / max(1e-12, sum(v[1] for v in ss.values())), 4
    )
    # RoPE bands of mu: the rotary part is the first r = 2 len(inv_freq) dims, pairs j <-> j + r/2
    # (rotate_half);
    # dims beyond r (partial rotary, e.g. Pythia) never rotate and are reported as "pass"; T = 2048
    phi = inv_freq * 2048
    band = {"low": 0.0, "mid": 0.0, "high": 0.0, "pass": 0.0}
    rdim = 2 * len(phi)
    for l, mu in S["mu"].items():
        v = mu[0, :, 0].cpu()  # [H, D]
        h = rdim // 2
        pe = v[:, :h] ** 2 + v[:, h:rdim] ** 2  # [H, r/2]
        for j in range(h):
            b = "low" if phi[j] < LOW else ("high" if phi[j] > HIGH else "mid")
            band[b] += float(pe[:, j].sum())
        band["pass"] += float((v[:, rdim:] ** 2).sum())
    tot = sum(band.values())
    res["band_share"] = {k: round(v / tot, 4) for k, v in band.items()}
    base = fwd("base", ev)
    res["D_all_on"] = kl(base, fwd("quant", ev, rot=True))
    res["D_all_off"] = kl(base, fwd("quant", ev, rot=False))
    res["D_all_on_pm"] = kl(base, fwd("quant", ev, rot=True, pm=True))
    if "stats" in parts:
        fwd("stats", ev)
        res["layers_stats"] = {
            str(l): {k: round(v, 5) for k, v in S["stats"][l].items()} for l in range(n_layer)
        }
    if "layers" in parts:
        res["D_single_layer_on"] = [
            kl(base, fwd("quant", ev, layers={l}, rot=True)) for l in range(n_layer)
        ]
    if "prop" in parts:
        _, hb = fwd("base", ev, hidden=True)
        _, hq = fwd("quant", ev, hidden=True, layers={0}, rot=True)
        res["dev_layer0_only"] = [
            round(float((a[0, -NQ:] - b[0, -NQ:]).float().norm() / b[0, -NQ:].float().norm()), 5)
            for a, b in zip(hq[1:], hb[1:])
        ]
        res["D_prefix_keys_only"] = kl(
            base, fwd("quant", ev, pos=torch.arange(0, T_LEN - NQ, device=dev))
        )
        res["D_last_keys_only"] = kl(
            base, fwd("quant", ev, pos=torch.arange(T_LEN - NQ, T_LEN, device=dev))
        )
    if "inject" in parts:
        d = next(iter(S["mu"].values())).shape[-1]
        h = d // 2
        for where in ("slow", "fast", "all"):
            g = torch.Generator().manual_seed(0)  # same seed for every arm (ME amendment 1)
            pairs = (
                list(range(h - 12, h))
                if where == "slow"
                else (list(range(12)) if where == "fast" else list(range(h)))
            )
            dims = pairs + [j + h for j in pairs]
            off = {}
            for l in range(n_layer):
                Hk = S["mu"][l].shape[1]
                v = torch.zeros(Hk, d)
                v[:, dims] = torch.randn(Hk, len(dims), generator=g)
                v = v / v.norm(dim=-1, keepdim=True) * (8 * S["cbar"][l].cpu()[:, None])
                off[l] = v[None, :, None, :].to(dev)
            saved = {k: dict(S[k]) for k in ("mu", "m", "cbar")}
            S["off"] = off
            fwd("calib", calib)  # m of the injected model
            b_inj = fwd("base", ev)  # reference: the injected model unquantized
            res[f"inj_{where}_D_plain_on"] = kl(b_inj, fwd("quant", ev, rot=True))
            res[f"inj_{where}_D_pm_on"] = kl(b_inj, fwd("quant", ev, rot=True, pm=True))
            res[f"inj_{where}_D_plain_off"] = kl(b_inj, fwd("quant", ev, rot=False))
            S["off"] = {}
            for k, v in saved.items():
                S[k] = v
        # ME6: offset into V (all dims, norm 8 x mean V content norm), V quantized with q4_0, K
        # unquantized
        vmods = [lay.self_attn.v_proj for lay in base_mod.layers]
        hd = d
        vst: dict = {}

        def vcal(i: int):
            def f(m, inp, outp):
                v = outp.float().reshape(*outp.shape[:2], -1, hd)  # [B, T, Hk, D]
                mu = v.mean(1, keepdim=True)
                vst[i] = (v - mu).norm(dim=-1).mean(1)[0]  # [Hk]

            return f

        hs = [m.register_forward_hook(vcal(i)) for i, m in enumerate(vmods)]
        try:
            fwd("base", calib)
        finally:
            for x in hs:
                x.remove()
        g = torch.Generator().manual_seed(1)
        voff = {}
        for i in range(n_layer):
            Hk = vst[i].shape[0]
            v = torch.randn(Hk, hd, generator=g)
            voff[i] = (
                (v / v.norm(dim=-1, keepdim=True) * (8 * vst[i].cpu()[:, None])).reshape(-1).to(dev)
            )
        state = {"q": False}

        def vinj(i: int):
            def f(m, inp, outp):
                y = outp + voff[i].to(outp.dtype)
                if state["q"]:
                    s = y.shape
                    y = q4_0_div(y.float().reshape(*s[:2], -1, hd)).reshape(s).to(y.dtype)
                return y

            return f

        hs = [m.register_forward_hook(vinj(i)) for i, m in enumerate(vmods)]
        try:
            b_v = fwd("base", ev)
            state["q"] = True
            res["injV_D_quantV"] = kl(
                b_v, fwd("base", ev)
            )  # V q4_0 without rotation, K unquantized
        finally:
            for x in hs:
                x.remove()
    res["sec"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {
                k: v
                for k, v in res.items()
                if k not in ("layers_stats", "D_single_layer_on", "dev_layer0_only")
            }
        )
    )


if __name__ == "__main__":
    main()
