# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""LC1 (REGISTRATION_LC1.md): does static mean-centring of post-RoPE keys stop working as the
context grows?

One teacher-forced pass over L tokens per arm, every layer's keys replaced by the arm's
reconstruction, values exact (the protocol of c27_kl.py, at L up to 32768). The next-token KL to the
exact model is read in buckets of 512 positions. Key formats are llama.cpp's 4-bit K-cache formats
as emulated in eg1_engine.reconstruct (validated against the engine in EG1): `nr` = Q4_0 on
32-element blocks, `ro` = the same after the block-diagonal Hadamard rotation. Treatments of the
shared key offset, each in both formats:
  A    none
  B2   static centring: subtract the post-RoPE mean key of the calibration text at positions [0, CAL2)
  BL   static centring with the mean over positions [0, L) of the calibration text (calibration as long as the test)
  D    rotary centring: subtract R(t) mu, mu = the pre-RoPE mean key of the calibration text at positions [0, CAL2)
  E    D, except that rows which centring lengthens (|k - R(t) mu| >= |k|) are coded uncentred in the same format
Calibration comes from the other text (qasper <-> mfqa). X = the arm path with exact keys (must give
KL 0).

Mechanism readings from the exact pass of the test text, per layer, KV head and bucket:
  rho_meas  |mean_t R(-t)(k_t - m)|^2 / |mean_t R(-t) k_t|^2   offset energy left after subtracting the static mean m
  rho_pred  |mu_cal - mean_t R(-t) m|^2 / |mu_cal|^2           the same from the calibration text and the positions only
  rho_cf    sum_p |mu_p|^2 |1 - a_p conj(b_p)|^2 / sum_p |mu_p|^2   closed form over the rotary pairs p of frequency
            theta_p: a_p = mean of exp(i t theta_p) over the calibration positions, b_p the same
            over the bucket
for m = B2's mean and BL's mean, and for rotary centring |mean_t R(-t) k_t - mu_cal|^2 / |mean_t
R(-t) k_t|^2.

Usage: python lc1_long.py real <out_json> <model_dir> <short> --order hf|gguf --data <dir> [--L
32768] [--smoke]
       python lc1_long.py agg <dir with lc1_<model>.json> --logs <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics as st
import sys
import time
from pathlib import Path

import torch

# ggml primitives come from the library (spintare/ggml_ref.py); KVREL_ROOT (experiments/scripts/env.sh) or the
# repository root of this file locates it, also for copies staged under $KVREL_WORK
sys.path.insert(0, str(Path(os.environ.get("KVREL_ROOT") or Path(__file__).resolve().parents[3]) / "src"))
from spintare.ggml_ref import hadamard, n_rot_k, quant_f32scale  # noqa: E402

TEXTS = {"qasper": "qasper_text.txt", "mfqa": "mfqa_text.txt"}
CAL = {"qasper": "mfqa", "mfqa": "qasper"}
CAL2, BW, CHUNK = 2048, 512, 256
DEROT_MAX = 0.05  # the run stops if the rotary tables miss the model's keys by more
EXPECTED = (
    "llama31_8b",
    "mistral7b",
    "olmo7b",
    "q25_1p5b",
    "q3_8b",
    "q25_7b",
    "granite8b",
    "llama3b",
    "yi6b",
    "ds7b",
    "r1q_7b",
    "q3_0p6b",
)
TREAT = ("A", "B2", "BL", "D", "E")
FMT = {"nr": False, "ro": True}
ARMS = ("X",) + tuple(f"{t}_{f}" for f in FMT for t in TREAT)
S: dict = {
    "mode": None,
    "arm": None,
    "inv": None,
    "tab": None,
    "order": "hf",
    "cal": {},
    "mu": {},
    "rho": {},
    "gate": {},
    "had": {},
    "perm": {},
    "buckets": (),
    "L": 0,
}


def bucket_starts(L: int) -> tuple[int, ...]:
    """512-position buckets ending at 1536, 4096, 8192, ... , L (those that fit)."""
    ends = [1536] + [e for e in (4096, 8192, 16384, 32768, 65536, 131072) if e <= L]
    return tuple(e - BW for e in ends)


def rot_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, -1)
    return torch.cat((-b, a), -1)


def tables(n: int, dev) -> tuple[torch.Tensor, torch.Tensor]:
    if S["tab"] is None or S["tab"][0].shape[0] != n or S["tab"][0].device != torch.device(dev):
        f = (
            torch.arange(n, dtype=torch.float64, device=dev)[:, None]
            * S["inv"].double().to(dev)[None]
        )
        e = torch.cat([f, f], -1)
        S["tab"] = (e.cos(), e.sin())
    return S["tab"]


def reconstruct(k: torch.Tensor, rot: bool) -> torch.Tensor:
    """eg1_engine.reconstruct for q4_0: k (KVH, n, d) fp32 in HF channel order -> the keys attention
    reads."""
    d, dev = k.shape[-1], k.device
    assert d % 32 == 0, d
    x = k
    if S["order"] == "gguf":
        if (d, dev) not in S["perm"]:
            half = d // 2
            p = torch.tensor([(e % 2) * half + e // 2 for e in range(d)], device=dev)
            S["perm"][(d, dev)] = (p, torch.argsort(p))
        x = x[..., S["perm"][(d, dev)][0]]
    n = n_rot_k(d) if rot else 0
    if rot and not n:
        raise SystemExit(f"head dim {d}: the engine applies no rotation; the ro arms are undefined")
    if n:
        if (n, dev) not in S["had"]:
            S["had"][(n, dev)] = hadamard(n, dev)
        h = S["had"][(n, dev)]
        x = (x.reshape(*x.shape[:-1], d // n, n) @ h).reshape(x.shape)
    y = quant_f32scale(x, "q4_0")
    if n:
        y = (y.reshape(*y.shape[:-1], d // n, n) @ h).reshape(y.shape)
    if S["order"] == "gguf":
        y = y[..., S["perm"][(d, dev)][1]]
    return y


def arm_keys(y32: torch.Tensor, li: int, arm: str) -> torch.Tensor:
    """y32 (KVH, n, d): post-RoPE keys of one layer -> the arm's reconstruction (fp32)."""
    if arm == "X":
        return y32
    treat, fmt = arm.split("_")
    rot = FMT[fmt]
    kvh, n, d = y32.shape
    mu, yb2, ybl = S["mu"][li]
    if treat == "A":
        return reconstruct(y32, rot)
    if treat in ("B2", "BL"):
        yb = (yb2 if treat == "B2" else ybl)[:, None, :]
        return (reconstruct((y32.double() - yb).float(), rot).double() + yb).float()
    cos, sin = tables(n, y32.device)
    assert cos.shape[-1] == d, "partial rotary dimensions are not handled"
    mr = mu[:, None, :].expand(kvh, n, d)
    rmu = mr * cos + rot_half(mr) * sin
    y = y32.double()
    out = reconstruct((y - rmu).float(), rot).double() + rmu
    if treat == "D":
        return out.float()
    yn = y.norm(dim=-1).clamp(min=1e-30)
    rows = ((y - rmu).norm(dim=-1) / yn >= 1.0) & (yn >= 1e-12 * yn.median())
    if fmt == "nr":
        S["gate"][li] = float(rows.float().mean())
    return torch.where(rows[..., None], reconstruct(y32, rot).double(), out).float()


def derot_mean(
    k64: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, lo: int, hi: int
) -> torch.Tensor:
    """mean over positions [lo, hi) of R(-t) k_t; k64 (KVH, n, d) float64 or a (KVH, d) constant."""
    c, s_ = cos[lo:hi], sin[lo:hi]
    if k64.dim() == 2:
        k64 = k64[:, None, :]
        return (k64 * c - rot_half(k64) * s_).mean(1)
    kk = k64[:, lo:hi]
    return (kk * c - rot_half(kk) * s_).mean(1)


def record_cal(li: int, k: torch.Tensor) -> None:
    n = k.shape[1]
    cos, sin = tables(n, k.device)
    kd = k.double()
    if li == 0 and S.get("kpre0") is not None:  # the tables against the model's own
        kvh, _, d = k.shape  # pre-RoPE keys (the check of c27_kl.py)
        refk = S["kpre0"][0].double().reshape(n, kvh, d).transpose(0, 1)
        kpre = kd * cos - rot_half(kd) * sin
        S["derot"] = max(S.get("derot", 0.0), float((refk - kpre).norm() / refk.norm()))
        for lo in S["buckets"]:  # per bucket: does the error grow with t
            e = float(
                (refk[:, lo : lo + BW] - kpre[:, lo : lo + BW]).norm()
                / refk[:, lo : lo + BW].norm()
            )
            S.setdefault("derot_b", {})[lo] = max(S.get("derot_b", {}).get(lo, 0.0), e)
    S["cal"][li] = (derot_mean(kd, cos, sin, 0, CAL2), kd[:, :CAL2].mean(1), kd.mean(1))


def record_rho(li: int, k: torch.Tensor) -> None:
    """mechanism readings on the exact keys of the test text; S['mu'] holds the calibration text's
    quantities."""
    n, d = k.shape[1], k.shape[-1]
    cos, sin = tables(n, k.device)
    kd = k.double()
    mu, yb2, ybl = S["mu"][li]
    half = d // 2
    th = S["inv"].double().to(k.device)
    mp = mu[:, :half] ** 2 + mu[:, half:] ** 2  # |mu_p|^2 per rotary pair, (KVH, d/2)
    tc = torch.arange(CAL2, dtype=torch.float64, device=k.device)
    tl = torch.arange(n, dtype=torch.float64, device=k.device)
    a2 = torch.polar(
        torch.ones(CAL2, half, dtype=torch.float64, device=k.device), tc[:, None] * th[None]
    ).mean(0)
    al = torch.polar(
        torch.ones(n, half, dtype=torch.float64, device=k.device), tl[:, None] * th[None]
    ).mean(0)
    rec = {
        "meas_B2": [],
        "pred_B2": [],
        "cf_B2": [],
        "meas_BL": [],
        "pred_BL": [],
        "cf_BL": [],
        "meas_D": [],
    }
    den_c = (mu**2).sum(-1).clamp(min=1e-300)
    for b0 in S["buckets"]:
        lo, hi = b0, b0 + BW
        mk = derot_mean(kd, cos, sin, lo, hi)  # the bucket's own pre-RoPE mean key
        den = (mk**2).sum(-1).clamp(min=1e-300)
        tb = torch.arange(lo, hi, dtype=torch.float64, device=k.device)
        bb = torch.polar(
            torch.ones(BW, half, dtype=torch.float64, device=k.device), tb[:, None] * th[None]
        ).mean(0)
        for tag, yb, aa in (("B2", yb2, a2), ("BL", ybl, al)):
            ym = derot_mean(yb, cos, sin, lo, hi)  # mean of R(-t) m over the bucket
            rec["meas_" + tag].append(((mk - ym) ** 2).sum(-1) / den)
            rec["pred_" + tag].append(((mu - ym) ** 2).sum(-1) / den_c)
            rec["cf_" + tag].append(
                (mp * (1 - aa * bb.conj()).abs() ** 2).sum(-1) / mp.sum(-1).clamp(min=1e-300)
            )
        rec["meas_D"].append(((mk - mu) ** 2).sum(-1) / den)
    S["rho"][li] = {k_: torch.stack(v).cpu() for k_, v in rec.items()}  # (buckets, KVH)


def lc_attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    li = int(module.layer_idx)
    q, k, v = query[0].float(), key[0].float(), value[0].float()  # (H, n, d), (KVH, n, d)
    if kw.get("softcap") or kw.get("sliding_window"):
        raise SystemExit("soft cap / sliding window are not handled")
    if S["mode"] == "cal":
        record_cal(li, k)
    elif S["mode"] == "ref":
        record_rho(li, k)
    elif S["arm"] is not None:
        k = arm_keys(k, li, S["arm"])
    H, n, d = q.shape
    g = H // k.shape[0]
    scale = scaling if scaling is not None else 1.0 / math.sqrt(d)
    kk, vv = (k.repeat_interleave(g, 0), v.repeat_interleave(g, 0)) if g > 1 else (k, v)
    out = torch.empty(H, n, v.shape[-1], device=q.device)
    pos = torch.arange(n, device=q.device)
    for r0 in range(0, n, CHUNK):
        s = torch.einsum("hld,hmd->hlm", q[:, r0 : r0 + CHUNK], kk) * scale
        s = s.masked_fill(pos[None, :] > pos[r0 : r0 + CHUNK, None], float("-inf"))
        out[:, r0 : r0 + CHUNK] = torch.einsum("hlm,hmd->hld", torch.softmax(s, -1), vv)
    return out.unsqueeze(0).transpose(1, 2).contiguous().to(query.dtype), None


def text_ids(tok, path: str, n: int, dev) -> torch.Tensor:
    """cq_lab3.text_ids at start 0: the first n tokens of the text, a BOS in front if the tokenizer
    adds one."""
    ids = tok(
        Path(path).read_text(encoding="utf-8", errors="replace"),
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]
    b = tok.bos_token_id
    bos = b is not None and tok("x", add_special_tokens=True).input_ids[:1] == [b]
    w = ids[: n - (1 if bos else 0)]
    if bos:
        w = torch.cat([torch.tensor([b]), w])
    if w.shape[0] != n:
        raise SystemExit(f"{path}: only {w.shape[0]} of {n} tokens")
    return w.unsqueeze(0).to(dev)


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    rx, ry = x.argsort().argsort().double(), y.argsort().argsort().double()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp(min=1e-300))


def real(a) -> int:
    from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

    AttentionInterface.register("lc_attn", lc_attn)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if a.smoke != str(a.out).endswith(".smoke.json"):
        raise SystemExit("smoke output must end with .smoke.json (and only smoke)")
    dev, L = a.device, a.L
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    dt = torch.float32 if a.dtype == "float32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        a.model_dir, dtype=dt, attn_implementation="lc_attn"
    )
    model = model.to(dev).eval()
    re_ = model.model.rotary_emb
    if abs(float(getattr(re_, "attention_scaling", 1.0)) - 1.0) > 1e-6:
        raise SystemExit("attention_scaling != 1")
    S["inv"], S["order"], S["L"] = re_.inv_freq.detach().float(), a.order, L
    S["buckets"] = bucket_starts(L)
    nl = model.config.num_hidden_layers
    at0 = model.model.layers[0].self_attn
    hk = getattr(at0, "k_norm", None) or getattr(at0, "k_proj", None)
    assert hk is not None
    hk.register_forward_hook(lambda m_, i_, o_: S.__setitem__("kpre0", o_.detach()))
    rows_idx = torch.cat([torch.arange(b0, b0 + BW) for b0 in S["buckets"]]).to(dev)

    def bucket_logits(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = model.model(input_ids=x, use_cache=False).last_hidden_state[0, rows_idx]
            # Granite divides its logits by config.logits_scaling in ForCausalLM.forward (1 for the
            # others)
            return model.lm_head(h).float() / float(getattr(model.config, "logits_scaling", 1.0))

    def kl_top1(z0: torch.Tensor, za: torch.Tensor) -> tuple[list[float], list[float], list[float]]:
        kls, tops, meds = [], [], []
        for i in range(len(S["buckets"])):
            lr = torch.log_softmax(z0[i * BW : (i + 1) * BW].double(), -1)
            la = torch.log_softmax(za[i * BW : (i + 1) * BW].double(), -1)
            kl = (lr.exp() * (lr - la)).sum(-1)
            if not torch.isfinite(kl).all():
                raise SystemExit("non-finite KL")
            kls.append(float(kl.mean()))
            meds.append(float(kl.median()))
            tops.append(float((la.argmax(-1) == lr.argmax(-1)).double().mean()))
        return kls, tops, meds

    ids, ref, mus, rhos = {}, {}, {}, {}
    t0 = time.time()
    for t in TEXTS:  # exact pass: offsets and reference logits
        if a.smoke:
            gen = torch.Generator().manual_seed(7 + len(ids))
            ids[t] = torch.randint(100, tok.vocab_size - 100, (1, L), generator=gen).to(dev)
        else:
            ids[t] = text_ids(tok, str(Path(a.data) / TEXTS[t]), L, dev)
        S["mode"], S["cal"], S["arm"] = "cal", {}, None
        ref[t] = bucket_logits(ids[t]).cpu()
        S["mode"] = None
        assert len(S["cal"]) == nl, len(S["cal"])
        if not S.get("derot", 1.0) < DEROT_MAX:
            raise SystemExit(f"de-rotation check failed: {S.get('derot')}")
        mus[t] = S["cal"]
    for t in TEXTS:  # mechanism readings with the other text's offset
        S["mode"], S["mu"], S["rho"] = "ref", mus[CAL[t]], {}
        z = bucket_logits(ids[t])
        S["mode"] = None
        if float((z.cpu() - ref[t]).abs().max()) != 0.0:
            raise SystemExit("the exact pass is not reproducible")
        rhos[t] = S["rho"]
    rows = []
    for t in TEXTS:
        S["mu"], z0 = mus[CAL[t]], ref[t].to(dev)
        rec: dict = {"text": t, "cal": CAL[t], "arms": {}}
        for arm in ARMS:
            S["arm"], S["gate"] = arm, S["gate"] if arm != "E_nr" else {}
            za = bucket_logits(ids[t])
            S["arm"] = None
            kl, top1, klm = kl_top1(z0, za)
            del za
            rec["arms"][arm] = {"kl": kl, "top1": top1, "kl_med": klm}
        if rec["arms"]["X"]["kl"] != [0.0] * len(S["buckets"]):
            raise SystemExit(f"arm X is not exact: {rec['arms']['X']['kl']}")
        rho = rhos[t]
        allr = {
            k_: torch.stack([rho[li][k_] for li in range(nl)]) for k_ in rho[0]
        }  # (layers, buckets, KVH)
        rec["rho"] = {
            k_: [float(v[:, b].median()) for b in range(len(S["buckets"]))]
            for k_, v in allr.items()
        }
        rec["rho_layer_median"] = {
            k_: [[float(v[li, b].median()) for b in range(len(S["buckets"]))] for li in range(nl)]
            for k_, v in allr.items()
            if k_.startswith("meas")
        }
        for tag in ("B2", "BL"):
            m_, p_, c_ = (
                allr[f"{x}_{tag}"].flatten().clamp(min=1e-30).log10()
                for x in ("meas", "pred", "cf")
            )
            nb_ = len(S["buckets"])
            rec[f"fit_{tag}_within_bucket_spearman_pred"] = [
                spearman(allr[f"pred_{tag}"][:, b].flatten(), allr[f"meas_{tag}"][:, b].flatten())
                for b in range(nb_)
            ]
            rec[f"fit_{tag}"] = {
                "spearman_pred": spearman(p_, m_),
                "spearman_cf": spearman(c_, m_),
                "median_abs_log10_err_pred": float((p_ - m_).abs().median()),
                "median_abs_log10_err_cf": float((c_ - m_).abs().median()),
                "n": int(m_.numel()),
            }
        rec["gate_frac_rows"] = st.mean(S["gate"].values()) if S["gate"] else None
        rows.append(rec)
        print(
            a.short,
            t,
            f"({time.time() - t0:.0f}s)",
            {k_: [f"{x:.2e}" for x in v_["kl"]] for k_, v_ in rec["arms"].items() if k_ != "X"},
            flush=True,
        )
    cfg = model.config
    meta = {
        "script_md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
        "model": a.model_dir,
        "short": a.short,
        "order": a.order,
        "L": L,
        "buckets": list(S["buckets"]),
        "bucket_width": BW,
        "cal_len": CAL2,
        "n_layers": nl,
        "kv_heads": getattr(cfg, "num_key_value_heads", None),
        "head_dim": int(S["inv"].shape[0]) * 2,
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
        "dtype": f"{a.dtype} weights, float32 attention",
        "device": str(dev),
        "smoke": a.smoke,
        "derotation_check_rel_err": S.get("derot"),
        "derotation_rel_err_by_bucket": S.get("derot_b"),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "gpu_name": torch.cuda.get_device_name() if str(dev).startswith("cuda") else None,
        "seconds": time.time() - t0,
        "arms": list(ARMS),
    }
    tmp = Path(str(a.out) + ".tmp")
    tmp.write_text(json.dumps({"meta": meta, "rows": rows, "complete": True}, allow_nan=False))
    tmp.replace(a.out)
    return 0


def agg(a) -> int:
    models, excluded = {}, {}
    for f in sorted(Path(a.dir).glob("lc1_*.json")):
        if f.name.endswith(".smoke.json"):
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        short = d["meta"]["short"]
        rc = Path(a.logs) / f"lc1_{short}.rc"
        if d["meta"]["dtype"].startswith("float32") or short not in EXPECTED:
            excluded[f.stem] = "control run (float32 weights) or unplanned model: not aggregated"
            continue
        if not d.get("complete") or len(d["rows"]) != 2:
            excluded[short] = "incomplete"
            continue
        if not rc.exists() or rc.read_text().strip() != "0":
            excluded[short] = "exit code not 0"
            continue
        m = d["meta"]
        nb = len(m["buckets"])
        kl = {
            arm: [st.mean(r["arms"][arm]["kl"][b] for r in d["rows"]) for b in range(nb)]
            for arm in m["arms"]
        }
        rho = {
            k_: [st.mean(r["rho"][k_][b] for r in d["rows"]) for b in range(nb)]
            for k_ in d["rows"][0]["rho"]
        }
        fit = {
            tag: {
                k_: st.mean(r[f"fit_{tag}"][k_] for r in d["rows"])
                for k_ in d["rows"][0][f"fit_{tag}"]
            }
            for tag in ("B2", "BL")
        }
        in_range = (m["max_position_embeddings"] or 0) >= m["L"]
        per_text = {}
        for r in d["rows"]:  # the two texts separately (descriptive)
            a_ = r["arms"]
            per_text[r["text"]] = {
                f_: {
                    "g_first": a_[f"B2_{f_}"]["kl"][0] / a_[f"D_{f_}"]["kl"][0],
                    "g_last": a_[f"B2_{f_}"]["kl"][-1] / a_[f"D_{f_}"]["kl"][-1],
                    "BL_over_D_last": a_[f"BL_{f_}"]["kl"][-1] / a_[f"D_{f_}"]["kl"][-1],
                    "E_over_D_last": a_[f"E_{f_}"]["kl"][-1] / a_[f"D_{f_}"]["kl"][-1],
                }
                for f_ in FMT
            }
            per_text[r["text"]]["rho_meas_B2"] = r["rho"]["meas_B2"]
            per_text[r["text"]]["within_bucket_spearman_B2"] = r[
                "fit_B2_within_bucket_spearman_pred"
            ]
        out = {
            "L": m["L"],
            "buckets": m["buckets"],
            "in_range": in_range,
            "max_pos": m["max_position_embeddings"],
            "per_text": per_text,
            "derot": m.get("derotation_check_rel_err"),
            "derot_by_bucket": m.get("derotation_rel_err_by_bucket"),
            "order": m["order"],
            "kl": kl,
            "rho": rho,
            "fit": fit,
            "gate_frac_rows": st.mean(r["gate_frac_rows"] for r in d["rows"]),
        }
        for f_ in FMT:
            g0, g1 = (kl[f"B2_{f_}"][b] / kl[f"D_{f_}"][b] for b in (0, nb - 1))
            out[f"needs_centring_{f_}"] = kl[f"A_{f_}"][0] >= 2 * kl[f"D_{f_}"][0]
            out[f"L2_growth_{f_}"] = g1 / g0
            out[f"L2_g_first_{f_}"], out[f"L2_g_last_{f_}"] = g0, g1
            out[f"L3_BL_over_D_last_{f_}"] = kl[f"BL_{f_}"][nb - 1] / kl[f"D_{f_}"][nb - 1]
            out[f"L4_E_over_D_{f_}"] = [kl[f"E_{f_}"][b] / kl[f"D_{f_}"][b] for b in range(nb)]
            out[f"D_over_A_{f_}"] = [kl[f"D_{f_}"][b] / kl[f"A_{f_}"][b] for b in range(nb)]
            out[f"B2_over_A_{f_}"] = [kl[f"B2_{f_}"][b] / kl[f"A_{f_}"][b] for b in range(nb)]
        models[short] = out
    need = lambda n: max(1, math.ceil(0.75 * n))  # noqa: E731
    res = {
        "models": models,
        "excluded": excluded,
        "n": len(models),
        "missing": [x for x in EXPECTED if x not in models and x not in excluded],
        "verdicts": {},
    }
    for scope in ("all", "in_range"):
        ms = {k_: v for k_, v in models.items() if scope == "all" or v["in_range"]}
        n = len(ms)
        v: dict = {"n": n}
        if n >= 3:
            rec = sum(
                x["rho"]["meas_B2"][0] <= 0.5
                and x["rho"]["meas_B2"][-1] >= 1.5 * x["rho"]["meas_B2"][0]
                for x in ms.values()
            )
            nev = sum(x["rho"]["meas_B2"][0] > 0.5 for x in ms.values())
            v["L1"] = {
                "recurs": rec,
                "never_removed": nev,
                "verdict": "RECURS"
                if rec >= need(n)
                else "NEVER_REMOVED"
                if nev >= need(n)
                else "MIXED",
            }
            sp = sum(x["fit"]["B2"]["spearman_pred"] >= 0.8 for x in ms.values())
            v["L1_fit"] = {"n_spearman_ge_0.8": sp, "holds": sp >= need(n)}
            for f_ in FMT:
                sub = [x for x in ms.values() if x[f"needs_centring_{f_}"]]
                k_ = len(sub)
                if k_ < 3:
                    v[f"L2_{f_}"] = {"models": k_, "verdict": "UNDECIDABLE"}
                    v[f"L3_{f_}"] = {"models": k_, "verdict": "UNDECIDABLE"}
                    continue
                up = sum(x[f"L2_growth_{f_}"] >= 1.5 for x in sub)
                v[f"L2_{f_}"] = {
                    "models": k_,
                    "n_growth_ge_1.5": up,
                    "verdict": "RECURS"
                    if up >= need(k_)
                    else "NO"
                    if k_ - up >= need(k_)
                    else "MIXED",
                }
                bl = sum(x[f"L3_BL_over_D_last_{f_}"] >= 1.5 for x in sub)
                v[f"L3_{f_}"] = {"models": k_, "n_ge_1.5": bl, "holds": bl >= need(k_)}
        res["verdicts"][scope] = v
    json.dump(res, sys.stdout, indent=1)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("real")
    r.add_argument("out", type=Path)
    r.add_argument("model_dir")
    r.add_argument("short")
    r.add_argument("--order", required=True, choices=["hf", "gguf"])
    r.add_argument("--data", required=True)
    r.add_argument("--L", type=int, default=32768)
    r.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    r.add_argument("--device", default="cuda")
    r.add_argument("--smoke", action="store_true")
    g = sub.add_parser("agg")
    g.add_argument("dir")
    g.add_argument("--logs", required=True)
    a = ap.parse_args()
    return real(a) if a.cmd == "real" else agg(a)


if __name__ == "__main__":
    raise SystemExit(main())
