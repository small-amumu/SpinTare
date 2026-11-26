# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Exemption gate in simulation (line C23): whole-model effect of sink handling under centred Q4_0
keys. GPU, one model per run.

Every layer's key cache is replaced by an arm's reconstruction in one teacher-forced forward pass
(values exact); the next-token distribution is compared with the exact pass on the last 2048
positions of a 4096-token window.
Usage:
  python -I c23_kl.py real <out_json> <model_dir> <short> --lib <libggml-base.so> [--texts ...]
  [--device cuda] python -I c23_kl.py agg <dir with c23_<model>.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as st
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c22_q import LIB_DEFAULT, Ggml, q4x, q8x, same  # noqa: E402

OLD = {
    "qasper": ("qasper_text.txt", 4096),
    "musique": ("musique_text.txt", 4096),
    "long": ("long_text.txt", 4096),
}
NEW = {
    "mfqa": ("mfqa_text.txt", 4096),
    "diag": ("diag_long_text.txt", 4096),
    "qasper2": ("qasper_text.txt", 20480),
    "musique2": ("musique_text.txt", 20480),
    "long2": ("long_text.txt", 20480),
}
CAL = {
    "long": "qasper",
    "qasper": "musique",
    "musique": "long",
    "mfqa": "diag",
    "diag": "mfqa",
    "qasper2": "musique2",
    "musique2": "long2",
    "long2": "qasper2",
}
MODELS = [
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
]
ARMS = ("X_all", "A", "D", "D_16", "E", "E_bos", "S_8", "X_z", "X_z16", "X_bos")
NOISE = (
    ("X_z16", "X_z"),
    ("D_16", "D"),
)  # pairs of arms that differ by an immaterial fp16 rounding
NOISE_BAND, NOISE_SHARE, W1_MARGIN = (0.95, 1.05), 0.25, 0.9
SEQ, T0, NBLK, TAU, GATE_MAX = 4096, 2048, 8, 0.5, 204
PRED = {"W1": ("E", "D", 1.0), "W2": ("S_8", "X_z", 1.05), "W3": ("E", "X_z", 1.5)}
S: dict = {
    "mode": None,
    "mu": {},
    "arm": None,
    "inv": None,
    "tab": None,
    "gate": {},
    "eng": None,
    "eng_n": 0,
    "eng_layers": (),
}


def rot_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, -1)
    return torch.cat((-b, a), -1)


def tables(n: int, dev) -> tuple[torch.Tensor, torch.Tensor]:
    if S["tab"] is None or S["tab"][0].shape[0] != n:
        f = (
            torch.arange(n, dtype=torch.float64, device=dev)[:, None]
            * S["inv"].double().to(dev)[None]
        )
        e = torch.cat([f, f], -1)
        S["tab"] = (e.cos(), e.sin())
    return S["tab"]


def arm_keys(y32: torch.Tensor, li: int, arm: str) -> torch.Tensor:
    """y32 (KVH, n, d): post-RoPE keys of one layer. Returns the arm's reconstruction (fp32)."""
    kvh, n, d = y32.shape
    cos, sin = tables(n, y32.device)
    assert cos.shape[-1] == d, "partial rotary dimensions are not handled"
    y = y32.double()
    kpre = y * cos - rot_half(y) * sin  # de-rotation (attention_scaling is 1)
    mu = S["mu"][li]  # (KVH, d), pre-RoPE, calibration window
    mu = (mu.half().float() if arm == "D_16" else mu).double()  # D_16: the offset stored as fp16
    gate = (1 - torch.einsum("hnd,hd->hn", kpre, mu) / (mu * mu).sum(-1, keepdim=True)) >= TAU
    mur = mu[:, None, :].expand(kvh, n, d)
    rmu = mur * cos + rot_half(mur) * sin
    cen32 = (y - rmu).float()
    dev = y32.device

    def q4c(t: torch.Tensor) -> torch.Tensor:  # quantizers run on the CPU (A2): the GPU's
        return (
            q4x(t.reshape(kvh * n, d).cpu()).reshape(kvh, n, d).to(dev)
        )  # fp32 arithmetic is not bit-identical

    def q8c(t: torch.Tensor) -> torch.Tensor:
        return q8x(t.reshape(kvh * n, d).cpu()).reshape(kvh, n, d).to(dev)

    if arm == "X_all":  # gate statistics on the exact keys
        S["gate"][li] = (gate.sum(1).cpu(), gate[:, 0].cpu())
        return y32
    if arm == "A":
        return q4c(y32)
    c4 = q4c(cen32)
    if li in S["eng_layers"] and arm == "D":  # the quantizer as used, against ggml
        g: Ggml = S["eng"]
        for nm, mine, ref in (
            ("q4 cen", c4, g.q4(cen32.cpu())),
            ("q4 raw", q4c(y32), g.q4(y32.cpu())),
            ("q8 raw", q8c(y32), g.q8(y32.cpu())),
        ):
            bad = same(mine.cpu(), ref)
            if bad:
                raise SystemExit(f"quantizer differs from ggml: layer {li} {nm}: {bad} values")
            S["eng_n"] += ref.numel()
    out = c4.double() + rmu
    if arm in ("D", "D_16"):
        return out.float()
    bos = torch.zeros_like(gate)
    bos[:, 0] = True
    rows = bos if arm in ("E_bos", "X_bos") else gate
    if arm in ("E", "E_bos"):
        sub = q4c(y32).double()
    elif arm == "S_8":
        sub = q8c(y32).double()
    elif arm == "X_z16":
        sub = y32.half().double()
    elif arm in ("X_z", "X_bos"):
        sub = y
    else:
        raise ValueError(arm)
    return torch.where(rows[..., None], sub, out).float()


def c23_attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    import sim_utils as Q5  # exact attention (the registered runs used pkv5_q.attend, same code)

    li = int(module.layer_idx)
    q, k, v = query[0].float(), key[0].float(), value[0].float()
    if S["mode"] == "cal":
        cos, sin = tables(k.shape[1], k.device)
        kd = k.double()
        kpre = kd * cos - rot_half(kd) * sin
        S["cal"][li] = kpre.mean(1)  # pre-RoPE mean key (KVH, d)
        if li == 0 and S.get("kpre0") is not None:  # rotary tables against the model's own k
            kvh, n, d = kpre.shape
            refk = S["kpre0"][0].double().reshape(n, kvh, d).transpose(0, 1)
            S["derot"] = max(S.get("derot", 0.0), float((refk - kpre).norm() / refk.norm()))
    elif S["arm"] is not None:
        k = arm_keys(k, li, S["arm"])
    scale = scaling if scaling is not None else 1 / math.sqrt(q.shape[-1])
    out = Q5.attend(q, k, v, li, None, scale, kw)
    return out.unsqueeze(0).transpose(1, 2).contiguous().to(query.dtype), None


def real(a) -> int:
    import sim_utils as CQ  # text windows and DATA (as cq_lab3)
    from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

    AttentionInterface.register("c23_attn", c23_attn)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dev = a.device
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = (
        AutoModelForCausalLM.from_pretrained(
            a.model_dir, dtype=torch.float32, attn_implementation="c23_attn"
        )
        .to(dev)
        .eval()
    )
    re_ = model.model.rotary_emb
    if abs(float(getattr(re_, "attention_scaling", 1.0)) - 1.0) > 1e-6:
        raise SystemExit("attention_scaling != 1")
    S["inv"] = re_.inv_freq.detach().float()
    at0 = model.model.layers[0].self_attn
    hk = getattr(at0, "k_norm", None) or getattr(at0, "k_proj", None)
    assert hk is not None
    hk.register_forward_hook(lambda m_, i_, o_: S.__setitem__("kpre0", o_.detach()))
    nl = model.config.num_hidden_layers
    S["eng"], S["eng_layers"] = Ggml(a.lib or LIB_DEFAULT), (0, nl // 2, nl - 1)
    b = tok.bos_token_id
    has_bos = b is not None and tok("x", add_special_tokens=True).input_ids[:1] == [b]
    texts = a.texts or [*OLD, *NEW]
    need = sorted({*texts, *(CAL[t] for t in texts)})
    ids, ref, mus = {}, {}, {}

    def logits(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(input_ids=x, use_cache=False).logits[0, T0 - 1 : SEQ - 1].float()

    def compare(
        z0: torch.Tensor, za: torch.Tensor, tgt: torch.Tensor
    ) -> tuple[torch.Tensor, float, float]:
        """KL(exact || arm) per position, mean NLL of the arm and top-1 agreement; float64, in
        chunks."""
        kls, nll, agree = [], 0.0, 0.0
        for i in range(0, z0.shape[0], 256):
            lr = torch.log_softmax(z0[i : i + 256].to(za.device).double(), -1)
            la = torch.log_softmax(za[i : i + 256].double(), -1)
            kls.append((lr.exp() * (lr - la)).sum(-1))
            nll += float(-la.gather(1, tgt[i : i + 256, None]).sum())
            agree += float((la.argmax(-1) == lr.argmax(-1)).sum())
        return torch.cat(kls), nll / z0.shape[0], agree / z0.shape[0]

    for t in need:  # exact pass: reference logits and the offset
        fn, st0 = {**OLD, **NEW}[t]
        x = CQ.text_ids(tok, CQ.DATA + fn, st0, SEQ, dev)
        if x.shape[1] != SEQ:
            raise SystemExit(f"{t}: {x.shape[1]} tokens")
        ids[t] = x
        S["mode"], S["cal"], S["arm"] = "cal", {}, None
        z0 = logits(x)
        S["mode"] = None
        assert len(S["cal"]) == nl, len(S["cal"])
        if not S.get("derot", 1.0) < 1e-3:
            raise SystemExit(f"de-rotation check failed: {S.get('derot')}")
        ref[t], mus[t] = z0.cpu(), {li: m.float() for li, m in S["cal"].items()}
        del z0
    rows = []
    for t in texts:
        x, z0 = ids[t], ref[t]
        tgt = x[0, T0:SEQ]
        _, nll0, _ = compare(z0, z0.to(dev), tgt)
        S["mu"] = mus[CAL[t]]
        rec: dict = {"text": t, "cal": CAL[t], "nll_exact": nll0, "arms": {}}
        for arm in ARMS:
            S["arm"] = arm
            if arm == "X_all":
                S["gate"] = {}
            za = logits(x)
            S["arm"] = None
            kl, nll, top1 = compare(z0, za, tgt)
            del za
            if not torch.isfinite(kl).all():
                raise SystemExit(f"{t} {arm}: non-finite KL")
            rec["arms"][arm] = {
                "kl": float(kl.mean()),
                "kl_med": float(kl.median()),
                "kl_blocks": [float(c.mean()) for c in kl.chunk(NBLK)],
                "dnll": nll - nll0,
                "top1": top1,
            }
        g = torch.stack(
            [S["gate"][li][0] for li in range(nl)]
        ).float()  # (layers, KVH) gated rows per head (exact keys)
        g0 = torch.stack([S["gate"][li][1] for li in range(nl)])
        rec["gate"] = {
            "mean_rows": float(g.mean()),
            "median_rows": float(g.median()),
            "frac_heads_1_204": float(((g >= 1) & (g <= GATE_MAX)).float().mean()),
            "frac_heads_0": float((g == 0).float().mean()),
            "frac_heads_gt204": float((g > GATE_MAX).float().mean()),
            "frac_rows": float(g.sum() / (g.numel() * SEQ)),
            "frac_heads_row0_gated": float(g0.float().mean()),
            "frac_heads_only_row0": float(((g == 1) & g0).float().mean()),
        }
        rows.append(rec)
        print(a.short, t, {k_: f"{v_['kl']:.3e}" for k_, v_ in rec["arms"].items()}, flush=True)
    meta = {
        "script_md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
        "model": a.model_dir,
        "short": a.short,
        "n_layers": nl,
        "has_bos": bool(has_bos),
        "seq": SEQ,
        "t0": T0,
        "tau": TAU,
        "c22_q_md5": hashlib.md5((HERE / "c22_q.py").read_bytes()).hexdigest(),
        "ggml_md5": S["eng"].md5,
        "engine_values_checked": S["eng_n"],
        "dtype": "float32",
        "derotation_check_rel_err": S["derot"],
        "torch": torch.__version__,
        "sim_utils_md5": hashlib.md5((HERE / "sim_utils.py").read_bytes()).hexdigest(),
    }
    if not S["eng_n"] > 0:
        raise SystemExit("no engine check ran")
    tmp = Path(str(a.out) + ".tmp")
    tmp.write_text(json.dumps({"meta": meta, "rows": rows}, allow_nan=False))
    tmp.replace(a.out)
    return 0


def agg(a) -> int:
    d = a.out
    data, missing = {}, []
    for m in MODELS:
        f = d / f"c23_{m}.json"
        if f.exists():
            data[m] = json.loads(f.read_text())
        else:
            missing.append(m)
    md5s = {
        (
            x["meta"]["script_md5"],
            x["meta"]["ggml_md5"],
            x["meta"]["c22_q_md5"],
            # runs of this repository record sim_utils.py; the registered runs recorded the two
            # harness modules it was copied from
            x["meta"].get("sim_utils_md5")
            or (x["meta"].get("pkv5_q_md5"), x["meta"].get("cq_lab3_md5")),
            x["meta"]["torch"],
        )
        for x in data.values()
    }
    if len(md5s) > 1:
        raise SystemExit(f"mixed versions: {sorted(md5s)}")
    out: dict = {
        "missing_models": missing,
        "script_md5": next(iter(md5s))[0] if md5s else None,
        "texts": {},
    }
    for t in [*NEW, *OLD]:
        per = {}
        for m, x in data.items():
            r = next((r_ for r_ in x["rows"] if r_["text"] == t), None)
            if r is None:
                continue
            ar = r["arms"]
            e: dict = {
                "kl": {k_: v_["kl"] for k_, v_ in ar.items()},
                "dnll": {k_: v_["dnll"] for k_, v_ in ar.items()},
                "nll_exact": r["nll_exact"],
                "gate": r["gate"],
                **{
                    f"noise {x_}/{y_}": (ar[x_]["kl"] / ar[y_]["kl"] if ar[y_]["kl"] > 0 else None)
                    for x_, y_ in NOISE
                },
            }
            for nm, den in (
                ("E", "D"),
                ("S_8", "X_z"),
                ("E", "X_z"),
                ("D", "A"),
                ("E", "A"),
                ("E", "E_bos"),
                ("S_8", "E"),
                ("X_bos", "X_z"),
            ):
                e[f"{nm}/{den}"] = ar[nm]["kl"] / ar[den]["kl"] if ar[den]["kl"] > 0 else None
                e[f"blocks {nm}<{den}"] = sum(
                    x_ < y_ for x_, y_ in zip(ar[nm]["kl_blocks"], ar[den]["kl_blocks"])
                )
            for h, (nm, den, thr) in PRED.items():
                e[h] = None if e[f"{nm}/{den}"] is None else e[f"{nm}/{den}"] <= thr
            per[m] = e
        line = {}
        for h in PRED:
            w = [e[h] for e in per.values() if e[h] is not None]
            line[h] = {
                "models": len(w),
                "meeting": sum(w),
                "holds": len(w) >= 6 and sum(w) >= 0.75 * len(w),
            }
        sizes = {}
        for k_ in (
            "E/D",
            "S_8/X_z",
            "E/X_z",
            "D/A",
            "E/A",
            "E/E_bos",
            "S_8/E",
            "X_bos/X_z",
            *(f"noise {x_}/{y_}" for x_, y_ in NOISE),
        ):
            v = sorted(e[k_] for e in per.values() if e[k_] is not None)
            sizes[k_] = (
                None if not v else {"med": st.median(v), "min": v[0], "max": v[-1], "n": len(v)}
            )
        out["texts"][t] = {"line": line, "sizes": sizes, "per_model": per}
    new_ok = not missing and all(len(out["texts"][t]["per_model"]) == len(MODELS) for t in NEW)
    pairs = [
        out["texts"][t]["per_model"][m] for t in NEW for m in out["texts"][t]["per_model"]
    ]  # the 50 new pairs
    noise = {}
    for x_, y_ in NOISE:  # registered noise rule, on the new pairs only
        v = [e[f"noise {x_}/{y_}"] for e in pairs]
        n_out = sum(r is None or not (NOISE_BAND[0] <= r <= NOISE_BAND[1]) for r in v)
        noise[f"{x_}/{y_}"] = {
            "pairs": len(v),
            "outside_band": n_out,
            "none": sum(r is None for r in v),
            "unresolved": bool(v) and n_out / len(v) > NOISE_SHARE,
        }
    out["noise"] = noise
    out["claim_new"] = {}
    for h in PRED:
        votes = {
            t: f"{out['texts'][t]['line'][h]['meeting']}/{out['texts'][t]['line'][h]['models']}"
            for t in NEW
        }
        claim = all(out["texts"][t]["line"][h]["holds"] for t in NEW) if new_ok else None
        if h == "W2" and claim is not None and noise["X_z16/X_z"]["unresolved"]:
            claim = "undecidable"
        out["claim_new"][h] = {"votes": votes, "claim": claim}
    ed = [e["E/D"] for e in pairs if e["E/D"] is not None]
    out["W1_margin"] = {
        "threshold": W1_MARGIN,
        "pairs": len(ed),
        "below": sum(r <= W1_MARGIN for r in ed),
        "noise_D16_D_unresolved": noise["D_16/D"]["unresolved"],
    }
    out["over_selecting"] = {
        m: [
            round(out["texts"][t]["per_model"][m]["gate"]["frac_heads_gt204"], 3)
            for t in NEW
            if m in out["texts"][t]["per_model"]
        ]
        for m in data
    }
    out["complete_new"] = new_ok
    print(json.dumps(out, indent=1, allow_nan=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["real", "agg"])
    ap.add_argument("out", type=Path)
    ap.add_argument("model_dir", nargs="?")
    ap.add_argument("short", nargs="?")
    ap.add_argument("--lib")
    ap.add_argument("--texts", nargs="*")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    return {"real": real, "agg": agg}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
