# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""One clean 16-bit forward per model: every per-cell feature used by the predictor comparison
(GATE-PRED a1) and the mechanism-discrimination study (MECH-D).

Nothing is quantised in the model's forward pass. The hook only reads the post-RoPE queries and keys
and computes, per layer, statistics of what an engine quantiser *would* see, in the engine's
coordinates:
  - channel order: "hf" or "gguf" (llama.cpp permutes q/k of Llama-family models: engine channel 2j+s is HF
    channel s*d/2+j); the runner passes the order per model.
  - basis: "identity" (LLAMA_ATTN_ROT_DISABLE=1) or "hadamard" (whole-head Hadamard, llama.cpp default).
Formats: q4_0 (symmetric 32-blocks, step = absmax/8), q4_1 (asymmetric, step = (max-min)/15), q8_0
(step = absmax/127).

Per layer and format we report:
  b_meas  = mean over (kv head, block) of log2(2*sqrt(3)*sigma_b / mean_t step_tb)   (resolution left for the
            token content; sigma_b is the per-element std of the centred keys of block b; no
            quantisation)
  b_eff   = mean over (kv head, block) of 3 - log2(r_b), r_b = max|mu_b| / sigma_b (the analytic law)
  key_rel_mse = ||K - Q(K)||^2 / ||K||^2 (simulated quantiser on the captured keys)
  attn_kl  = mean KL(p_fp || p_q) over heads and the last --nq queries (K quantised, V exact, fp32)
  out_cos  = mean cosine(o_fp, o_q) over heads and queries (SageAttention-style per-layer check)
Format-independent (16-bit attention of the last --nq queries): normalised entropy, top-10 mass,
fraction of heads whose top-10 mass < 0.5, mass on position 0. Weights: the vLLM PR 63081 statistic
(largest |k bias| on the slowest-rotating eighth of rotary pairs, first layer with a key bias) and
has_k_bias.

Token statistics (mu, sigma, steps) skip the first --skip positions (default 4); attention uses all
keys. Usage: python kvprobe_unified.py --model DIR --order hf|gguf --out JSON [--text FILE] [--seq
2048] [--nq 256]
       [--skip 4] [--device cuda|cpu] [--dtype bfloat16|float32]
"""

from __future__ import annotations
import os

import argparse
import json
import sys
import math
from pathlib import Path

import torch
from transformers import AttentionInterface, AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ggml primitives come from the library (spintare/ggml_ref.py); KVREL_ROOT (experiments/scripts/env.sh) or the
# repository root of this file locates it, also for copies staged under $KVREL_WORK
sys.path.insert(0, str(Path(os.environ.get("KVREL_ROOT") or Path(__file__).resolve().parents[3]) / "src"))
from spintare.ggml_ref import gguf_perm, hadamard, quant_f32scale, steps_f32scale  # noqa: E402

ST: dict = {"layers": {}, "args": None, "perm": {}, "had": {}}
FORMATS = ("q4_0", "q4_1", "q8_0")


def to_engine(x: torch.Tensor, order: str, basis: str) -> torch.Tensor:
    """x: [..., d] in HF channel order -> engine coordinates."""
    d = x.shape[-1]
    if order == "gguf":
        key = (d, str(x.device))
        if key not in ST["perm"]:
            ST["perm"][key] = gguf_perm(d).to(x.device)
        x = x[..., ST["perm"][key]]
    if basis == "hadamard":
        key = (d, str(x.device))
        if key not in ST["had"]:
            ST["had"][key] = hadamard(d, x.device)
        x = x @ ST["had"][key]
    return x


def block_features(k: torch.Tensor, fmt: str, skip: int) -> dict:
    """k: [kvh, L, d] engine coordinates (float32)."""
    ks = k[:, skip:, :]
    kvh, n, d = ks.shape
    kb = ks.reshape(kvh, n, d // 32, 32)
    mu = kb.mean(1)  # [kvh, nb, 32]
    c = kb - mu.unsqueeze(1)
    sigma = c.pow(2).mean(dim=(1, 3)).sqrt()  # [kvh, nb]
    st = steps_f32scale(ks, fmt).mean(1)  # [kvh, nb]
    ok = (sigma > 0) & (st > 0)
    bmeas = torch.log2(2.0 * math.sqrt(3.0) * sigma[ok] / st[ok])
    r = mu.abs().amax(-1)[ok] / sigma[ok]
    beff = 3.0 - torch.log2(torch.clamp(r, min=1e-12))
    return {
        "b_meas": float(bmeas.mean()),
        "b_meas_min": float(bmeas.min()),
        "b_eff": float(beff.mean()),
        "b_eff_min": float(beff.min()),
        "r_mean": float(r.mean()),
        "r_max": float(r.max()),
    }


def attn_hook(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    a = ST["args"]
    li = int(getattr(module, "layer_idx", len(ST["layers"])))
    q, k, v = query.float(), key.float(), value.float()  # [1, H, L, d], [1, KVH, L, d]
    H, KVH, L, d = q.shape[1], k.shape[1], q.shape[2], q.shape[-1]
    g = H // KVH
    scale = scaling if scaling is not None else 1.0 / math.sqrt(d)
    # full attention output for the model's own forward (exact, 16-bit path semantics in fp32)
    kk, vv = (k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)) if g > 1 else (k, v)
    causal = torch.ones(L, L, dtype=torch.bool, device=q.device).tril()
    win = kw.get("sliding_window")
    if win:
        causal &= ~torch.ones(L, L, dtype=torch.bool, device=q.device).tril(-int(win))
    if isinstance(attention_mask, torch.Tensor):
        m = attention_mask[0, 0, :L, :L] if attention_mask.dim() == 4 else attention_mask[:L, :L]
        causal &= m if m.dtype == torch.bool else (m > -1e4)
    softcap = kw.get("softcap")

    def scores(qx: torch.Tensor, kx: torch.Tensor, eq: str) -> torch.Tensor:
        s = torch.einsum(eq, qx, kx) * scale
        return torch.tanh(s / softcap) * softcap if softcap else s

    s_full = scores(q, kk, "bhld,bhmd->bhlm")
    s_full = s_full.masked_fill(~causal, float("-inf"))
    out = torch.einsum("bhlm,bhmd->bhld", torch.softmax(s_full, -1), vv)
    if li not in ST["layers"] and d % 32 == 0:
        rec: dict = {"d": d, "H": H, "KVH": KVH}
        nq = min(a.nq, L)
        qs = q[0, :, L - nq :, :]  # [H, nq, d]
        sc = s_full[0, :, L - nq :, :]  # [H, nq, L]
        p_fp = torch.softmax(sc, -1)
        o_fp = torch.einsum("hqm,hmd->hqd", p_fp, vv[0])
        logp_fp = torch.log_softmax(sc, -1)
        ent = -(p_fp * torch.where(p_fp > 0, logp_fp, torch.zeros_like(logp_fp))).sum(-1)
        nvalid = causal[L - nq :, :].sum(-1).float().clamp(min=2.0)
        top10 = p_fp.topk(min(10, L), -1).values.sum(-1)  # [H, nq]
        rec["attn"] = {
            "norm_entropy": float((ent / torch.log(nvalid)).mean()),
            "top10_mass": float(top10.mean()),
            "diffuse_head_frac": float((top10.mean(-1) < 0.5).float().mean()),
            "sink_mass": float(p_fp[..., 0].mean()),
        }
        for basis in ("identity", "hadamard"):
            ke = to_engine(k[0], a.order, basis)  # [KVH, L, d]
            qe = to_engine(qs, a.order, basis)  # [H, nq, d]
            for fmt in FORMATS:
                f = block_features(ke, fmt, a.skip)
                kq = quant_f32scale(ke, fmt)
                f["key_rel_mse"] = float((ke - kq).pow(2).sum() / ke.pow(2).sum())
                kqr = kq.repeat_interleave(g, 0) if g > 1 else kq
                sq = scores(qe, kqr, "hqd,hmd->hqm")
                sq = sq.masked_fill(~causal[L - nq :, :].unsqueeze(0), float("-inf"))
                logp_q = torch.log_softmax(sq, -1)
                kl = p_fp * (
                    logp_fp - torch.where(torch.isfinite(logp_q), logp_q, torch.zeros_like(logp_q))
                )
                kl = torch.where(p_fp > 0, kl, torch.zeros_like(kl)).sum(-1)
                o_q = torch.einsum("hqm,hmd->hqd", logp_q.exp(), vv[0])
                cos = torch.nn.functional.cosine_similarity(o_fp, o_q, dim=-1)
                f["attn_kl"] = float(kl.mean())
                f["out_cos"] = float(cos.mean())
                rec[f"{basis}/{fmt}"] = f
        ST["layers"][li] = rec
    return out.transpose(1, 2).to(query.dtype), None


AttentionInterface.register("kvprobe_unified", attn_hook)


def bias_stat(model, cfg) -> dict:
    """vLLM PR 63081: first attention layer with a key bias; largest |bias| on the slowest-rotating
    eighth of the rotary pairs. NeoX layout pairs (i, i + rot/2); GLM/ChatGLM interleaved pairs (2i,
    2i+1)."""
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    rp = getattr(cfg, "rope_parameters", None) or {}
    prf = (
        getattr(cfg, "partial_rotary_factor", None)
        or (rp.get("partial_rotary_factor") if isinstance(rp, dict) else None)
        or getattr(cfg, "rotary_pct", None)
    )
    rot = getattr(cfg, "rotary_dim", None) or int(d * (prf if prf else 1.0))
    mt = str(getattr(cfg, "model_type", "")).lower()
    interleaved = mt.startswith("glm") or mt.startswith("chatglm")
    if interleaved and not getattr(cfg, "rotary_dim", None):
        rot = d // 2
    nkv = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    for name, p in model.named_parameters():
        lname = name.lower()
        if not lname.endswith("bias"):
            continue
        b = None
        if "k_proj" in lname:
            b = p.detach().float()
        elif mt == "gpt_neox" and "query_key_value" in lname:
            # GPT-NeoX fuses q/k/v per head: [n_head, 3, d]
            b = p.detach().float().reshape(cfg.num_attention_heads, 3, d)[:, 1, :]
        elif "qkv" in lname or "query_key_value" in lname:
            qsz = cfg.num_attention_heads * d
            ksz = nkv * d
            if p.numel() >= qsz + 2 * ksz:
                b = p.detach().float()[qsz : qsz + ksz]
        if b is None:
            continue
        b = b.reshape(nkv, d)[:, :rot]
        npair = rot // 2
        lo = npair - max(1, npair // 8)
        if interleaved:
            idx = [2 * i + s for i in range(lo, npair) for s in (0, 1)]
        else:
            idx = [i + s * npair for i in range(lo, npair) for s in (0, 1)]
        return {
            "has_k_bias": True,
            "first_bias_param": name,
            "bias_stat": float(b[:, idx].abs().max()),
            "bias_max_all": float(b.abs().max()),
            "rot_dim": rot,
            "interleaved": interleaved,
        }
    return {"has_k_bias": False, "bias_stat": 0.0, "rot_dim": rot, "interleaved": interleaved}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--order", required=True, choices=["hf", "gguf"])
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--text", default=os.path.join(os.environ.get("KVREL_DATA", "work/data"), "long_text.txt")
    )
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--nq", type=int, default=256)
    ap.add_argument("--skip", type=int, default=4)
    ap.add_argument("--add-bos", type=int, choices=[0, 1], default=0)
    ap.add_argument(
        "--bos-id", type=int, default=-1, help="GGUF BOS id (check_layout gguf_bos_id); -1 = HF id"
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    a = ap.parse_args()
    ST["args"] = a
    torch.manual_seed(0)
    dev = torch.device(a.device)
    dty = torch.bfloat16 if a.dtype == "bfloat16" else torch.float32
    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    ids = tok(
        Path(a.text).read_text(encoding="utf-8", errors="replace"),
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0][: a.seq - a.add_bos]
    bos = None
    if a.add_bos:
        hf_bos = tok.bos_token_id
        if a.bos_id >= 0 and hf_bos is not None and hf_bos != a.bos_id:
            raise SystemExit(f"BOS id mismatch: HF {hf_bos} vs GGUF {a.bos_id}")
        bos = a.bos_id if a.bos_id >= 0 else hf_bos
        assert bos is not None, (
            "add-bos requested but neither the GGUF nor the tokenizer has a BOS id"
        )
        ids = torch.cat([torch.tensor([bos]), ids])
    ids = ids.unsqueeze(0).to(dev)
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    # the engine applies its K Hadamard only when head_dim % 64 == 0; every listed model satisfies
    # this
    assert hd % 64 == 0, f"head_dim {hd} not a multiple of 64 (engine rule differs; not supported)"
    model, info = AutoModelForCausalLM.from_pretrained(
        a.model,
        dtype=dty,
        trust_remote_code=True,
        attn_implementation="kvprobe_unified",
        output_loading_info=True,
    )
    bad = [
        k for k in info.get("unexpected_keys", []) if "attn" in k or "attention" in k or "bias" in k
    ]
    bad += [k for k in info.get("missing_keys", []) if "attn" in k or "attention" in k]
    if bad:
        raise SystemExit(f"attention weights not loaded as stored: {bad[:8]}")
    model = model.to(dev)
    model.eval()
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    assert ST["layers"], "the attention hook never ran (model does not use AttentionInterface?)"
    nl = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "num_layers", None)
    res = {
        "model": a.model,
        "order": a.order,
        "seq": int(ids.numel()),
        "skip": a.skip,
        "nq": a.nq,
        "bos": bos,
        "loading_info": {
            k: list(v)[:20] for k, v in info.items() if k in ("missing_keys", "unexpected_keys")
        },
        "layers_captured": len(ST["layers"]),
        "layers_config": nl,
        "weights": bias_stat(model, cfg),
        "params_b": sum(p.numel() for p in model.parameters()) / 1e9,
        "per_layer": ST["layers"],
    }
    summ: dict = {}
    for basis in ("identity", "hadamard"):
        for fmt in FORMATS:
            key = f"{basis}/{fmt}"
            for field in ("b_meas", "b_eff", "key_rel_mse", "attn_kl", "out_cos", "r_mean"):
                vals = [L[key][field] for L in ST["layers"].values() if key in L]
                summ[f"{key}/{field}_mean"] = sum(vals) / len(vals)
                summ[f"{key}/{field}_min"] = min(vals)
                summ[f"{key}/{field}_max"] = max(vals)
    for field in ("norm_entropy", "top10_mass", "diffuse_head_frac", "sink_mass"):
        vals = [L["attn"][field] for L in ST["layers"].values()]
        summ[f"attn/{field}_mean"] = sum(vals) / len(vals)
    res["summary"] = summ
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {
                "model": a.model,
                "layers": res["layers_captured"],
                "of": nl,
                "q4_0_had_bmeas": summ["hadamard/q4_0/b_meas_mean"],
                "bias_stat": res["weights"]["bias_stat"],
            }
        )
    )


if __name__ == "__main__":
    main()
