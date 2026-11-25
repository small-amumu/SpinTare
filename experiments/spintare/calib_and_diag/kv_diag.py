# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""Pre-deployment diagnosis for any HF causal LM: per-layer offset ratio of the pre-RoPE keys,
r_l = mean over KV heads of |mu_h| / mean |k - mu_h|. Needs a model directory and a short English
text and finishes in seconds. Also reports whether k_proj has a bias, its norm and its cosine with
mu (does the offset come from the bias or the activations). Usage: python kv_diag.py <model_dir>
<text_file> [--T 4096] [--out result.json] Falls back to the CPU (float32) without a GPU; for models
with one fused q/k/v projection (e.g. the qkv_proj of Phi-3) K is cut out of the fused output.
"""

import json
import os
import sys
import time

T0 = time.time()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def as_thd(out, sl, nkv, hd):
    """Bring captured keys to [T, H, D]. k_norm acts on [B, T, H, D] in Qwen3, on [B, H, T, D] in
    Gemma3."""
    x = (out[..., sl[0] : sl[1]] if sl else out).float()
    if x.dim() == 4 and x.shape[1] == nkv and x.shape[2] != nkv:
        x = x.transpose(1, 2)
    return x.reshape(x.shape[1], -1, hd)


@torch.no_grad()
def main():
    path, text = sys.argv[1], open(sys.argv[2], encoding="utf-8").read()
    T = int(sys.argv[sys.argv.index("--T") + 1]) if "--T" in sys.argv else 4096
    t_imp = time.time() - T0
    t1 = time.time()
    tok = AutoTokenizer.from_pretrained(path)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = (
        AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16
            if (dev != "cpu" or os.environ.get("DIAG_BF16"))
            else torch.float32,
        )
        .to(dev)
        .eval()
    )
    cfg = getattr(model.config, "text_config", None) or model.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    nq, nkv = (
        cfg.num_attention_heads,
        getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads,
    )
    cap, kproj = {}, {}
    layers = (
        model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
    )
    for i, layer in enumerate(layers):
        a = layer.self_attn
        if hasattr(a, "k_norm"):
            m, sl = a.k_norm, None
        elif hasattr(a, "k_proj"):
            m, sl = a.k_proj, None
        else:  # fused qkv: output laid out as [q | k | v]
            m, sl = a.qkv_proj, (nq * hd, nq * hd + nkv * hd)
        kproj[i] = getattr(a, "k_proj", None)
        m.register_forward_hook(
            lambda mod, inp, out, i=i, sl=sl: cap.__setitem__(i, as_thd(out, sl, nkv, hd))
        )
    ids = tok(text, return_tensors="pt").input_ids[:, :T].to(dev)
    if dev != "cpu":
        torch.cuda.synchronize()
    t_load = time.time() - t1
    t2 = time.time()
    model(ids)
    rows = []
    for i in sorted(cap):
        k = cap[i]
        mu = k.mean(0)
        r = float((mu.norm(dim=-1) / (k - mu).norm(dim=-1).mean(0)).mean())
        b = kproj[i].bias if kproj[i] is not None else None
        cos = (
            float(
                torch.nn.functional.cosine_similarity(b.float().reshape(-1, hd), mu, dim=-1).mean()
            )
            if b is not None
            else None
        )
        kb = k.reshape(k.shape[0], k.shape[1], -1, 32)
        mb = kb.mean(0)  # llama.cpp q4_0 / q4_1: one scale per 32 values
        rb = float(
            (mb.norm(dim=-1) / (kb - mb).norm(dim=-1).mean(0)).mean(0).max()
        )  # mean over heads, then the worst block
        rows.append(
            {
                "layer": i,
                "r": round(r, 3),
                "r_blk32": round(rb, 3),
                "bias_cos_mu": None if cos is None else round(cos, 3),
            }
        )
    if dev != "cpu":
        torch.cuda.synchronize()
    t_diag = time.time() - t2
    rmax = max(rows, key=lambda x: x["r"])
    out = {
        "model": path,
        "T": ids.shape[1],
        "has_k_bias": rows[0]["bias_cos_mu"] is not None,
        "r_max": rmax["r"],
        "argmax_layer": rmax["layer"],
        "r_blk32_max": max(x["r_blk32"] for x in rows),
        "argmax_blk_layer": max(rows, key=lambda x: x["r_blk32"])["layer"],
        "layers": rows,
        "verdict": "per-token symmetric 4-bit K is predicted to collapse (r_max > 3)"
        if rmax["r"] > 3
        else "r_max <= 3: symmetric 4-bit K predicted safe",
    }
    print(json.dumps({k: v for k, v in out.items() if k != "layers"}))
    t_warm = None
    if "--timing" in sys.argv:  # second forward: no CUDA init or first compiles
        t3 = time.time()
        model(ids)
        for i in sorted(cap):
            k = cap[i]
            mu = k.mean(0)
            float((mu.norm(dim=-1) / (k - mu).norm(dim=-1).mean(0)).mean())
        if dev != "cpu":
            torch.cuda.synchronize()
        t_warm = time.time() - t3
    print(
        f"TIMING import {t_imp:.1f}s load {t_load:.1f}s forward+r {t_diag:.2f}s"
        + (f" warm {t_warm:.2f}s" if t_warm is not None else "")
    )
    print("r per layer:", " ".join(f"{x['r']:.2f}" for x in rows))
    dst = (
        sys.argv[sys.argv.index("--out") + 1]
        if "--out" in sys.argv
        else path.rstrip("/") + "_kvdiag.json"
    )
    json.dump(out, open(dst, "w"), indent=1)


if __name__ == "__main__":
    main()
