# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""Offset ratio of the keys, per layer, for Hugging Face decoder LMs with model.model.layers (one
forward).

r_l = mean over kv heads of ||mu_h|| / mean_t ||k_t - mu_h|| on the pre-RoPE keys (the k_proj or
k_norm output); r_blk32 is the same ratio on 32-element blocks (the block size of llama.cpp's q4_0 /
q4_1 scales), worst block. The keys are taken before RoPE, in the Hugging Face channel order and
without the engine's rotation (this is a property of the model, not a simulated quantiser). Fused
q/k/v projections (e.g. Phi-3 qkv_proj) are sliced.
Usage: python -m spintare offset <model_dir> <text_file> [--T 4096] [--out result.json] [--timing]
       [--trust-remote-code]   (runs the checkpoint's own Python code; off by default)
Runs on CPU in float32 when no GPU is present (SPINTARE_BF16=1, or the older name TAREKV_BF16=1,
forces bfloat16).
"""

from __future__ import annotations

import argparse
import json
import os
import time

T0 = time.time()
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def as_thd(out: torch.Tensor, sl: tuple[int, int] | None, nkv: int, hd: int) -> torch.Tensor:
    """Captured K as [T, H, D]. k_norm acts on [B, T, H, D] in Qwen3 and on the transposed
    [B, H, T, D] in Gemma3."""
    x = (out[..., sl[0] : sl[1]] if sl else out).float()
    if x.dim() == 4:
        assert x.shape[1] != x.shape[2], "T equals the number of kv heads: layout is ambiguous"
    if x.dim() == 4 and x.shape[1] == nkv and x.shape[2] != nkv:
        x = x.transpose(1, 2)
    return x.reshape(x.shape[1], -1, hd)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(prog="spintare offset")
    ap.add_argument("model")
    ap.add_argument("text")
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--out", default=None, help="default: <model>_offset.json")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true", help="run the checkpoint's code")
    args = ap.parse_args()
    path, T, trc = args.model, args.T, args.trust_remote_code
    with open(args.text, encoding="utf-8") as f:
        text = f.read()
    t_imp = time.time() - T0
    t1 = time.time()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=trc)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if (dev != "cpu" or os.environ.get("SPINTARE_BF16") or os.environ.get("TAREKV_BF16"))
        else torch.float32
    )
    model = (
        AutoModelForCausalLM.from_pretrained(path, trust_remote_code=trc, dtype=dtype)
        .to(dev)
        .eval()
    )
    cfg = getattr(model.config, "text_config", None) or model.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    nq, nkv = (
        cfg.num_attention_heads,
        getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads,
    )
    cap = {}
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
        if hd % 32:
            rb = None
        else:
            kb = k.reshape(
                k.shape[0], k.shape[1], -1, 32
            )  # one q4_0 / q4_1 scale per token per 32 dims
            mb = kb.mean(0)
            rb = float(
                (mb.norm(dim=-1) / (kb - mb).norm(dim=-1).mean(0)).mean(0).max()
            )  # head mean, worst block
        rows.append(
            {
                "layer": i,
                "r": round(r, 3),
                "r_blk32": None if rb is None else round(rb, 3),
            }
        )
    if dev != "cpu":
        torch.cuda.synchronize()
    t_diag = time.time() - t2
    rmax = max(rows, key=lambda x: x["r"])
    blk = [x for x in rows if x["r_blk32"] is not None]
    out = {
        "model": path,
        "T": ids.shape[1],
        "r_max": rmax["r"],
        "argmax_layer": rmax["layer"],
        "r_blk32_max": max(x["r_blk32"] for x in blk) if blk else None,
        "argmax_blk_layer": max(blk, key=lambda x: x["r_blk32"])["layer"] if blk else None,
        "layers": rows,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "layers"}))
    t_warm = None
    if args.timing:  # second forward: excludes CUDA init and first compile
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
    dst = args.out or path.rstrip("/") + "_offset.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
