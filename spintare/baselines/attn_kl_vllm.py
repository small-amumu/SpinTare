# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""Simulated attention KL for vLLM's int4_per_token_head KV format.

Follows the write path of vllm/v1/attention/ops/int4_per_token_head.py at vLLM commit 0c16eee3 (vLLM
is Apache-2.0, Copyright the vLLM project contributors); this file re-states that format in PyTorch
for an offline simulation of K only, and is not vLLM code.

vLLM write path: K' = H*D*k (single randomized Hadamard: fixed +-1 signs D from torch.Generator seed
0x9E3779B9, unnormalized Sylvester Hadamard H), then per-(token, head) asymmetric INT4 over the
whole head: scale = max((max-min)/15, 1e-6), zero_point = round_half_away(-min/scale) clamped to [0,
15], q = clamp(round(x/scale)+zp, 0, 15). Q is rotated the same way, so scores equal q.k up to the d
factor; we simulate in the orthonormal basis (equivalent). Measures KL(original || quantized) of
attention for the last Q queries per layer (K only; V is also INT4 in vLLM but is not part of this
score). Gemma-family models, models with a logit softcap, and models whose active sliding window is
shorter than --T are refused (not modelled here). Usage: python -m spintare attn-kl-vllm <model_dir>
<text_file> [--T 2048] [--Q 256] [--out file.json]
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def hadamard(n: int, dev: str | torch.device) -> torch.Tensor:
    h = torch.ones(1, 1, device=dev)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def vllm_signs(d: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0x9E3779B9)
    return 2.0 * torch.bernoulli(torch.full((d,), 0.5), generator=gen) - 1.0


def round_half_away(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


def int4_asym_token_head(x: torch.Tensor) -> torch.Tensor:
    mn, mx = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    scale = ((mx - mn) / 15.0).clamp(min=1e-6)
    zp = round_half_away(-mn / scale).clamp(0, 15)
    q = (round_half_away(x / scale) + zp).clamp(0, 15)
    return (q - zp) * scale


@torch.no_grad()
def main() -> int:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    path = sys.argv[1]
    with open(sys.argv[2], encoding="utf-8") as f:
        text = f.read()

    def arg(k: str, v: Any) -> Any:
        return type(v)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else v

    T, Q = arg("--T", 2048), arg("--Q", 256)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(path)
    dt = (
        torch.bfloat16
        if (dev != "cpu" or os.environ.get("SPINTARE_BF16") or os.environ.get("TAREKV_BF16"))
        else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dt).to(dev).eval()
    cfg = getattr(model.config, "text_config", None) or model.config
    if getattr(cfg, "attn_logit_softcapping", None) or str(cfg.model_type).startswith("gemma"):
        sys.exit(
            f"{cfg.model_type}: Gemma-family models (logit softcap, local attention layers) and "
            "softcapped models are not modelled here"
        )
    win = getattr(cfg, "sliding_window", None)
    if win and getattr(cfg, "use_sliding_window", True) is not False and win < T:
        sys.exit(f"sliding window {win} < --T {T}: windowed attention is not modelled here")
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    assert hd & (hd - 1) == 0, f"head dim {hd}: the vLLM format needs a power-of-two Hadamard"
    layers = model.model.layers
    scales = [float(getattr(x.self_attn, "scaling", None) or 1.0 / math.sqrt(hd)) for x in layers]
    mod = sys.modules[type(layers[0].self_attn).__module__]
    orig, cap = mod.apply_rotary_pos_emb, []

    def wrapped(
        q: torch.Tensor, k: torch.Tensor, *a: Any, **kw: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qo, ko = orig(q, k, *a, **kw)
        cap.append((qo[0, :, -Q:].float(), ko[0].float()))
        return qo, ko

    mod.apply_rotary_pos_emb = wrapped
    ids = tok(text, return_tensors="pt").input_ids[:, :T].to(dev)
    t0 = time.time()
    try:
        model(ids)
    finally:
        mod.apply_rotary_pos_emb = orig
    assert len(cap) == len(layers), f"captured {len(cap)} of {len(layers)} layers"
    n = ids.shape[1]
    Q = min(Q, n)
    cap = [(q[:, -Q:], k) for q, k in cap]
    rows = []
    for i, (q, k) in enumerate(cap):
        hq, hk, d = q.shape[0], k.shape[0], k.shape[-1]
        k = k.repeat_interleave(hq // hk, 0)
        mask = torch.arange(n, device=dev)[None, :] > torch.arange(n - Q, n, device=dev)[:, None]

        def attn(kk: torch.Tensor) -> torch.Tensor:
            s = (q @ kk.transpose(-1, -2)) * scales[i]
            return torch.log_softmax(s.masked_fill(mask, float("-inf")), -1)

        p = attn(k)
        R = hadamard(d, dev) @ torch.diag(vllm_signs(d).to(dev))  # orthonormal H*D
        k_vllm = int4_asym_token_head(k @ R.T) @ R  # rotate, quantize, rotate back
        out = {"layer": i}
        for name, kq in (("vllm_int4", k_vllm),):
            lq = attn(kq)
            out["kl_" + name] = round(float((p.exp() * (p - lq)).nan_to_num().sum(-1).mean()), 5)
        rows.append(out)
    res = {
        "model": path,
        "T": n,
        "Q": Q,
        "format": "vLLM int4_per_token_head @0c16eee3 (K only)",
        "kl_vllm_int4_max": max(r["kl_vllm_int4"] for r in rows),
        "argmax_vllm_int4": max(rows, key=lambda r: r["kl_vllm_int4"])["layer"],
        "kl_vllm_int4_l0": rows[0]["kl_vllm_int4"],
        "sec": round(time.time() - t0, 2),
        "layers": rows,
    }
    print(json.dumps({k: v for k, v in res.items() if k != "layers"}))
    dst = arg("--out", path.rstrip("/") + "_attndiag_vllm.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
