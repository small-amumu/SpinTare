# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Dump post-RoPE keys of one layer, all KV heads, for a 3D surface figure (descriptive, no
measurement).

One model load per call, truncated to layers 0..layer (the keys of a layer do not depend on later
layers, so the truncation is exact and skips reading their weights). Keys are read where
apply_rotary_pos_emb returns them, on the first --T tokens of a text; the per-channel mean over
those tokens is the postmean vector of each head. Saves keys[:, :show] and means as .npz. Usage:
python dump_keys.py <hf_dir> <text> <layer> <out.npz> [--T 512] [--show 200]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def arg(name: str, default: int) -> int:
    return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


@torch.no_grad()
def main() -> int:
    hf, text, layer, out = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    T, show = arg("--T", 512), arg("--show", 200)
    tok = AutoTokenizer.from_pretrained(hf)
    cfg = AutoConfig.from_pretrained(hf)
    cfg.num_hidden_layers = layer + 1
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = cfg.layer_types[: layer + 1]
    model = AutoModelForCausalLM.from_pretrained(hf, config=cfg, torch_dtype=torch.bfloat16).eval()
    n_layer = model.config.num_hidden_layers
    mod = sys.modules[type(model.model.layers[0].self_attn).__module__]
    orig = mod.apply_rotary_pos_emb
    st = {"i": 0, "k": None}

    def wrapped(q, k, *a, **kw):
        qo, ko = orig(q, k, *a, **kw)
        if st["i"] % n_layer == layer:
            st["k"] = ko[0].float().cpu().numpy()  # [H_kv, T, d]
        st["i"] += 1
        return qo, ko

    mod.apply_rotary_pos_emb = wrapped
    ids = tok(text.read_text(encoding="utf-8"), return_tensors="pt").input_ids[:, :T]
    model(ids)
    k = st["k"]
    np.savez(out, keys=k[:, :show], mean=k.mean(1), layer=layer, T=k.shape[1])
    print(out, k.shape, "max |m| per head", np.abs(k.mean(1)).max(-1).round(1).tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
