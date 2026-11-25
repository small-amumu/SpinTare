# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Pre-RoPE key mean mu and the model's RoPE frequencies, for the vLLM add-back (KVMR file).

mu[l][h] is the mean over T calibration tokens of the pre-RoPE key of layer l, KV head h, taken
exactly where apply_rotary_pos_emb receives it (after any k bias and k norm), as the llama.cpp
mu_calib.py. inv_freq is the model's own rotary_emb.inv_freq (after frequency scaling such as
llama3); models whose RoPE rescales the output (attention_scaling != 1, e.g. YaRN), rotates part of
the head, or uses interleaved pairs are refused.
Usage: python -m spintare_vllm calib-mu <hf_model_dir> <calib_text> <out.kvmr> [--T 2048]
       [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from spintare_vllm import kvmr

NEOX_TYPES = ("llama", "mistral", "qwen2", "qwen3")  # rotate_half (NeoX pairs) in HF


@torch.no_grad()
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="spintare_vllm calib-mu")
    ap.add_argument("model")
    ap.add_argument("text")
    ap.add_argument("out")
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(a.device).eval()
    cfg = model.config
    if cfg.model_type not in NEOX_TYPES:
        sys.exit(f"model type {cfg.model_type} not in {NEOX_TYPES} (NeoX RoPE not verified)")
    if float(getattr(cfg, "partial_rotary_factor", 1.0)) != 1.0:
        sys.exit("partial rotary embeddings are not supported")
    rot = model.model.rotary_emb
    asc = float(getattr(rot, "attention_scaling", 1.0))
    if asc != 1.0:
        sys.exit(f"RoPE attention_scaling {asc} != 1 (e.g. YaRN) is not supported")
    inv_freq = rot.inv_freq.detach().float().cpu()
    n_layer: int = cfg.num_hidden_layers
    attn0 = model.model.layers[0].self_attn
    mod = sys.modules[type(attn0).__module__]
    orig = mod.apply_rotary_pos_emb
    mu: dict[int, torch.Tensor] = {}
    calls = {"i": 0}

    def wrapped(q: torch.Tensor, k: torch.Tensor, *args: Any, **kw: Any) -> Any:
        layer = calls["i"] % n_layer
        calls["i"] += 1
        mu[layer] = k.float().mean(dim=2)[0]  # k: [B, H_kv, T, D] -> [H_kv, D]
        return orig(q, k, *args, **kw)

    mod.apply_rotary_pos_emb = wrapped
    ids = tok(Path(a.text).read_text(encoding="utf-8"), return_tensors="pt").input_ids
    ids = ids[:, : a.T].to(a.device)
    if ids.shape[1] != a.T:
        sys.exit(f"calibration text too short: {ids.shape[1]} < {a.T} tokens")
    t0 = time.time()
    model(ids)
    if len(mu) != n_layer or calls["i"] != n_layer:
        sys.exit(f"hook saw {calls['i']} calls / {len(mu)} layers, expected {n_layer}")
    data = torch.stack([mu[i] for i in range(n_layer)]).cpu()
    if inv_freq.shape != (data.shape[2] // 2,):
        sys.exit(f"inv_freq {tuple(inv_freq.shape)} does not cover head_dim {data.shape[2]}")
    kvmr.write(a.out, data, inv_freq)
    norms = data.norm(dim=-1)
    print(
        json.dumps(
            {
                "model": a.model,
                "T": a.T,
                "n_layer": n_layer,
                "n_head_kv": int(data.shape[1]),
                "head_dim": int(data.shape[2]),
                "mu_norm_max": round(float(norms.max()), 3),
                "layer_of_max": int(norms.max(dim=1).values.argmax()),
                "inv_freq_min": float(inv_freq.min()),
                "sec": round(time.time() - t0, 1),
                "out": a.out,
            }
        )
    )


if __name__ == "__main__":
    main()
