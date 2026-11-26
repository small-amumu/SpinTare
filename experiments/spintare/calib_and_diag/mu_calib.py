# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""B1: estimate the shared key component mu for the llama.cpp KV de-offset patch
(LLAMA_KV_DEBIAS_MU).

mu[l][h] is the mean over calibration tokens of the pre-RoPE key of layer l, kv head h, taken
exactly where apply_rotary_pos_emb receives it (after any k bias and k norm). Output: the KVMU file
read by src/models/kv-mu.h. Usage: python mu_calib.py <hf_model_dir> <calib_text> <out.kvmu> [--T
2048]
"""

import json
import struct
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def arg(name: str, default: int) -> int:
    return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


@torch.no_grad()
def main() -> None:
    model_dir, text_path, out_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    n_tok = arg("--T", 2048)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = (
        AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).to(dev).eval()
    )
    n_layer: int = model.config.num_hidden_layers
    attn0 = (
        model.gpt_neox.layers[0].attention
        if hasattr(model, "gpt_neox")
        else model.model.layers[0].self_attn
    )
    mod = sys.modules[type(attn0).__module__]
    orig = mod.apply_rotary_pos_emb
    mu: dict[int, torch.Tensor] = {}
    calls = {"i": 0}

    def wrapped(q: torch.Tensor, k: torch.Tensor, *a, **kw):
        layer = calls["i"] % n_layer
        calls["i"] += 1
        mu[layer] = k.float().mean(dim=2)[0]  # k: [B, H_kv, T, D] -> [H_kv, D]
        return orig(q, k, *a, **kw)

    mod.apply_rotary_pos_emb = wrapped
    ids = (
        tok(text_path.read_text(encoding="utf-8"), return_tensors="pt").input_ids[:, :n_tok].to(dev)
    )
    assert ids.shape[1] == n_tok, f"calibration text too short: {ids.shape[1]} < {n_tok} tokens"
    t0 = time.time()
    model(ids)
    assert len(mu) == n_layer, f"hook saw {len(mu)} layers, expected {n_layer}"
    n_head_kv, head_dim = mu[0].shape
    data = torch.stack([mu[i] for i in range(n_layer)]).cpu().contiguous()  # [L, H_kv, D]
    with out_path.open("wb") as f:
        f.write(b"KVMU")
        f.write(struct.pack("<4i", 1, n_layer, n_head_kv, head_dim))
        f.write(data.numpy().astype("<f4").tobytes())
    norms = data.norm(dim=-1)  # [L, H_kv]
    top = int(norms.max(dim=1).values.argmax())
    print(
        json.dumps(
            {
                "model": model_dir,
                "T": n_tok,
                "n_layer": n_layer,
                "n_head_kv": int(n_head_kv),
                "head_dim": int(head_dim),
                "mu_norm_max": round(float(norms.max()), 3),
                "layer_of_max": top,
                "sec": round(time.time() - t0, 1),
                "out": str(out_path),
            }
        )
    )


if __name__ == "__main__":
    main()
