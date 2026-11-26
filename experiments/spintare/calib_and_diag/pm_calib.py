# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""B3 T2: static post-RoPE key mean m for the llama.cpp postmean variant (LLAMA_KV_POSTMEAN).

m[l][h] is the mean over calibration positions 0..T-1 of the post-RoPE key of layer l, kv head h
(the output of apply_rotary_pos_emb, after any k bias and k norm). Output: a KVMU file (same layout
as mu_calib.py). Qwen2 / Qwen3 only: their GGUF keeps the Hugging Face (NeoX) dimension order, so
the post-RoPE layouts match. Usage: python pm_calib.py <hf_model_dir> <calib_text> <out.kvmu> [--T
2048] [--device cpu|cuda] [--bf16]
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
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16 if "--bf16" in sys.argv else torch.float32
        )
        .to(dev)
        .eval()
    )
    n_layer: int = model.config.num_hidden_layers
    mod = sys.modules[type(model.model.layers[0].self_attn).__module__]
    orig = mod.apply_rotary_pos_emb
    m: dict[int, torch.Tensor] = {}
    calls = {"i": 0}

    def wrapped(q: torch.Tensor, k: torch.Tensor, *a, **kw):
        layer = calls["i"] % n_layer
        calls["i"] += 1
        qo, ko = orig(q, k, *a, **kw)
        m[layer] = ko.float().mean(dim=2)[0]  # [B, H_kv, T, D] -> [H_kv, D]
        return qo, ko

    mod.apply_rotary_pos_emb = wrapped
    ids = (
        tok(text_path.read_text(encoding="utf-8"), return_tensors="pt").input_ids[:, :n_tok].to(dev)
    )
    assert ids.shape[1] == n_tok, f"calibration text too short: {ids.shape[1]} < {n_tok} tokens"
    t0 = time.time()
    model(ids)
    assert len(m) == n_layer, f"hook saw {len(m)} layers, expected {n_layer}"
    n_head_kv, head_dim = m[0].shape
    data = torch.stack([m[i] for i in range(n_layer)]).cpu().contiguous()  # [L, H_kv, D]
    with out_path.open("wb") as f:
        f.write(b"KVMU")
        f.write(struct.pack("<4i", 1, n_layer, n_head_kv, head_dim))
        f.write(data.numpy().astype("<f4").tobytes())
    print(
        json.dumps(
            {
                "model": model_dir,
                "T": n_tok,
                "n_layer": n_layer,
                "n_head_kv": int(n_head_kv),
                "head_dim": int(head_dim),
                "m_norm_max": round(float(data.norm(dim=-1).max()), 3),
                "sec": round(time.time() - t0, 1),
                "out": str(out_path),
            }
        )
    )


if __name__ == "__main__":
    main()
