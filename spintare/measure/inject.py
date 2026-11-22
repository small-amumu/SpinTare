# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""Controlled injection of a shared pre-RoPE key offset into a released checkpoint.

A pre-RoPE key offset on the slowest RoPE pairs (HF rotate_half layout: dims [d/2-P, d/2) and [d-P,
d)) is written into the k bias, with ||b|| = target_r * mean_t ||k_t - mean(k)|| per (layer, kv
head) and a random direction from --seed. The rest of the model is unchanged; the realised offset
ratio (||mu + b|| / content, reported per layer as r_expected) grows with --r. Models that normalise
keys after k_proj (k_norm: Qwen3, Gemma-3) are refused: the norm would rescale the bias.
- models whose k_proj already has a bias (Qwen2): the offset is ADDED to the existing k bias; q/v
  biases are kept and no o_proj bias is written;
- models without biases: zero q/v/o biases are written so that llama.cpp's converter and loader
  accept the k bias; for gemma2 no o_proj bias is written (llama.cpp's gemma2 loader has no o-proj
  bias tensor);
- tied input/output embeddings (gemma-2, SmolLM2): lm_head.weight is dropped from the state dict
  (the loader and convert_hf_to_gguf re-tie it from tie_word_embeddings);
- --as-llama: rewrite the saved config as LlamaForCausalLM (for Mistral-7B-v0.3, whose HF attention
  hard-codes bias=False and would silently drop the injected bias; the architecture is otherwise
  identical to Llama).
Verify the result with `spintare inject-verify`.

Usage: python -m spintare inject <src_model> <out_dir> <diag_text> --r 4 [--layers all|0] [--pairs
12]
       [--seed 0] [--T 2048] [--as-llama]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser(prog="spintare inject")
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("text")
    ap.add_argument("--r", type=float, required=True)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--as-llama", action="store_true")
    a = ap.parse_args()
    c0 = AutoConfig.from_pretrained(a.src)
    hd0 = getattr(c0, "head_dim", None) or c0.hidden_size // c0.num_attention_heads
    if not 0 < a.pairs <= hd0 // 2:
        ap.error(f"--pairs must be in 1..{hd0 // 2} (head dim {hd0})")
    if a.layers != "all":
        try:
            want = [int(x) for x in a.layers.split(",")]
        except ValueError:
            ap.error("--layers must be all or comma-separated integers")
        if any(not 0 <= x < c0.num_hidden_layers for x in want):
            ap.error(f"--layers must be in 0..{c0.num_hidden_layers - 1}")

    tok = AutoTokenizer.from_pretrained(a.src)
    model = AutoModelForCausalLM.from_pretrained(a.src, dtype=torch.bfloat16).eval()
    cfg = model.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    layers = model.model.layers
    has_bias = layers[0].self_attn.k_proj.bias is not None
    if hasattr(layers[0].self_attn, "k_norm"):
        raise SystemExit(
            "model normalises keys after k_proj (k_norm): a k bias offset is not preserved"
        )
    cap: dict[int, torch.Tensor] = {}
    for i, layer in enumerate(layers):
        layer.self_attn.k_proj.register_forward_hook(
            lambda m, inp, out, i=i: cap.__setitem__(
                i, out[0].float().reshape(out.shape[1], nkv, hd)
            )
        )
    ids = tok(Path(a.text).read_text(encoding="utf-8"), return_tensors="pt").input_ids[:, : a.T]
    with torch.no_grad():
        model(ids)

    half = hd // 2
    dims = list(range(half - a.pairs, half)) + list(
        range(hd - a.pairs, hd)
    )  # slowest pairs, both halves
    sel = range(len(layers)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    g = torch.Generator().manual_seed(a.seed)
    sd = {k: v for k, v in model.state_dict().items()}
    meta = {
        "src": a.src,
        "target_r": a.r,
        "layers": a.layers,
        "pairs": a.pairs,
        "dims": dims,
        "seed": a.seed,
        "existing_k_bias": has_bias,
        "as_llama": a.as_llama,
        "per_layer": [],
    }
    for i in range(len(layers)):
        k = cap[i]  # includes the existing k bias, if any
        mu = k.mean(0)  # [nkv, hd]
        content = (k - mu).norm(dim=-1).mean(0)  # [nkv]
        b = torch.zeros(nkv, hd)
        if i in sel and a.r > 0:
            for h in range(nkv):
                u = torch.zeros(hd)
                u[dims] = torch.randn(len(dims), generator=g)
                u = u / u.norm()
                b[h] = a.r * float(content[h]) * u
        p = f"model.layers.{i}.self_attn."
        old = sd[p + "k_proj.bias"].float() if has_bias else torch.zeros(nkv * hd)
        if has_bias:
            sd[p + "k_proj.bias"] = (sd[p + "k_proj.bias"].float() + b.reshape(-1)).to(
                torch.bfloat16
            )
        else:
            sd[p + "k_proj.bias"] = b.reshape(-1).to(torch.bfloat16)
            sd[p + "q_proj.bias"] = torch.zeros(cfg.num_attention_heads * hd, dtype=torch.bfloat16)
            sd[p + "v_proj.bias"] = torch.zeros(nkv * hd, dtype=torch.bfloat16)
            if (
                cfg.model_type != "gemma2"
            ):  # llama.cpp gemma2 has no o-proj bias tensor; HF init zeros it when missing
                sd[p + "o_proj.bias"] = torch.zeros(cfg.hidden_size, dtype=torch.bfloat16)
        r_new = float(((mu + b).norm(dim=-1) / content).mean())
        # the offset as stored: bf16(old + b) - old, which differs from b by the bf16 rounding
        realised = (sd[p + "k_proj.bias"].float() - old).reshape(nkv, hd)
        meta["per_layer"].append(
            {
                "layer": i,
                "content_norm_mean": round(float(content.mean()), 4),
                "bias_norm_mean": round(float(b.norm(dim=-1).mean()), 4),
                "realised_norm_mean": round(float(realised.norm(dim=-1).mean()), 6),
                "r_expected": round(r_new, 3),
            }
        )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not has_bias:
        cfg.attention_bias = True
    cfg.save_pretrained(out)
    if a.as_llama:  # Mistral-7B-v0.3 -> Llama class with biases
        c = json.loads((out / "config.json").read_text(encoding="utf-8"))
        c.update(
            architectures=["LlamaForCausalLM"],
            model_type="llama",
            attention_bias=True,
            mlp_bias=False,
        )
        c.pop("sliding_window", None)
        (out / "config.json").write_text(json.dumps(c, indent=2), encoding="utf-8")
    from safetensors.torch import save_file

    if getattr(cfg, "tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    save_file(
        {k: v.contiguous() for k, v in sd.items() if "rotary_emb" not in k},
        str(out / "model.safetensors"),
        metadata={"format": "pt"},
    )
    for f in Path(a.src).iterdir():
        if (
            f.name.startswith(("tokenizer", "special_tokens", "generation_config", "added_tokens"))
            or f.name in ("vocab.json", "merges.txt", "vocab.txt", "chat_template.json")
            or f.suffix == ".jinja"
        ):
            shutil.copy(f, out / f.name)
    (out / "inject_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(
        "INJECTED",
        out,
        "target_r",
        a.r,
        "layers",
        a.layers,
        "existing_k_bias",
        has_bias,
        "r_expected_max",
        max(x["r_expected"] for x in meta["per_layer"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
