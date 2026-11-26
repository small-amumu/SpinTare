# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Sanity check of a T6c injected checkpoint against its source (CPU, bf16).

Checks: (1) the injected k bias minus the source k bias is non-zero only on the slow RoPE dims
listed in inject_meta.json; (2) q/v biases equal the source's (or are zero when the source had
none); (3) the checkpoint loads with AutoModelForCausalLM and its next-token argmax agrees with the
source on >= 95% of 512 positions of the diagnostic text. Prints one JSON line; exit code 1 on
failure. Usage: python check_inject.py <src_model> <injected_dir> <diag_text>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def tensors(d: Path, suffix: str) -> dict[str, torch.Tensor]:
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                if k.endswith(suffix):
                    out[k] = h.get_tensor(k).float()
    return out


def main() -> int:
    src, inj, text = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    meta = json.loads((inj / "inject_meta.json").read_text())
    dims = set(meta["dims"])
    cfg = json.loads((inj / "config.json").read_text())
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    ok, notes = True, []
    kb_src, kb_inj = tensors(src, "k_proj.bias"), tensors(inj, "k_proj.bias")
    off_max = 0.0
    for k, v in kb_inj.items():
        d = (v - kb_src.get(k, torch.zeros_like(v))).reshape(-1, hd)
        mask = torch.tensor([j not in dims for j in range(hd)])
        off_max = max(off_max, float(d[:, mask].abs().max()))
    if off_max > 1e-2:
        ok = False
        notes.append(f"k-bias change outside slow dims: {off_max:.4f}")
    for name in ("q_proj.bias", "v_proj.bias"):
        a, b = tensors(src, name), tensors(inj, name)
        diff = max(
            (float((b[k] - a.get(k, torch.zeros_like(b[k]))).abs().max()) for k in b), default=0.0
        )
        if diff > 1e-2:
            ok = False
            notes.append(f"{name} changed: {diff:.4f}")
    tok = AutoTokenizer.from_pretrained(str(inj))
    ids = tok(Path(text).read_text(encoding="utf-8"), return_tensors="pt").input_ids[:, :512]
    with torch.no_grad():
        m0 = AutoModelForCausalLM.from_pretrained(str(src), torch_dtype=torch.bfloat16).eval()
        a0 = m0(ids).logits.argmax(-1)
        del m0
        m1 = AutoModelForCausalLM.from_pretrained(str(inj), torch_dtype=torch.bfloat16).eval()
        kb_loaded = float(m1.model.layers[0].self_attn.k_proj.bias.abs().max())
        a1 = m1(ids).logits.argmax(-1)
    agree = float((a0 == a1).float().mean())
    if agree < 0.95:
        ok = False
        notes.append(f"argmax agreement {agree:.3f} < 0.95")
    if meta["target_r"] > 0 and kb_loaded == 0.0:
        ok = False
        notes.append("loaded model has a zero k bias (bias dropped on load)")
    print(
        json.dumps(
            {
                "inj": str(inj),
                "ok": ok,
                "argmax_agree": round(agree, 4),
                "k_bias_offdim_max": round(off_max, 5),
                "loaded_k_bias_absmax_l0": round(kb_loaded, 4),
                "arch": cfg.get("architectures"),
                "notes": notes,
            }
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
