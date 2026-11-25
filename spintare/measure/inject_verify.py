# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Sanity check of an injected checkpoint (`spintare inject`) against its source (CPU, bf16).

Checks: (1) the injected k bias minus the source k bias is non-zero only on the slow RoPE dims
listed in inject_meta.json, and its norm per layer matches the realised_norm_mean recorded by
`spintare inject` (within 0.1%; files without it are checked against bias_norm_mean within 5%);
(2) q/v biases equal the source's (or are zero when the source had none); (3) the checkpoint loads
with AutoModelForCausalLM, keeps its k bias, and its next-token argmax agrees with the source on >=
95% of 512 positions of the diagnostic text. Prints one JSON line; exit code 1 on failure.
Usage: python -m spintare inject-verify <src_model> <injected_dir> <diag_text>
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
    meta = json.loads((inj / "inject_meta.json").read_text(encoding="utf-8"))
    dims = set(meta["dims"])
    cfg = json.loads((inj / "config.json").read_text(encoding="utf-8"))
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    ok, notes = True, []
    kb_inj = tensors(inj, "k_proj.bias")
    # the source bias as `spintare inject` saw it (loaded in bf16)
    kb_src = {k: v.to(torch.bfloat16).float() for k, v in tensors(src, "k_proj.bias").items()}
    off_max, norm_err = 0.0, 0.0
    exact = all("realised_norm_mean" in x for x in meta["per_layer"])
    key, tol = ("realised_norm_mean", 1e-3) if exact else ("bias_norm_mean", 0.05)
    target = {x["layer"]: x[key] for x in meta["per_layer"]}
    sel = (
        set(target) if meta["layers"] == "all" else {int(x) for x in str(meta["layers"]).split(",")}
    )
    if meta["target_r"] > 0 and any(target.get(li, 0.0) <= 0 for li in sel):
        ok = False
        notes.append("a selected layer has no injected offset in inject_meta.json")
    mask = torch.tensor([j not in dims for j in range(hd)])
    for k, v in kb_inj.items():
        d = (v - kb_src.get(k, torch.zeros_like(v))).reshape(-1, hd)
        li = int(k.split("layers.")[1].split(".")[0])
        # untouched layers must not change at all; injected ones only on the slow dims
        off_max = max(off_max, float((d if li not in sel else d[:, mask]).abs().max()))
        if target.get(li, 0.0) > 0:
            norm_err = max(norm_err, abs(float(d.norm(dim=-1).mean()) / target[li] - 1.0))
    if off_max > 0.0:  # exact: untouched bf16 entries are copied unchanged
        ok = False
        notes.append(f"k bias changed outside the injected slow dims: {off_max:.4g}")
    if norm_err > tol:
        ok = False
        notes.append(f"injected offset norm differs from inject_meta.json by {norm_err:.1%}")
    for name in ("q_proj.bias", "v_proj.bias"):
        a = {k: v.to(torch.bfloat16).float() for k, v in tensors(src, name).items()}
        b = tensors(inj, name)
        diff = max(
            (float((b[k] - a.get(k, torch.zeros_like(b[k]))).abs().max()) for k in b), default=0.0
        )
        if diff > 1e-2:
            ok = False
            notes.append(f"{name} changed: {diff:.4f}")
    tok = AutoTokenizer.from_pretrained(str(inj))
    ids = tok(Path(text).read_text(encoding="utf-8"), return_tensors="pt").input_ids[:, :512]
    with torch.no_grad():
        m0 = AutoModelForCausalLM.from_pretrained(str(src), dtype=torch.bfloat16).eval()
        a0 = m0(ids).logits.argmax(-1)
        del m0
        m1 = AutoModelForCausalLM.from_pretrained(str(inj), dtype=torch.bfloat16).eval()
        probe_layer = min(sel) if sel else 0  # a layer that received an offset
        kb = m1.model.layers[probe_layer].self_attn.k_proj.bias
        kb_loaded = 0.0 if kb is None else float(kb.abs().max())
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
                "offset_norm_rel_err": round(norm_err, 4),
                "loaded_k_bias_layer": probe_layer,
                "loaded_k_bias_absmax": round(kb_loaded, 4),
                "arch": cfg.get("architectures"),
                "notes": notes,
            }
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
