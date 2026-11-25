# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""Find the engine's K channel order from the GGUF weights: HF order (Qwen2 / Qwen3) or the HF order
permuted by gguf_perm (Llama-family conversions).

Reads blk.<l>.attn_k.weight from the GGUF (dequantised with gguf-py) and
model.layers.<l>.self_attn.k_proj.weight from the HF checkpoint, and compares, head by head, the
GGUF rows with the HF rows in the identity order and in the gguf_perm order (relative Frobenius
error; q8_0 weights give ~1e-2 for the right order). Prints a JSON line with "order": "gguf" (every
head matches the permuted rows, none the identity), "hf" (the reverse) and exits 0; exits 1 when
neither holds for every checked head. Also reports the GGUF tokenizer.ggml.add_bos_token flag
(gguf_add_bos), which decides whether llama-perplexity adds BOS, and tokenizer.ggml.bos_token_id
(gguf_bos_id), the token it adds. Usage: python check_layout.py <gguf> <hf_model_dir> <gguf-py dir>
[layers ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

sys.path.insert(0, sys.argv[3])
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

from permute_kvmu import gguf_perm  # noqa: E402


def hf_weight(model_dir: Path, name: str) -> np.ndarray:
    index = model_dir / "model.safetensors.index.json"
    shard = (
        json.loads(index.read_text())["weight_map"][name] if index.exists() else "model.safetensors"
    )
    with safe_open(str(model_dir / shard), framework="pt") as f:
        return f.get_tensor(name).float().numpy()


def main() -> None:
    gg, hf = sys.argv[1], Path(sys.argv[2])
    cfg = json.loads((hf / "config.json").read_text())
    nl = cfg["num_hidden_layers"]
    layers = [int(x) for x in sys.argv[4:]] or [0, nl // 2 - 1, nl - 1]
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    r = GGUFReader(gg)
    tensors = {t.name: t for t in r.tensors}
    fld = r.fields.get("tokenizer.ggml.add_bos_token")
    gguf_add_bos = bool(fld.parts[fld.data[0]][0]) if fld is not None else None
    fld = r.fields.get("tokenizer.ggml.bos_token_id")
    gguf_bos_id = int(fld.parts[fld.data[0]][0]) if fld is not None else None
    p = gguf_perm(hd)
    rows = []
    for li in layers:
        t = tensors[f"blk.{li}.attn_k.weight"]
        g = dequantize(t.data, t.tensor_type).reshape(n_kv * hd, -1).astype(np.float64)
        w = hf_weight(hf, f"model.layers.{li}.self_attn.k_proj.weight").astype(np.float64)
        assert g.shape == w.shape, (g.shape, w.shape)
        for h in range(n_kv):
            gh, wh = g[h * hd : (h + 1) * hd], w[h * hd : (h + 1) * hd]
            e_id = np.linalg.norm(gh - wh) / np.linalg.norm(wh)
            e_p = np.linalg.norm(gh - wh[p]) / np.linalg.norm(wh)
            rows.append(
                {
                    "layer": li,
                    "head": h,
                    "err_identity": round(float(e_id), 5),
                    "err_perm": round(float(e_p), 5),
                }
            )
    gguf_ok = all(x["err_perm"] < 0.05 and x["err_identity"] > 0.5 for x in rows)
    hf_ok = all(x["err_identity"] < 0.05 and x["err_perm"] > 0.5 for x in rows)
    order = "gguf" if gguf_ok else ("hf" if hf_ok else "none")
    print(
        json.dumps(
            {
                "head_dim": hd,
                "n_kv": n_kv,
                "layers": layers,
                "order": order,
                "perm_matches": gguf_ok,
                "gguf_add_bos": gguf_add_bos,
                "gguf_bos_id": gguf_bos_id,
                "max_err_perm": max(x["err_perm"] for x in rows),
                "min_err_perm": min(x["err_perm"] for x in rows),
                "max_err_identity": max(x["err_identity"] for x in rows),
                "min_err_identity": min(x["err_identity"] for x in rows),
            }
        )
    )
    if order == "none":
        print(json.dumps(rows))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
