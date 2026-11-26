# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""SmoothAttention (QServe) for the K cache, folded into the checkpoint (REG_LCD A3 / REG_RLV A6).

1. Calibration (16-bit HF forward, the B3 static-mean setting: long_text, first T = 2048 tokens): per layer, KV head
   and channel, a_hj = max_t |K_thj| of the pre-RoPE key (Qwen2: k_proj output incl. bias; Qwen3: k_norm output);
   lambda = a ** alpha (alpha = 0.5); RoPE pairs tied (rotate_half layout: channels j and j + d/2 take the max of the
   two), so the scaling commutes with RoPE.
   Qwen3 normalises q / k per head with ONE weight vector of size d shared by all heads, so a per-KV-head lambda
   cannot be folded without a runtime op: there lambda_j = max over KV heads (one shared vector per layer).
2. Fold (no runtime op): K / lambda, Q * lambda (query heads of a GQA group take their KV head's lambda), computed in
   fp32 from the stored tensors, then rounded to the stored dtype.
   Qwen2.5: k_proj weight rows and bias / lambda, q_proj weight rows and bias * lambda.
   Qwen3: k_norm.weight / lambda, q_norm.weight * lambda.
   The folded checkpoint is the original directory with only those tensors rewritten (all other tensors, names,
   shard layout and every non-weight file copied unchanged).
3. Gate (16-bit, prompts: 4 windows of 1024 tokens of diag_text, disjoint from the calibration text):
   (i)  math: original vs in-memory fold, both fp32, no rounding: mean KL <= 1e-6 and top-1 agreement >= 0.999;
   (ii) files: original bf16 vs the folded directory loaded from disk in bf16: mean KL <= max(3 x floor, 1e-4),
        floor = mean KL(original fp32 || original bf16), the bf16 rounding noise of the unfolded model.
   Max |logit difference| is reported for both. Exit 0 only if both pass.
Usage: python sa_fold.py --model DIR --out DIR --calib long_text.txt --gate-text diag_text.txt --result JSON
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def tokens(tok, path: str, n: int, offset: int = 0) -> torch.Tensor:
    ids = tok(Path(path).read_text(encoding="utf-8"), add_special_tokens=False).input_ids
    if len(ids) < offset + n:
        raise SystemExit(f"{path}: {len(ids)} tokens, need {offset + n}")
    return torch.tensor(ids[offset: offset + n])


def is_qwen3(model) -> bool:
    return hasattr(model.model.layers[0].self_attn, "k_norm")


@torch.no_grad()
def calibrate(model, ids: torch.Tensor, alpha: float) -> torch.Tensor:
    """lambda [n_layer, n_kv, d] (Qwen3: identical across KV heads)."""
    cfg = model.config
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    amax: dict[int, torch.Tensor] = {}
    q3 = is_qwen3(model)
    hooks = []
    for li, layer in enumerate(model.model.layers):
        mod = layer.self_attn.k_norm if q3 else layer.self_attn.k_proj

        def hook(_m, _inp, out, li=li):
            k = out.float().reshape(-1, nkv, d)  # [tokens, kv, d] (Qwen3 out is [B, L, kv, d])
            a = k.abs().amax(0)
            amax[li] = a if li not in amax else torch.maximum(amax[li], a)

        hooks.append(mod.register_forward_hook(hook))
    model(input_ids=ids.unsqueeze(0).to(model.device), use_cache=False)
    for h in hooks:
        h.remove()
    if len(amax) != len(model.model.layers):
        raise SystemExit("calibration hook did not run on every layer")
    a = torch.stack([amax[i] for i in range(len(model.model.layers))]).cpu()  # [L, kv, d]
    lam = a.clamp(min=1e-6) ** alpha
    half = d // 2
    tied = torch.maximum(lam[..., :half], lam[..., half:])
    lam = torch.cat([tied, tied], -1)
    if q3:
        lam = lam.amax(1, keepdim=True).expand_as(lam).contiguous()
    return lam


def expected_count(lam: torch.Tensor, q3: bool) -> int:
    return lam.shape[0] * (2 if q3 else 4)


def write_folded(src: Path, dst: Path, lam: torch.Tensor, cfg, q3: bool) -> dict[str, int]:
    if dst.exists():
        raise SystemExit(f"{dst} exists; refusing to overwrite")
    tmp = dst.with_name(dst.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    final, dst = dst, tmp
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy2(f, dst / f.name)
    changed = 0
    for shard in sorted(src.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as fh:
            meta = fh.metadata() or {}
            ts = {k: fh.get_tensor(k) for k in fh.keys()}
        new = fold_tensors_subset(ts, lam, cfg, q3)
        changed += len(new)
        ts.update(new)
        save_file(ts, str(dst / shard.name), metadata=meta or {"format": "pt"})
    if changed != expected_count(lam, q3):
        raise SystemExit(f"rewrote {changed} tensors, expected {expected_count(lam, q3)}")
    ver = verify_folded(src, dst, lam, cfg, q3)
    dst.rename(final)
    return {"tensors_rewritten": changed, **ver}


def verify_folded(src: Path, dst: Path, lam: torch.Tensor, cfg, q3: bool) -> dict[str, int]:
    """Every written tensor: target -> equals the fold of the source and differs from it; other -> bit-identical."""
    same = target = 0
    for shard in sorted(src.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as fa, safe_open(str(dst / shard.name), framework="pt") as fb:
            if set(fa.keys()) != set(fb.keys()) or (fa.metadata() or {}) != (fb.metadata() or {"format": "pt"}) and fa.metadata():
                raise SystemExit(f"{shard.name}: keys or metadata differ")
            orig = {k: fa.get_tensor(k) for k in fa.keys()}
            want = fold_tensors_subset(orig, lam, cfg, q3)
            for k in fa.keys():
                b = fb.get_tensor(k)
                if k in want:
                    if b.dtype != orig[k].dtype or not torch.equal(b, want[k]) or torch.equal(b, orig[k]):
                        raise SystemExit(f"{k}: folded tensor wrong or unchanged")
                    target += 1
                else:
                    if b.dtype != orig[k].dtype or not torch.equal(b, orig[k]):
                        raise SystemExit(f"{k}: non-target tensor changed")
                    same += 1
    if target != expected_count(lam, q3):
        raise SystemExit(f"verified {target} folded tensors, expected {expected_count(lam, q3)}")
    return {"verified_folded": target, "verified_identical": same}


def fold_tensors_subset(ts: dict[str, torch.Tensor], lam: torch.Tensor, cfg, q3: bool) -> dict[str, torch.Tensor]:
    """The rewritten q / k tensors (fp32 math, then the stored dtype) among those present in ts (one shard, or the
    in-memory parameters); q and k of a layer may sit in different shards."""
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    nkv, nq = cfg.num_key_value_heads, cfg.num_attention_heads
    g = nq // nkv
    out: dict[str, torch.Tensor] = {}
    for li in range(lam.shape[0]):
        p = f"model.layers.{li}.self_attn."
        lk = lam[li].float()
        lq = lk.repeat_interleave(g, 0)
        if q3:
            pairs = (("k_norm.weight", 1.0 / lk[0], False), ("q_norm.weight", lq[0], False))
        else:
            pairs = (("k_proj.weight", 1.0 / lk, True), ("k_proj.bias", 1.0 / lk, False),
                     ("q_proj.weight", lq, True), ("q_proj.bias", lq, False))
        for nm, s, is_w in pairs:
            if p + nm not in ts:
                continue
            t = ts[p + nm]
            s = s.to(t.device)
            if is_w:
                out[p + nm] = (t.float().view(-1, d, t.shape[1]) * s[..., None]).view(t.shape).to(t.dtype)
            elif q3:
                out[p + nm] = (t.float() * s).to(t.dtype)
            else:
                out[p + nm] = (t.float().view(-1, d) * s).view(t.shape).to(t.dtype)
    return out


@torch.no_grad()
def logits(model, prompts: list[torch.Tensor]) -> list[torch.Tensor]:
    return [model(input_ids=p.unsqueeze(0).to(model.device), use_cache=False).logits[0].float().cpu() for p in prompts]


def compare(ref: list[torch.Tensor], test: list[torch.Tensor]) -> dict[str, float]:
    kl, agree, mx, n = 0.0, 0.0, 0.0, 0
    for a, b in zip(ref, test):
        la, lb = a.log_softmax(-1), b.log_softmax(-1)
        kl += float((la.exp() * (la - lb)).sum(-1).sum())
        agree += float((a.argmax(-1) == b.argmax(-1)).float().sum())
        mx = max(mx, float((a - b).abs().max()))
        n += a.shape[0]
    return {"kl_mean": kl / n, "top1_agree": agree / n, "max_abs_logit_diff": mx, "tokens": n}


def load(path: str, dtype: torch.dtype, device: str):
    # no device_map: it needs accelerate, which this environment lacks (REG_LCD A3 note); host RAM holds the fp32 7B
    m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device)
    m.eval()
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--gate-text", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--gate-windows", type=int, default=4)
    ap.add_argument("--gate-len", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    src, dst = Path(a.model), Path(a.out)
    cfg = AutoConfig.from_pretrained(a.model)
    if dst.exists():
        raise SystemExit(f"{dst} exists; refusing to overwrite")
    d0 = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    rp = getattr(cfg, "rope_parameters", None) or {}
    prf = getattr(cfg, "partial_rotary_factor", None) or (rp.get("partial_rotary_factor") if isinstance(rp, dict) else None) or 1.0
    if d0 % 2 or prf != 1.0:
        raise SystemExit(f"pair tying needs full rotary over an even head_dim (d {d0}, partial {prf})")
    tok = AutoTokenizer.from_pretrained(a.model)
    calib_ids = tokens(tok, a.calib, a.T)
    prompts = [tokens(tok, a.gate_text, a.gate_len, i * a.gate_len) for i in range(a.gate_windows)]
    res: dict = {"model": a.model, "out": a.out, "alpha": a.alpha, "T": a.T, "calib_md5": md5(Path(a.calib)),
                 "gate_text_md5": md5(Path(a.gate_text)), "script_md5": md5(Path(__file__))}

    m16 = load(a.model, torch.bfloat16, a.device)
    q3 = is_qwen3(m16)
    kp = m16.model.layers[0].self_attn.k_proj
    if kp.out_features != cfg.num_key_value_heads * d0:
        raise SystemExit(f"k_proj out {kp.out_features} != n_kv x head_dim {cfg.num_key_value_heads} x {d0}")
    lam = calibrate(m16, calib_ids, a.alpha)
    lam_path = dst.parent / f"{dst.name}.lambda.pt"
    lam_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lambda": lam, "alpha": a.alpha, "T": a.T, "shared_over_kv_heads": q3, "model": a.model}, lam_path)
    res.update(qwen3_shared_lambda=q3, lambda_file=str(lam_path), lambda_md5=md5(lam_path),
               lambda_min=float(lam.min()), lambda_max=float(lam.max()), lambda_shape=list(lam.shape))
    ref16 = logits(m16, prompts)
    del m16
    torch.cuda.empty_cache()

    m32 = load(a.model, torch.float32, a.device)
    ref32 = logits(m32, prompts)
    sd = dict(m32.named_parameters())
    new = fold_tensors_subset({k: v.data for k, v in sd.items()}, lam, cfg, q3)
    if len(new) != expected_count(lam, q3) or any(torch.equal(new[k], sd[k].data) for k in new):
        raise SystemExit(f"in-memory fold: {len(new)} tensors (expected {expected_count(lam, q3)}) or some unchanged")
    for k, v in new.items():
        sd[k].data.copy_(v)
    fold32 = logits(m32, prompts)
    del m32, sd, new
    torch.cuda.empty_cache()

    res["write"] = write_folded(src, dst, lam, cfg, q3)
    f16 = load(a.out, torch.bfloat16, a.device)
    fold16 = logits(f16, prompts)
    del f16
    torch.cuda.empty_cache()

    g1 = compare(ref32, fold32)
    floor = compare(ref32, ref16)
    g2 = compare(ref16, fold16)
    lim2 = max(3 * floor["kl_mean"], 1e-4)
    res["gate"] = {"i_math_fp32": g1, "i_pass": g1["kl_mean"] <= 1e-6 and g1["top1_agree"] >= 0.999,
                   "bf16_floor_fp32_vs_bf16": floor, "ii_files_bf16": g2, "ii_limit": lim2,
                   "ii_pass": g2["kl_mean"] <= lim2}
    res["gate"]["pass"] = res["gate"]["i_pass"] and res["gate"]["ii_pass"]
    res["out_files_md5"] = {f.name: md5(f) for f in sorted(dst.glob("*.safetensors"))}
    Path(a.result).write_text(json.dumps(res, indent=1))
    print("SAFOLD", json.dumps({k: res[k] for k in ("model", "lambda_md5", "qwen3_shared_lambda")}),
          json.dumps(res["gate"]), flush=True)
    return 0 if res["gate"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
