"""b_top2 for vLLM's INT4 per-token-head K cache: kvprobe_unified.py (the census probe) plus one
format.

The configuration check (effective bits, b_top2, frozen tau_b = 2.028) forms the engine's
blocks from the post-RoPE keys of one 16-bit pass. llama.cpp's blocks are 32 wide (q4_0 / q4_1,
optional whole-head Hadamard). vLLM's int4_per_token_head (5cb0fd265d, ops/int4_per_token_head.py)
quantizes each (token, KV head) as ONE block of head_dim after its own randomized Hadamard transform
single_rht (x -> H(x * s), s = the +-1 signs of torch.Generator seed 0x9E3779B9, Bernoulli(0.5)),
asymmetric with step = max((max - min) / 15, 1e-6) and a 4-bit zero point. This script adds that
format, "int4pth", to the probe without changing anything else:
- coordinates: the probe's "identity" basis in HF channel order (vLLM caches HF order), then vLLM's RHT (normalized
  Hadamard; a common scale cancels in sigma / step);
- block: the whole head (d channels); step as above; b_meas, sigma and the attention-KL fields as in the probe;
- quantizer for the attention-KL field: vLLM's rounding (half away from zero, zero point clamped to [0, 15]), then
  the inverse RHT, so the score uses the HF-order query. The scale's low-mantissa zero-point packing
  is ignored.
The "hadamard/int4pth" entry (Hadamard, then RHT) has no engine meaning and is not used. The
llama.cpp formats are recomputed in the same pass, so their b_top2 can be compared with the stored
probe.json of the model (same text, seq, skip): this validates the pass. Usage: python
kvprobe_pth.py --model DIR --order hf --out JSON --text FILE [kvprobe_unified args]
       (kvprobe_unified.py must sit next to this file)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kvprobe_unified as K  # noqa: E402  (registers its attention hook on import)

_SIGNS: dict = {}


def rht_signs(d: int, device: torch.device) -> torch.Tensor:
    """vLLM _get_rht_signs(d, round_idx=0): CPU generator, seed 0x9E3779B9, 2 * Bernoulli(0.5) -
    1."""
    key = (d, str(device))
    if key not in _SIGNS:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0x9E3779B9)
        s = 2.0 * torch.bernoulli(torch.full((d,), 0.5, dtype=torch.float32), generator=gen) - 1.0
        _SIGNS[key] = s.to(device)
    return _SIGNS[key]


def rht(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    return (x * rht_signs(d, x.device)) @ K.hadamard(
        d, x.device
    )  # K.hadamard is symmetric and orthonormal


def irht(y: torch.Tensor) -> torch.Tensor:
    d = y.shape[-1]
    return (y @ K.hadamard(d, y.device)) * rht_signs(d, y.device)


def _round_away(u: torch.Tensor) -> torch.Tensor:
    return torch.where(u >= 0, torch.floor(u + 0.5), torch.ceil(u - 0.5))


# the probe quantizes through the registered float32-scale variants of spintare/ggml_ref.py
_steps, _quant, _block_features = K.steps_f32scale, K.quant_f32scale, K.block_features


def steps(x: torch.Tensor, fmt: str) -> torch.Tensor:
    if fmt != "int4pth":
        return _steps(x, fmt)
    y = rht(x)
    return (
        ((y.amax(-1) - y.amin(-1)) / 15.0).clamp(min=1e-6).unsqueeze(-1)
    )  # one block per (token, head)


def quant(x: torch.Tensor, fmt: str) -> torch.Tensor:
    if fmt != "int4pth":
        return _quant(x, fmt)
    y = rht(x)
    mn, mx = y.amin(-1, keepdim=True), y.amax(-1, keepdim=True)
    sc = ((mx - mn) / 15.0).clamp(min=1e-6)
    zp = _round_away(-mn / sc).clamp(0.0, 15.0)
    q = _round_away(y / sc + zp).clamp(0.0, 15.0)
    return irht((q - zp) * sc)


def block_features(k: torch.Tensor, fmt: str, skip: int) -> dict:
    """k: [kvh, L, d] in probe coordinates; int4pth: RHT, then one block of d per (token, head)."""
    if fmt != "int4pth":
        return _block_features(k, fmt, skip)
    ks = rht(k[:, skip:, :])
    kvh, n, d = ks.shape
    kb = ks.reshape(kvh, n, 1, d)
    mu = kb.mean(1)  # [kvh, 1, d]
    c = kb - mu.unsqueeze(1)
    sigma = c.pow(2).mean(dim=(1, 3)).sqrt()  # [kvh, 1]
    st = steps(k[:, skip:, :], fmt).mean(1)  # [kvh, 1]
    ok = (sigma > 0) & (st > 0)
    bmeas = torch.log2(2.0 * math.sqrt(3.0) * sigma[ok] / st[ok])
    r = mu.abs().amax(-1)[ok] / sigma[ok]
    beff = 3.0 - torch.log2(torch.clamp(r, min=1e-12))
    return {
        "b_meas": float(bmeas.mean()),
        "b_meas_min": float(bmeas.min()),
        "b_eff": float(beff.mean()),
        "b_eff_min": float(beff.min()),
        "r_mean": float(r.mean()),
        "r_max": float(r.max()),
    }


K.steps_f32scale, K.quant_f32scale, K.block_features = steps, quant, block_features
K.FORMATS = K.FORMATS + ("int4pth",)

if __name__ == "__main__":
    if "--order" not in sys.argv or sys.argv[sys.argv.index("--order") + 1] != "hf":
        raise SystemExit("int4pth models vLLM, which caches HF channel order: use --order hf")
    K.main()
