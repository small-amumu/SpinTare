# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Reference re-implementations of llama.cpp's K-cache numerics (one place for every caller).

Engine-exact functions (used by the `spintare` tools):
  hadamard, n_rot_k, gguf_perm  the engine's K rotation and the GGUF q/k channel permutation
  quant, steps                  ggml quantise-dequantise and block steps with the fp16 scale the block stores

Registered variants, kept bit-for-bit as the frozen experiments computed them (do not "fix" them:
the registered numbers depend on them):
  quant_f32scale, steps_f32scale  the detection probe (kvprobe_unified.py) and LC1 (lc1_long.py): the
                                  float32 block scale (no fp16 rounding); q8_0 rounds with floor(|u|
                                  + 0.5)
  q4_0_div                        the ME simulation (me_sim.py): q4_0 with x / d instead of x * (1 / d)

Every function works on float32 tensors and on the last dimension in blocks of 32.
"""

from __future__ import annotations

import math

import torch


def hadamard(n: int, device: torch.device) -> torch.Tensor:
    """Orthonormal Sylvester Walsh-Hadamard matrix of size n (float32), as llama_kv_cache builds
    it."""
    assert n & (n - 1) == 0, n
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(n)).to(device)


def n_rot_k(head_dim: int) -> int:
    """Size of llama.cpp's K rotation: 64, doubled while it divides the head dim; 0 when 64 does not
    divide it (upstream rule in src/llama-kv-cache.cpp, llama.cpp commit 3bb63cb)."""
    if head_dim % 64:
        return 0
    n = 64
    while head_dim % (2 * n) == 0:
        n *= 2
    return n


def gguf_perm(d: int) -> torch.Tensor:
    """Channel order of a Llama-family GGUF head in terms of the Hugging Face order (int64, CPU)."""
    half = d // 2
    return torch.tensor([(e % 2) * half + e // 2 for e in range(d)])


def _inv(d: torch.Tensor) -> torch.Tensor:
    return torch.where(
        d == 0, torch.zeros_like(d), 1.0 / torch.where(d == 0, torch.ones_like(d), d)
    )


def quant(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """ggml reference quantise-dequantise (float32 in, float32 out). As in ggml, the integer code is
    computed with the float32 scale (and minimum) and dequantised with the fp16 values the block
    stores."""
    b = x.reshape(*x.shape[:-1], -1, 32)
    if fmt == "q4_0":
        vmax = b.gather(-1, b.abs().argmax(-1, keepdim=True))
        d = vmax / -8.0
        y = (torch.clamp(torch.floor(b * _inv(d) + 8.5), 0.0, 15.0) - 8.0) * d.half().float()
    elif fmt == "q4_1":
        mn, mx = b.amin(-1, keepdim=True), b.amax(-1, keepdim=True)
        d = (mx - mn) / 15.0
        q = torch.clamp(torch.floor((b - mn) * _inv(d) + 0.5), 0.0, 15.0)
        y = q * d.half().float() + mn.half().float()
    elif fmt == "q8_0":
        d = b.abs().amax(-1, keepdim=True) / 127.0
        u = b * _inv(d)
        t = torch.floor(u.abs())
        r = t + (u.abs() - t >= 0.5).float()  # roundf: half away from zero, no float32 carry
        y = torch.sign(u) * r * d.half().float()
    else:
        raise ValueError(f"unknown format {fmt}")
    return y.reshape(x.shape)


def steps(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """Per-block quantisation step as the block stores it (fp16), shape [..., n_blocks]."""
    b = x.reshape(*x.shape[:-1], -1, 32)
    if fmt == "q4_0":
        d = b.abs().amax(-1) / 8.0
    elif fmt == "q4_1":
        d = (b.amax(-1) - b.amin(-1)) / 15.0
    else:
        d = b.abs().amax(-1) / 127.0
    return d.half().float()


def quant_f32scale(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """Registered variant (detection probe, LC1): ggml's integer rule with the float32 block scale
    kept for dequantisation; q8_0 rounds with floor(|u| + 0.5)."""
    b = x.reshape(*x.shape[:-1], -1, 32)
    if fmt == "q4_0":
        idx = b.abs().argmax(-1, keepdim=True)
        vmax = b.gather(-1, idx)
        dlt = vmax / -8.0
        q = torch.clamp(torch.floor(b * _inv(dlt) + 8.5), 0.0, 15.0)
        y = (q - 8.0) * dlt
    elif fmt == "q4_1":
        mn, mx = b.amin(-1, keepdim=True), b.amax(-1, keepdim=True)
        dlt = (mx - mn) / 15.0
        q = torch.clamp(torch.floor((b - mn) * _inv(dlt) + 0.5), 0.0, 15.0)
        y = q * dlt + mn
    elif fmt == "q8_0":
        amax = b.abs().amax(-1, keepdim=True)
        dlt = amax / 127.0
        u = b * _inv(dlt)
        y = torch.sign(u) * torch.floor(u.abs() + 0.5) * dlt
    else:
        raise ValueError(fmt)
    return y.reshape(x.shape)


def steps_f32scale(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """Registered variant (detection probe): per-block step with the float32 scale."""
    b = x.reshape(*x.shape[:-1], -1, 32)
    if fmt == "q4_0":
        return b.abs().amax(-1) / 8.0
    if fmt == "q4_1":
        return (b.amax(-1) - b.amin(-1)) / 15.0
    return b.abs().amax(-1) / 127.0


def q4_0_div(x: torch.Tensor) -> torch.Tensor:
    """Registered variant (ME simulation): ggml q4_0 rule computed as floor(x / d + 8.5), float32
    scale."""
    b = x.reshape(*x.shape[:-1], -1, 32)
    mx = torch.gather(b, -1, b.abs().argmax(-1, keepdim=True))
    d = mx / -8.0
    ds = torch.where(d == 0, torch.ones_like(d), d)
    q = torch.clamp(torch.floor(b / ds + 8.5), 0, 15)
    return ((q - 8) * d).reshape(x.shape)
