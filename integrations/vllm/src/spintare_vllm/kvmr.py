# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Pre-RoPE key mean with the model's RoPE frequencies (KVMR files) and the fp32 RoPE table.

Layout (little endian): b"KVMR", int32 version (1), n_layer, n_head_kv, head_dim, then float32
mu[n_layer][n_head_kv][head_dim] (pre-RoPE key mean, after any k bias / k norm, Hugging Face
channel order = vLLM's) and float32 inv_freq[head_dim / 2] (the model's rotary frequencies after
any frequency scaling; NeoX pairs j, j + head_dim / 2 over the whole head).
Written by `python -m spintare_vllm calib-mu`.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import torch

MAGIC = b"KVMR"
HEADER = struct.Struct("<4i")


@dataclass(frozen=True)
class MuFile:
    mu: torch.Tensor  # [n_layer, n_head_kv, head_dim] float32, CPU
    inv_freq: torch.Tensor  # [head_dim / 2] float32, CPU
    md5: str
    path: str

    @property
    def n_layer(self) -> int:
        return int(self.mu.shape[0])

    @property
    def n_head_kv(self) -> int:
        return int(self.mu.shape[1])

    @property
    def head_dim(self) -> int:
        return int(self.mu.shape[2])


def load(path: str | Path) -> MuFile:
    p = Path(path)
    raw = p.read_bytes()
    if raw[:4] != MAGIC:
        raise ValueError(f"{p}: bad magic {raw[:4]!r}, expected {MAGIC!r}")
    version, n_layer, n_head_kv, head_dim = HEADER.unpack(raw[4:20])
    if version != 1 or head_dim % 2:
        raise ValueError(f"{p}: unsupported version {version} or odd head_dim {head_dim}")
    n_mu = n_layer * n_head_kv * head_dim
    n = n_mu + head_dim // 2
    if len(raw) != 20 + 4 * n:
        raise ValueError(f"{p}: {len(raw) - 20} payload bytes, expected {4 * n}")
    data = torch.frombuffer(bytearray(raw[20:]), dtype=torch.float32)
    if not torch.isfinite(data).all():
        raise ValueError(f"{p}: non-finite values")
    mu = data[:n_mu].view(n_layer, n_head_kv, head_dim).clone()
    inv_freq = data[n_mu:].clone()
    if (inv_freq <= 0).any():
        raise ValueError(f"{p}: non-positive rotary frequency")
    return MuFile(mu=mu, inv_freq=inv_freq, md5=hashlib.md5(raw).hexdigest(), path=str(p))


def write(path: str | Path, mu: torch.Tensor, inv_freq: torch.Tensor) -> None:
    m = mu.detach().float().cpu().contiguous()
    f = inv_freq.detach().float().cpu().contiguous()
    if m.dim() != 3 or f.shape != (m.shape[2] // 2,):
        raise ValueError("mu must be [n_layer, n_head_kv, head_dim], inv_freq [head_dim / 2]")
    with Path(path).open("wb") as fh:
        fh.write(MAGIC)
        fh.write(HEADER.pack(1, *m.shape))
        fh.write(m.numpy().astype("<f4").tobytes())
        fh.write(f.numpy().astype("<f4").tobytes())


def rope_table(inv_freq: torch.Tensor, max_pos: int) -> torch.Tensor:
    """fp32 [max_pos, head_dim] = [cos(p theta_j) | sin(p theta_j)], angles in float64."""
    p = torch.arange(max_pos, dtype=torch.float64)[:, None]
    ang = p * inv_freq.double()[None, :]
    return torch.cat([ang.cos(), ang.sin()], dim=1).float().contiguous()


def rotate_mu(mu: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """R(p) mu for NeoX pairs: mu [..., H, D], cos / sin [..., D/2] broadcast over H."""
    h = mu.shape[-1] // 2
    m1, m2 = mu[..., :h], mu[..., h:]
    c, s = cos.unsqueeze(-2), sin.unsqueeze(-2)
    return torch.cat([m1 * c - m2 * s, m2 * c + m1 * s], dim=-1)
