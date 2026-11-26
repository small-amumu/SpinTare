# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the add-back algebra the patched kernels implement (no vLLM, no GPU).

1. NeoX RoPE: score(q, R(p) k) = score(q, R(p)(k - mu)) + sum_j A_j cos(p t_j) + B_j sin(p t_j).
2. The kernel's angle-addition form (table row p0, offsets i) equals the direct form.
3. The exemption gate on the post-RoPE key equals the pre-RoPE rule k . mu <= |mu|^2 / 2.
4. KVMR round trip and rope_table / rotate_mu against a direct NeoX rotation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from spintare_vllm import kvmr

D = 128
H = D // 2


def neox_rope(x: torch.Tensor, pos: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """HF rotate_half convention: x [..., n, D], pos [n]."""
    ang = pos.double()[:, None] * inv_freq.double()[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1)
    sin = torch.cat([ang.sin(), ang.sin()], -1)
    x1, x2 = x[..., :H], x[..., H:]
    rot = torch.cat([-x2, x1], -1)
    return x * cos + rot * sin


def ab_terms(q: torch.Tensor, mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q1, q2, m1, m2 = q[..., :H], q[..., H:], mu[..., :H], mu[..., H:]
    return q1 * m1 + q2 * m2, q2 * m1 - q1 * m2


@pytest.fixture()
def setup() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    # float32 frequencies, as the model and the KVMR file hold them
    inv_freq = (1.0 / (1e6 ** (torch.arange(0, D, 2, dtype=torch.float64) / D))).float().double()
    n = 300
    pos = torch.randint(0, 32768, (n,), generator=g)
    mu = torch.randn(D, generator=g, dtype=torch.float64) * 50  # large shared offset
    k = mu + torch.randn(n, D, generator=g, dtype=torch.float64)  # keys share mu
    q = torch.randn(D, generator=g, dtype=torch.float64)
    return {"inv_freq": inv_freq, "pos": pos, "mu": mu, "k": k, "q": q}


def test_addback_identity(setup: dict[str, torch.Tensor]) -> None:
    s = setup
    kp = neox_rope(s["k"], s["pos"], s["inv_freq"])
    kc = neox_rope(s["k"] - s["mu"], s["pos"], s["inv_freq"])  # what the cache holds
    a, b = ab_terms(s["q"], s["mu"])
    ang = s["pos"].double()[:, None] * s["inv_freq"][None, :]
    bias = (a[None, :] * ang.cos() + b[None, :] * ang.sin()).sum(-1)
    assert torch.allclose(kp @ s["q"], kc @ s["q"] + bias, rtol=0, atol=1e-8)


def test_angle_addition(setup: dict[str, torch.Tensor]) -> None:
    s = setup
    tile = 32
    tab = kvmr.rope_table(s["inv_freq"].float(), 40000).double()
    a, b = ab_terms(s["q"], s["mu"])
    for p0 in (0, 4096, 32768 - tile):
        c0, s0 = tab[p0, :H], tab[p0, H:]
        a2, b2 = a * c0 + b * s0, b * c0 - a * s0
        bias = tab[:tile, :H] @ a2 + tab[:tile, H:] @ b2
        ang = (p0 + torch.arange(tile, dtype=torch.float64))[:, None] * s["inv_freq"][None, :]
        ref = (a[None, :] * ang.cos() + b[None, :] * ang.sin()).sum(-1)
        # float32 table entries: |error| <= sum_j (|A_j| + |B_j|) * ~3e-7 (a few fp32 ulps)
        assert torch.allclose(bias, ref, rtol=0, atol=1e-6 * (a.abs() + b.abs()).sum().item())


def test_gate_rotation_invariant(setup: dict[str, torch.Tensor]) -> None:
    s = setup
    kp = neox_rope(s["k"], s["pos"], s["inv_freq"])
    tab = kvmr.rope_table(s["inv_freq"].float(), 32768).double()
    rows = tab[s["pos"]]
    # as the plugin calls it: mu [H_kv, D], cos / sin [n, D/2] -> [n, H_kv, D]
    rmu = kvmr.rotate_mu(s["mu"][None, :], rows[:, :H], rows[:, H:])[:, 0]
    post = (kp * rmu).sum(-1)
    pre = s["k"] @ s["mu"]
    assert torch.allclose(post, pre, rtol=1e-6, atol=1e-6 * (s["mu"] @ s["mu"]).item())


def test_rotate_mu_matches_rope(setup: dict[str, torch.Tensor]) -> None:
    s = setup
    tab = kvmr.rope_table(s["inv_freq"].float(), 32768).double()
    rows = tab[s["pos"]]
    got = kvmr.rotate_mu(s["mu"][None, :], rows[:, :H], rows[:, H:])[:, 0]
    ref = neox_rope(s["mu"].expand(len(s["pos"]), D), s["pos"], s["inv_freq"])
    assert torch.allclose(got, ref, rtol=0, atol=1e-5 * s["mu"].norm().item())


def test_kvmr_roundtrip(tmp_path: Path) -> None:
    mu = torch.randn(3, 2, D)
    inv = torch.rand(H) + 0.01
    p = tmp_path / "m.kvmr"
    kvmr.write(p, mu, inv)
    f = kvmr.load(p)
    assert torch.equal(f.mu, mu) and torch.equal(f.inv_freq, inv)
    assert (f.n_layer, f.n_head_kv, f.head_dim) == (3, 2, D)
    p.write_bytes(p.read_bytes()[:-4])
    with pytest.raises(ValueError, match="payload"):
        kvmr.load(p)
