# SPDX-License-Identifier: Apache-2.0
"""GPU op-level test of the add-back (patched vLLM kernels + SpinTareAddbackImpl write path).

Keys are drawn pre-RoPE as mu + noise (a large shared offset, |mu| ~ 500) with a few sink-like rows
(noise only: the gate exempts them), rotated with the fp32 table, written through the impl's own
do_kv_cache_update (centring, gate, sign flag) and read through impl.forward (patched kernel).
Reference (independent, float64): dequantise the cache (int8 / fp8: q * s; int4: nibbles, zero
point, inverse RHT), add R(p) mu back on the rows whose K scale is negative-free, causal softmax,
dequantised V. The add-back's error vs the reference must stay within 2x the plain kernel's error
vs its own dequantised reference + 1e-3 (P10: the tolerance is set by the unchanged path).
Gate path (amendment A4 of REG_VP2): gate 'exempt' (every row stored uncentred, sign-flagged) must
equal mu = 0 in 'centre' mode (every row stored as is, unflagged) bit for bit: both run the same
compiled add-back kernel on the same stored key bits and a bias of exactly 0, so any difference
comes from the flag / gate path. 'exempt' must also meet the add-back's accuracy bound vs the
float64 reference. The add-back kernel vs the stock kernel (different compiled programs) is only
reported (bit_identical, max_ulps, frac_diff) in the JSON-lines file named by SPINTARE_TEST_REPORT.
Run on a CUDA machine with the add-back-patched vLLM: pytest -q tests_gpu/test_addback_kernel.py
"""

from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE  # noqa: E402
from vllm.v1.attention.backends.triton_attn import (  # noqa: E402
    TritonAttentionImpl,
    TritonAttentionMetadata,
)

from spintare_vllm import addback, kvmr  # noqa: E402

DEV = torch.device("cuda")
REPORT = os.environ.get("SPINTARE_TEST_REPORT")
D = 128
BLOCK = 16
SEGS = 16


def bf16_ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|a - b| in units of the bf16 spacing at max(|a|, |b|) (8-bit significand)."""
    a, b = a.double(), b.double()
    m = torch.maximum(a.abs(), b.abs()).clamp_min(2.0**-126)
    return (a - b).abs() / torch.exp2(torch.floor(torch.log2(m)) - 7)


def compare(name: str, a: torch.Tensor, b: torch.Tensor, **meta: object) -> bool:
    """Within 2 bf16 ulps everywhere; the comparison is recorded in the report either way."""
    u = bf16_ulps(a, b)
    rec = {
        "check": name,
        "bit_identical": bool(torch.equal(a, b)),
        "max_ulps": float(u.max()),
        "frac_diff": float((a != b).double().mean()),
        **meta,
    }
    if REPORT:
        with open(REPORT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    return rec["max_ulps"] <= 2.0


def make_impl(cls, nh: int, nkv: int, kv: str):  # type: ignore[no-untyped-def]
    impl = object.__new__(cls)
    TritonAttentionImpl.__init__(impl, nh, D, 1 / math.sqrt(D), nkv, None, None, kv)
    return impl


def attach_addback(
    impl: TritonAttentionImpl, mu: torch.Tensor, tab: torch.Tensor, gate: str, pos: torch.Tensor
) -> None:
    impl._mf = SimpleNamespace(head_dim=D)
    impl._gate = gate
    impl._mu = mu.float().contiguous()
    impl._half_mu2 = 0.5 * (mu.double() ** 2).sum(-1).float()
    impl._tab = tab
    impl._max_pos = tab.shape[0]
    impl._positions = lambda layer, n, device: pos[:n].to(device)  # noqa: ARG005


def kv_cache(kv: str, n_blocks: int, nkv: int) -> torch.Tensor:
    hs = D // 2 if kv == "int4_per_token_head" else D
    content = 2 * (hs + 4)  # bytes: data + fp32 scale per K / V half (1-byte storage)
    return torch.zeros(
        n_blocks, nkv, BLOCK, content, dtype=STR_DTYPE_TO_TORCH_DTYPE[kv], device=DEV
    )


def metadata(
    q_lens: list[int], seq_lens: list[int], nh: int, use_3d: bool
) -> TritonAttentionMetadata:
    qsl = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32, device=DEV)
    nblk = [-(-s // BLOCK) for s in seq_lens]
    bt = torch.zeros(len(seq_lens), max(nblk), dtype=torch.int32, device=DEV)
    start = 0
    for i, n in enumerate(nblk):
        bt[i, :n] = torch.arange(start, start + n, dtype=torch.int32, device=DEV)
        start += n
    T = sum(q_lens)
    slots = []
    for i, (ql, sl) in enumerate(zip(q_lens, seq_lens, strict=True)):
        for p in range(sl - ql, sl):
            slots.append(int(bt[i, p // BLOCK]) * BLOCK + p % BLOCK)
    return TritonAttentionMetadata(
        num_actual_tokens=T,
        max_query_len=max(q_lens),
        query_start_loc=qsl,
        max_seq_len=max(seq_lens),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEV),
        block_table=bt,
        slot_mapping=torch.tensor(slots, dtype=torch.int64, device=DEV),
        seq_threshold_3D=64 if use_3d else 0,
        num_par_softmax_segments=SEGS,
        softmax_segm_output=torch.empty(T, nh, SEGS, D, dtype=torch.float32, device=DEV),
        softmax_segm_max=torch.empty(T, nh, SEGS, dtype=torch.float32, device=DEV),
        softmax_segm_expsum=torch.empty(T, nh, SEGS, dtype=torch.float32, device=DEV),
        causal=True,
    )


def dequant(
    impl, kvc: torch.Tensor, kv: str, slots: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[no-untyped-def]
    """float64 K, V [n, nkv, D] at the given slots and the raw K scale sign (True = exempt)."""
    kc, vc = impl._pth_key_value_caches(kvc)
    ks, vs = impl._k_scale_cache, impl._v_scale_cache
    b, o = slots // BLOCK, slots % BLOCK
    # the cache views carry the 4 inline fp32-scale bytes after each row's data: keep the data only
    nbytes = D // 2 if kv == "int4_per_token_head" else D
    kraw, vraw = kc[b, o][..., :nbytes], vc[b, o][..., :nbytes]  # [n, nkv, nbytes]
    ksr, vsr = ks[b, o].double(), vs[b, o].double()  # [n, nkv]
    exempt = ksr < 0
    if kv == "int4_per_token_head":
        from vllm.utils.math_utils import largest_power_of_2_divisor
        from vllm.v1.attention.ops.int4_per_token_head import single_rht

        def un4(raw: torch.Tensor, sraw: torch.Tensor) -> torch.Tensor:
            bits = sraw.float().view(torch.int32)
            zp = (bits & 0xF).double()
            sc = (bits & -16).view(torch.float32).double().abs()
            lo, hi = (raw & 0xF).double(), ((raw >> 4) & 0xF).double()
            x = torch.stack([lo, hi], -1).flatten(-2)  # even = low nibble
            xr = (x - zp[..., None]) * sc[..., None]
            b2 = largest_power_of_2_divisor(D)
            return single_rht(xr.float(), inverse=True).double() / b2

        return un4(kraw, ks[b, o]), un4(vraw, vs[b, o]), exempt
    kf, vf = kraw.double(), vraw.double()
    return kf * ksr.abs()[..., None], vf * vsr.abs()[..., None], exempt


def reference(q, K, V, rmu, exempt, q_lens, seq_lens, scale):  # type: ignore[no-untyped-def]
    """float64 causal attention; K gets R(p) mu back on non-exempt rows."""
    Keff = torch.where(exempt[..., None], K, K + rmu)
    out = torch.empty(q.shape, dtype=torch.float64, device=DEV)
    t = k0 = 0
    g = q.shape[1] // K.shape[1]
    for ql, sl in zip(q_lens, seq_lens, strict=True):
        Ks = Keff[k0 : k0 + sl].repeat_interleave(g, 1)
        Vs = V[k0 : k0 + sl].repeat_interleave(g, 1)
        for i in range(ql):
            p = sl - ql + i
            s = torch.einsum("hd,nhd->hn", q[t + i].double(), Ks[: p + 1]) * scale
            out[t + i] = torch.einsum("hn,nhd->hd", s.softmax(-1), Vs[: p + 1])
        t += ql
        k0 += sl
    return out


CASES = [
    # (q_lens, seq_lens, use_3d): decode batch (3D), decode (2D), prefill, chunk with context
    ([1, 1, 1], [37, 300, 1100], True),
    ([1, 1, 1], [37, 300, 1100], False),
    ([600], [600], False),
    ([100, 1], [600, 900], False),
]


@pytest.mark.parametrize("kv", ["int8_per_token_head", "fp8_per_token_head", "int4_per_token_head"])
@pytest.mark.parametrize(("nh", "nkv"), [(28, 4), (32, 8)])
@pytest.mark.parametrize("case", range(len(CASES)))
def test_addback_matches_reference(kv: str, nh: int, nkv: int, case: int) -> None:
    torch.manual_seed(1000 * case + nh)
    q_lens, seq_lens, use_3d = CASES[case]
    inv = (1.0 / (1e6 ** (torch.arange(0, D, 2, dtype=torch.float64) / D))).float()
    tab = kvmr.rope_table(inv, 4096).to(DEV)
    mu = torch.randn(nkv, D, device=DEV) * (500 / math.sqrt(D))
    # every token of every sequence (context rows are written first, as earlier steps did)
    n_all = sum(seq_lens)
    pos_all = torch.cat([torch.arange(s, device=DEV) for s in seq_lens])
    k_pre = mu[None] + torch.randn(n_all, nkv, D, device=DEV) * 3
    sink = torch.rand(n_all, nkv, device=DEV) < 0.05
    k_pre = torch.where(sink[..., None], torch.randn(n_all, nkv, D, device=DEV) * 3, k_pre)
    rows = tab[pos_all]
    h = D // 2
    k_post = kvmr.rotate_mu(k_pre, rows[:, :h], rows[:, h:])  # R(p) k, [n, nkv, D]
    rmu = kvmr.rotate_mu(mu, rows[:, :h], rows[:, h:]).double()  # [n, nkv, D]
    k_post = k_post.to(torch.bfloat16)
    v = torch.randn(n_all, nkv, D, device=DEV).to(torch.bfloat16)
    q = torch.randn(sum(q_lens), nh, D, device=DEV).to(torch.bfloat16)
    n_blocks = sum(-(-s // BLOCK) for s in seq_lens) + 1
    full = metadata(seq_lens, seq_lens, nh, False)  # writes every row
    md = metadata(q_lens, seq_lens, nh, use_3d)
    layer = SimpleNamespace(layer_name="model.layers.0.self_attn.attn")

    outs, refs, errs = {}, {}, {}
    for arm in ("plain", "addback", "exempt", "zero"):
        cls = TritonAttentionImpl if arm == "plain" else addback.SpinTareAddbackImpl
        impl = make_impl(cls, nh, nkv, kv)
        if arm == "zero":
            attach_addback(impl, torch.zeros_like(mu), tab, "centre", pos_all)
        elif arm != "plain":
            attach_addback(impl, mu, tab, "1" if arm == "addback" else "exempt", pos_all)
        kvc = kv_cache(kv, n_blocks, nkv)
        impl.do_kv_cache_update(layer, k_post, v, kvc, full.slot_mapping)
        out = torch.empty(sum(q_lens), nh, D, dtype=torch.bfloat16, device=DEV)
        impl.forward(layer, q, None, None, kvc, md, out)
        K, V, ex = dequant(impl, kvc, kv, full.slot_mapping)
        r = reference(
            q,
            K,
            V,
            rmu if arm in ("addback", "exempt") else torch.zeros_like(rmu),
            ex if arm != "plain" else torch.ones_like(ex),
            q_lens,
            seq_lens,
            impl.scale,
        )
        outs[arm], refs[arm] = out, r
        errs[arm] = (out.double() - r).abs().max().item()
        if arm == "addback":
            n_ex = int(ex.sum())
            assert 0 < n_ex < ex.numel(), "the gate exempted none or all rows"
            assert bool((ex == sink).float().mean() > 0.99), "gate disagrees with the planted sinks"
    meta = {
        "kv": kv,
        "nh": nh,
        "case": case,
        "err_plain": errs["plain"],
        "err_addback": errs["addback"],
        "err_exempt": errs["exempt"],
        "err_zero": errs["zero"],
    }
    compare("exempt_vs_plain", outs["exempt"], outs["plain"], **meta)
    same = torch.equal(outs["exempt"], outs["zero"])
    compare("exempt_vs_zero", outs["exempt"], outs["zero"], **meta)
    assert errs["addback"] <= 2 * errs["plain"] + 1e-3, errs
    assert errs["exempt"] <= 2 * errs["plain"] + 1e-3, errs
    assert same, "gate 'exempt' must equal mu = 0 'centre' bit for bit (same compiled kernel)"


@pytest.mark.parametrize("kv", ["int8_per_token_head", "fp8_per_token_head", "int4_per_token_head"])
@pytest.mark.parametrize("case", range(len(CASES)))
def test_zero_mu_vs_plain_report(kv: str, case: int) -> None:
    """Report only (amendment A4): the add-back kernel with mu = 0 vs the stock kernel."""
    torch.manual_seed(7 + case)
    nh, nkv = 28, 4
    q_lens, seq_lens, use_3d = CASES[case]
    inv = (1.0 / (1e6 ** (torch.arange(0, D, 2, dtype=torch.float64) / D))).float()
    tab = kvmr.rope_table(inv, 2048).to(DEV)
    n_all = sum(seq_lens)
    pos_all = torch.cat([torch.arange(s, device=DEV) for s in seq_lens])
    k = torch.randn(n_all, nkv, D, device=DEV).to(torch.bfloat16)
    v = torch.randn(n_all, nkv, D, device=DEV).to(torch.bfloat16)
    q = torch.randn(sum(q_lens), nh, D, device=DEV).to(torch.bfloat16)
    full = metadata(seq_lens, seq_lens, nh, False)
    md = metadata(q_lens, seq_lens, nh, use_3d)
    layer = SimpleNamespace(layer_name="model.layers.0.self_attn.attn")
    outs = {}
    for arm in ("plain", "zero"):
        cls = TritonAttentionImpl if arm == "plain" else addback.SpinTareAddbackImpl
        impl = make_impl(cls, nh, nkv, kv)
        if arm == "zero":
            attach_addback(impl, torch.zeros(nkv, D, device=DEV), tab, "centre", pos_all)
        kvc = kv_cache(kv, sum(-(-s // BLOCK) for s in seq_lens) + 1, nkv)
        impl.do_kv_cache_update(layer, k, v, kvc, full.slot_mapping)
        out = torch.empty(sum(q_lens), nh, D, dtype=torch.bfloat16, device=DEV)
        impl.forward(layer, q, None, None, kvc, md, out)
        outs[arm] = out
    compare("zero_mu_vs_plain", outs["zero"], outs["plain"], kv=kv, nh=nh, case=case)
    assert (
        torch.isfinite(outs["zero"].float()).all() and torch.isfinite(outs["plain"].float()).all()
    )
