# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""SpinTare rotary centring with exact score-side add-back and the exemption gate, for vLLM.

Write path (do_kv_cache_update): the post-RoPE key k_p = R(p) k of every new token is cached as
R(p)(k - mu) = k_p - R(p) mu (mu: pre-RoPE key mean of the layer / KV head, KVMR file), unless the
exemption gate exempts it: z = 1 - k.mu / |mu|^2 >= 0.5 (centring would lengthen it; k.mu = k_p .
R(p)mu as RoPE is a rotation), in which case k_p is cached unchanged and the sign bit of its
per-(token, head) K scale is set (the scale is positive otherwise; INT4's zero point in the low
mantissa bits is unaffected). Read path (patched vLLM kernels, MU_ADDBACK): every centred key's
score gets q . R(p) mu back, exactly, in fp32; exempt keys get nothing. So the softmax equals the
uncentred model's up to the quantisation of the centred keys, with or without attention sinks.

Needs vLLM with the SpinTare add-back patch (vllm_patch/), a per-token-head KV cache
(int4 / int8 / fp8_per_token_head), NeoX RoPE over the whole head and no ALiBi.
Enabled by SPINTARE_KVMU=<KVMR file>; SPINTARE_GATE = 1 (default) | centre (no gate, every key
centred) | exempt (every key uncentred: must reproduce the plain cache; test mode).
"""

from __future__ import annotations

import os
import sys

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionLayer, AttentionType
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend, TritonAttentionImpl

from spintare_vllm import kvmr, kvpm

ENV_FILE = "SPINTARE_KVMU"
ENV_GATE = "SPINTARE_GATE"
BACKEND_NAME = "SPINTARE_AB"
GATES = ("1", "centre", "exempt")
INSTANCES = {"n": 0, "writes": 0}
_FILE: kvmr.MuFile | None = None
_FILE_LOADED = False
_TABLES: dict[tuple[str, int], torch.Tensor] = {}


def get_file() -> kvmr.MuFile | None:
    global _FILE, _FILE_LOADED
    if not _FILE_LOADED:
        path = os.environ.get(ENV_FILE)
        _FILE = kvmr.load(path) if path else None
        _FILE_LOADED = True
        if _FILE is not None:
            print(
                f"[spintare] add-back mu {_FILE.path} md5 {_FILE.md5} ({_FILE.n_layer} layers, "
                f"{_FILE.n_head_kv} KV heads, d {_FILE.head_dim}) gate {gate_mode()}",
                file=sys.stderr,
                flush=True,
            )
    return _FILE


def gate_mode() -> str:
    g = os.environ.get(ENV_GATE, "1") or "1"
    if g not in GATES:
        raise ValueError(f"{ENV_GATE} must be one of {GATES}, got {g!r}")
    return g


def _table(device: torch.device, max_pos: int, f: kvmr.MuFile) -> torch.Tensor:
    key = (str(device), max_pos)
    if key not in _TABLES:
        _TABLES[key] = kvmr.rope_table(f.inv_freq, max_pos).to(device)
    return _TABLES[key]


@triton.jit
def _center_kernel(
    key_ptr,
    pos_ptr,
    mu_ptr,
    tab_ptr,
    half_ptr,
    out_ptr,
    ex_ptr,
    s_kn,
    s_kh,
    s_on,
    s_oh,
    num_heads,
    GATE: tl.constexpr,
    HEAD: tl.constexpr,
):
    """W1 (PFX1): the write-path centring in one pass per (token, KV head):
    R(p) mu from the fp32 table row, the gate k . R(p) mu <= |mu|^2 / 2, and
    key_c = k (exempt) or k - R(p) mu; also the exempt flag as int8. GATE 1 =
    the gate, 0 = centre every row, 2 = exempt every row."""
    n = tl.program_id(0)
    h = tl.program_id(1)
    HALF: tl.constexpr = HEAD // 2
    o = tl.arange(0, HALF)
    p = tl.load(pos_ptr + n).to(tl.int64)
    c = tl.load(tab_ptr + p * HEAD + o)
    s = tl.load(tab_ptr + p * HEAD + HALF + o)
    m1 = tl.load(mu_ptr + h * HEAD + o)
    m2 = tl.load(mu_ptr + h * HEAD + HALF + o)
    r1 = m1 * c - m2 * s
    r2 = m2 * c + m1 * s
    kb = key_ptr + n.to(tl.int64) * s_kn + h * s_kh
    k1 = tl.load(kb + o).to(tl.float32)
    k2 = tl.load(kb + HALF + o).to(tl.float32)
    if GATE == 1:
        ex = (tl.sum(k1 * r1, 0) + tl.sum(k2 * r2, 0)) <= tl.load(half_ptr + h)
    elif GATE == 0:
        ex = False
    else:
        ex = True
    ob = out_ptr + n.to(tl.int64) * s_on + h * s_oh
    tl.store(ob + o, tl.where(ex, k1, k1 - r1).to(out_ptr.dtype.element_ty))
    tl.store(ob + HALF + o, tl.where(ex, k2, k2 - r2).to(out_ptr.dtype.element_ty))
    tl.store(ex_ptr + n.to(tl.int64) * num_heads + h, ex.to(tl.int8))


@triton.jit
def _flag_exempt_kernel(
    k_scale_ptr,
    slot_ptr,
    exempt_ptr,
    stride_blk,
    stride_slot,
    stride_head,
    num_kv_heads,
    BLOCK_SIZE: tl.constexpr,
    HEADS_PADDED: tl.constexpr,
):
    """Set the sign bit of the K scale of every exempt (token, KV head) just written."""
    t = tl.program_id(0)
    slot = tl.load(slot_ptr + t).to(tl.int64)
    if slot < 0:
        return
    offs = tl.arange(0, HEADS_PADDED)
    hm = offs < num_kv_heads
    ex = tl.load(exempt_ptr + t * num_kv_heads + offs, mask=hm, other=0)
    ptr = (
        k_scale_ptr
        + (slot // BLOCK_SIZE) * stride_blk
        + (slot % BLOCK_SIZE) * stride_slot
        + offs * stride_head
    )
    s = tl.load(ptr, mask=hm, other=0.0)
    tl.store(ptr, -tl.abs(s), mask=hm & (ex != 0))


class SpinTareAddbackImpl(TritonAttentionImpl):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        INSTANCES["n"] += 1
        self._mf = get_file()
        self._gate = gate_mode()
        self._mu: torch.Tensor | None = None
        self._half_mu2: torch.Tensor | None = None
        self._tab: torch.Tensor | None = None
        self._max_pos = 0
        if self._mf is None:
            return
        if "mu_addback_args" not in TritonAttentionImpl.__dict__:
            # unpatched vLLM never calls the hook: the cache would be centred
            # without the add-back, silently wrong
            raise RuntimeError("add-back needs vLLM with the SpinTare add-back patch")
        if not self._is_per_token_head_quant:
            raise ValueError("add-back needs int4 / int8 / fp8_per_token_head KV caches")
        if self.attn_type != AttentionType.DECODER:
            raise ValueError(f"add-back: attention type {self.attn_type} is not supported")
        if self.alibi_slopes is not None or self.chunk_lookback != -1:
            raise ValueError("add-back: ALiBi / chunked attention are not supported")
        if self._mf.head_dim != self.head_size or self.head_size & (self.head_size - 1):
            raise ValueError(f"add-back: head size {self.head_size} vs file {self._mf.head_dim}")
        from vllm.config import get_current_vllm_config

        self._max_pos = int(get_current_vllm_config().model_config.max_model_len) + 128

    def fused_rope_kvcache_supported(self) -> bool:
        if self._mf is not None:
            return False
        return super().fused_rope_kvcache_supported()

    def _build(self, layer: AttentionLayer, device: torch.device) -> None:
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.model_executor.models.utils import extract_layer_index

        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("add-back: first cache write happened during graph capture")
        mf = self._mf
        assert mf is not None
        li = extract_layer_index(layer.layer_name)  # type: ignore[attr-defined]
        if not 0 <= li < mf.n_layer:
            raise ValueError(f"add-back: layer {li} outside the file ({mf.n_layer} layers)")
        heads = kvpm.heads_for_rank(
            mf.n_head_kv,
            self.num_kv_heads,
            get_tensor_model_parallel_rank(),
            get_tensor_model_parallel_world_size(),
        )
        mu = mf.mu[li, heads].to(device=device, dtype=torch.float32).contiguous()
        self._mu = mu
        self._half_mu2 = 0.5 * (mu.double() ** 2).sum(-1).float()
        self._tab = _table(device, self._max_pos, mf)

    def _positions(self, layer: AttentionLayer, n: int, device: torch.device) -> torch.Tensor:
        """Position of each of the n new tokens, from this layer's attention metadata."""
        from vllm.model_executor.layers.attention.attention import get_attention_context

        md = get_attention_context(layer.layer_name)[0]  # type: ignore[attr-defined]
        t = torch.arange(n, device=device, dtype=torch.int32)
        if md is None:
            return torch.zeros_like(t)
        qsl = md.query_start_loc
        seq = md.seq_lens
        r = torch.searchsorted(qsl[1:].contiguous(), t, right=True)
        r = r.clamp_(max=seq.shape[0] - 1)
        pos = seq[r] - qsl[r + 1] + t
        return pos.clamp_(0, self._max_pos - 1)

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self._mf is None:
            super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
            return
        if self._mu is None:
            self._build(layer, key.device)
        assert self._mu is not None and self._tab is not None and self._half_mu2 is not None
        n = key.shape[0]
        pos = self._positions(layer, n, key.device).long()
        if os.environ.get("SPINTARE_AB_WRITE") == "fused":  # W1 (PFX1 experiment)
            key_c = torch.empty_like(key)
            ex8 = torch.empty(key.shape[:2], dtype=torch.int8, device=key.device)
            _center_kernel[(n, self.num_kv_heads)](
                key,
                pos,
                self._mu,
                self._tab,
                self._half_mu2,
                key_c,
                ex8,
                key.stride(0),
                key.stride(1),
                key_c.stride(0),
                key_c.stride(1),
                self.num_kv_heads,
                GATE={"1": 1, "centre": 0}.get(self._gate, 2),
                HEAD=self.head_size,
                enable_fp_fusion=False,
            )
            super().do_kv_cache_update(layer, key_c, value, kv_cache, slot_mapping)
            self._flag(ex8, slot_mapping, n)
            return
        half = self.head_size // 2
        rows = self._tab[pos]  # [n, D]
        rmu = kvmr.rotate_mu(self._mu, rows[:, :half], rows[:, half:])  # [n, H, D]
        kf = key.float()
        if self._gate == "1":
            exempt = (kf * rmu).sum(-1) <= self._half_mu2  # k.mu <= |mu|^2 / 2
        elif self._gate == "centre":
            exempt = torch.zeros(kf.shape[:2], dtype=torch.bool, device=key.device)
        else:
            exempt = torch.ones(kf.shape[:2], dtype=torch.bool, device=key.device)
        key_c = torch.where(exempt[..., None], kf, kf - rmu).to(key.dtype)
        super().do_kv_cache_update(layer, key_c, value, kv_cache, slot_mapping)
        self._flag(exempt.to(torch.int8).contiguous(), slot_mapping, n)

    def _flag(self, ex8: torch.Tensor, slot_mapping: torch.Tensor, n: int) -> None:
        """mark exempt rows: store -|scale| in the K scale slot of their (token, head)."""
        ks = self._k_scale_cache
        assert ks is not None
        _flag_exempt_kernel[(min(n, slot_mapping.shape[0]),)](
            ks,
            slot_mapping,
            ex8,
            ks.stride(0),
            ks.stride(1),
            ks.stride(2),
            self.num_kv_heads,
            BLOCK_SIZE=ks.shape[1],
            HEADS_PADDED=triton.next_power_of_2(self.num_kv_heads),
        )
        INSTANCES["writes"] += 1

    def mu_addback_args(self, layer: torch.nn.Module, query: torch.Tensor) -> tuple | None:
        if self._mf is None:
            return None
        if self._mu is None:
            raise RuntimeError("add-back: attention ran before the first cache write")
        return (query, self._mu, self._tab, self.scale)


class SpinTareAddbackBackend(TritonAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return BACKEND_NAME

    @staticmethod
    def get_impl_cls() -> type[SpinTareAddbackImpl]:
        return SpinTareAddbackImpl
