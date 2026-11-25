# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Exact attention and text windows for c23_kl.py.

Copied from the simulation harness the registered C23 runs imported (pkv5_q.attend with name None,
cq_lab3.text_ids and cq_lab3.DATA); only the exact-attention path that C23 uses is kept, with the
same
query-row chunking and the same operations, so the numbers do not change.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

DATA = os.path.join(os.environ.get("KVREL_DATA", "work/data"), "")


def attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    li: int,
    name: str | None,
    scale: float,
    kw: dict[str, Any],
) -> torch.Tensor:
    """Exact causal attention. q [H, Lq, d] (the last Lq positions), k, v [KVH, Lk, d] float32
    -> output [H, Lq, dv]. `li` is unused; `name` must be None (no quantized base)."""
    if name is not None:
        raise ValueError("only exact attention (name None) is supported here")
    h_q, lq, _ = q.shape
    kvh, lk = k.shape[:2]
    g = h_q // kvh
    kx, vx = (t.repeat_interleave(g, 0) for t in (k, v))
    out = torch.empty(h_q, lq, v.shape[-1], device=q.device)
    kpos = torch.arange(lk, device=q.device)[None, :]
    softcap, win_ = kw.get("softcap"), kw.get("sliding_window")
    rows = int(os.environ.get("PKV5_ROWS", "512"))  # query-row chunk, memory only
    for r0 in range(0, lq, rows):
        r1 = min(lq, r0 + rows)
        qpos = torch.arange(lk - lq + r0, lk - lq + r1, device=q.device)[:, None]
        al = kpos <= qpos
        if win_:
            al &= kpos > qpos - int(win_)
        s = torch.einsum("hqd,hkd->hqk", q[:, r0:r1], kx) * scale
        if softcap:
            s = torch.tanh(s / softcap) * softcap
        p = torch.softmax(s.masked_fill(~al, float("-inf")), -1)
        out[:, r0:r1] = torch.einsum("hqk,hkd->hqd", p, vx)
    return out


def text_ids(tok: Any, path: str, start: int, n: int, dev: torch.device | str) -> torch.Tensor:
    """n token ids of a text from token `start`, with BOS first when the tokenizer adds one."""
    ids = tok(
        Path(path).read_text(encoding="utf-8", errors="replace"),
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]
    b = tok.bos_token_id
    bos = b is not None and tok("x", add_special_tokens=True).input_ids[:1] == [b]
    w = ids[start : start + n - (1 if bos else 0)]
    if bos:
        w = torch.cat([torch.tensor([b]), w])
    return w.unsqueeze(0).to(dev)
