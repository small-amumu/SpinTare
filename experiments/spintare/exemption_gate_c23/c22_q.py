# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Q4_0 / Q8_0 round trips with llama.cpp's exact semantics, and a check against the real ggml
library through ctypes (used by c23_kl.py, which compares q4x / q8x with ggml bit for bit on three
layers of every run).

Reference C (ggml-quants.c):
  quantize_row_q4_0_ref: per block of 32, max = the element with the largest |x| (first one on
  ties), d = max / -8,
    id = d ? 1.0f / d : 0, stored d = fp16(d), code = MIN(15, (int8_t)(x * id + 8.5f)); dequantize:
    (code - 8) * fp32(d16).
  quantize_row_q8_0_ref: amax = max |x|, d = amax / 127, id = d ? 1.0f / d : 0, stored d = fp16(d),
  code = roundf(x * id);
    dequantize: code * fp32(d16).
All arithmetic is fp32, as in C. `blk` other than 32 (q4 only) is the same rule on a shorter block
and has no ggml counterpart.

"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

QK = 32
# libggml-base of a llama.cpp build (experiments/scripts/build_llama_cpp.sh); only its reference quantizers are
# called
LIB_DEFAULT = os.environ.get(
    "KVREL_GGML_LIB",
    os.path.join(
        os.environ.get("KVREL_LLAMA_CPP", "work/llama.cpp"), "build", "bin", "libggml-base.so"
    ),
)


def q4x(x: torch.Tensor, blk: int = QK) -> torch.Tensor:
    """Exact Q4_0 round trip of the rows of x (n, d); returns fp32."""
    n, d = x.shape
    assert d % blk == 0
    b = x.float().reshape(n, d // blk, blk)
    maxv = b.gather(-1, b.abs().argmax(-1, keepdim=True))  # first occurrence on ties, as the C loop
    dd = maxv / -8.0
    idd = torch.where(dd != 0, 1.0 / dd, torch.zeros_like(dd))
    code = torch.clamp(
        torch.trunc(b * idd + 8.5), max=15.0
    )  # (int8_t) truncates; the argument is >= 0.5
    return ((code - 8.0) * dd.half().float()).reshape(n, d)


def q8x(x: torch.Tensor) -> torch.Tensor:
    """Exact Q8_0 round trip of the rows of x (n, d); returns fp32."""
    n, d = x.shape
    assert d % QK == 0
    b = x.float().reshape(n, d // QK, QK)
    dd = b.abs().amax(-1, keepdim=True) / 127.0
    idd = torch.where(dd != 0, 1.0 / dd, torch.zeros_like(dd))
    x0 = b * idd
    r = torch.trunc(x0)
    code = (
        r + torch.sign(x0) * ((x0 - r).abs() >= 0.5) + 0.0
    )  # roundf: halves away from zero; the int8
    # code has no -0, so -0.0 is turned into +0.0
    return (code * dd.half().float()).reshape(n, d)


class Ggml:
    """Round trips through the reference quantizers of a built libggml-base."""

    def __init__(self, lib: str = LIB_DEFAULT) -> None:
        self.path = str(Path(lib).resolve())
        self.md5 = hashlib.md5(Path(self.path).read_bytes()).hexdigest()
        self.so = ctypes.CDLL(self.path)
        fp, vp, i64 = ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int64
        for nm, at in (
            ("quantize_row_q4_0_ref", (fp, vp, i64)),
            ("dequantize_row_q4_0", (vp, fp, i64)),
            ("quantize_row_q8_0_ref", (fp, vp, i64)),
            ("dequantize_row_q8_0", (vp, fp, i64)),
        ):
            f = getattr(self.so, nm)
            f.argtypes, f.restype = list(at), None

    def _rt(self, x: torch.Tensor, kind: str, bsz: int) -> torch.Tensor:
        a = np.ascontiguousarray(x.float().numpy().reshape(-1))
        k = a.size
        assert k % QK == 0
        buf = ctypes.create_string_buffer(k // QK * bsz)
        out = np.empty(k, dtype=np.float32)
        fp = ctypes.POINTER(ctypes.c_float)
        getattr(self.so, f"quantize_row_{kind}_ref")(a.ctypes.data_as(fp), buf, k)
        getattr(self.so, f"dequantize_row_{kind}")(buf, out.ctypes.data_as(fp), k)
        return torch.from_numpy(out).reshape(x.shape)

    def q4(self, x: torch.Tensor) -> torch.Tensor:
        return self._rt(x, "q4_0", 2 + QK // 2)

    def q8(self, x: torch.Tensor) -> torch.Tensor:
        return self._rt(x, "q8_0", 2 + QK)


def same(a: torch.Tensor, b: torch.Tensor) -> int:
    """Number of elements whose fp32 bit patterns differ."""
    return int((a.contiguous().view(torch.int32) != b.contiguous().view(torch.int32)).sum())


def assert_engine(g: Ggml, xs: dict[str, torch.Tensor]) -> dict:
    """Bit-exact comparison of q4x / q8x with ggml on each tensor of xs; raises on any mismatch."""
    n = 0
    for nm, x in xs.items():
        for kind, mine, ref in (("q4_0", q4x(x), g.q4(x)), ("q8_0", q8x(x), g.q8(x))):
            bad = same(mine, ref)
            if bad:
                raise SystemExit(
                    f"quantizer mismatch against ggml: {nm} {kind} {bad} of {x.numel()} values"
                )
        n += x.numel()
    return {"values_checked": n, "mismatch": 0}
