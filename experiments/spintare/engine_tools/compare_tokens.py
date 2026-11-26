# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Compare the evaluation token ids of energy_probe.py (<out>_tokens.i32) with those stored in a
llama-perplexity --kl-divergence-base file (perplexity.cpp: "_logits_", int32 n_ctx, n_vocab,
n_chunk, then n_ctx * n_chunk token ids; the same reader as md/code/engine_tools.py tokens). Writes
a JSON line with the number of compared and mismatching positions; the probe covers the first chunks
of the engine file. Usage: python compare_tokens.py <base.kld> <probe_tokens.i32> <out.json>
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    base, probe_f, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    with open(base, "rb") as f:
        assert f.read(8) == b"_logits_", "not a llama-perplexity logits file"
        n_ctx, n_vocab, n_chunk = struct.unpack("<3i", f.read(12))
        eng = np.frombuffer(f.read(4 * n_ctx * n_chunk), dtype="<i4")
    prb = np.fromfile(probe_f, dtype="<i4")
    n = min(len(eng), len(prb))
    mism = np.nonzero(eng[:n] != prb[:n])[0]
    res = {
        "n_ctx": n_ctx,
        "n_chunk": n_chunk,
        "compared": int(n),
        "probe_tokens": int(len(prb)),
        "mismatches": int(mism.size),
        "first_mismatch": int(mism[0]) if mism.size else None,
        "match": bool(mism.size == 0 and n == len(prb)),
    }
    out.write_text(json.dumps(res))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
