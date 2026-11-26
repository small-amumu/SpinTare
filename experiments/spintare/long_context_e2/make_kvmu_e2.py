# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""E2: write the m = 0 and random-constant KVMU files with the layout of an existing postmean file.

rc: per layer and kv head a Gaussian direction (seed 0) scaled to the L2 norm of m for that layer
and head. Usage: python make_kvmu_e2.py <m.kvmu> <zero.kvmu> <rc.kvmu>
"""

import struct
import sys

import torch


def main() -> None:
    src, zero_out, rc_out = sys.argv[1:4]
    raw = open(src, "rb").read()
    assert raw[:4] == b"KVMU"
    version, n_layer, n_head, head_dim = struct.unpack("<4i", raw[4:20])
    m = torch.frombuffer(bytearray(raw[20:]), dtype=torch.float32).view(n_layer, n_head, head_dim)
    g = torch.Generator().manual_seed(0)
    d = torch.randn(n_layer, n_head, head_dim, generator=g)
    rc = d / d.norm(dim=-1, keepdim=True) * m.norm(dim=-1, keepdim=True)
    for path, t in ((zero_out, torch.zeros_like(m)), (rc_out, rc)):
        with open(path, "wb") as f:
            f.write(raw[:20])
            f.write(t.contiguous().numpy().astype("<f4").tobytes())
    print(
        "m norm mean", float(m.norm(dim=-1).mean()), "rc norm mean", float(rc.norm(dim=-1).mean())
    )


if __name__ == "__main__":
    main()
