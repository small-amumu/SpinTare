# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""Static post-RoPE key mean files (KVPM, also the older KVMU magic of the vLLM hook).

Layout (little endian): 4-byte magic (b"KVPM" or b"KVMU"), int32 version (1), n_layer, n_head_kv,
head_dim, then float32 m[n_layer][n_head_kv][head_dim]. Channels are in the Hugging Face order,
which is also vLLM's order (no GGUF permutation). Written by `spintare calib` (HF, post-RoPE).
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import torch

MAGICS = (b"KVPM", b"KVMU")
HEADER = struct.Struct("<4i")


@dataclass(frozen=True)
class MeanTable:
    """m[layer, kv_head, dim] in float32 on the CPU, with the md5 of the file it came from."""

    data: torch.Tensor
    md5: str
    path: str

    @property
    def n_layer(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_head_kv(self) -> int:
        return int(self.data.shape[1])

    @property
    def head_dim(self) -> int:
        return int(self.data.shape[2])


def load(path: str | Path) -> MeanTable:
    p = Path(path)
    raw = p.read_bytes()
    if raw[:4] not in MAGICS:
        raise ValueError(f"{p}: bad magic {raw[:4]!r}, expected one of {MAGICS}")
    version, n_layer, n_head_kv, head_dim = HEADER.unpack(raw[4:20])
    if version != 1:
        raise ValueError(f"{p}: unsupported version {version}")
    n = n_layer * n_head_kv * head_dim
    if len(raw) != 20 + 4 * n:
        raise ValueError(f"{p}: {len(raw) - 20} payload bytes, expected {4 * n}")
    data = torch.frombuffer(bytearray(raw[20:]), dtype=torch.float32).view(
        n_layer, n_head_kv, head_dim
    )
    if not torch.isfinite(data).all():
        raise ValueError(f"{p}: non-finite values")
    return MeanTable(data=data.clone(), md5=hashlib.md5(raw).hexdigest(), path=str(p))


def write(path: str | Path, data: torch.Tensor, magic: bytes = b"KVPM") -> None:
    if magic not in MAGICS or data.dim() != 3:
        raise ValueError("data must be [n_layer, n_head_kv, head_dim] and magic KVPM / KVMU")
    d = data.detach().float().cpu().contiguous()
    with Path(path).open("wb") as f:
        f.write(magic)
        f.write(HEADER.pack(1, *d.shape))
        f.write(d.numpy().astype("<f4").tobytes())


def heads_for_rank(n_total: int, n_local: int, tp_rank: int, tp_size: int) -> list[int]:
    """Global KV-head indices held by one tensor-parallel rank, following vLLM's split.

    vLLM splits KV heads evenly when n_total >= tp_size and replicates them otherwise
    (each rank holds one head, shared by tp_size / n_total consecutive ranks).
    """
    if tp_size == 1 or n_total == n_local:
        if n_total != n_local:
            raise ValueError(f"table has {n_total} KV heads, the layer has {n_local}")
        return list(range(n_total))
    if n_total >= tp_size:
        if n_total % tp_size or n_total // tp_size != n_local:
            raise ValueError(
                f"cannot split {n_total} KV heads over {tp_size} ranks into {n_local} each"
            )
        return list(range(tp_rank * n_local, (tp_rank + 1) * n_local))
    if n_local != 1 or tp_size % n_total:
        raise ValueError(f"cannot replicate {n_total} KV heads over {tp_size} ranks")
    return [tp_rank * n_total // tp_size]
