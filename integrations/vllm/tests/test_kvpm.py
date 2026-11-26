# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the mean-file reader and the tensor-parallel head split (no vLLM needed)."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
import torch

from spintare_vllm import kvpm


def test_roundtrip(tmp_path: Path) -> None:
    data = torch.randn(3, 4, 8)
    for magic in kvpm.MAGICS:
        p = tmp_path / f"m_{magic.decode()}.bin"
        kvpm.write(p, data, magic=magic)
        t = kvpm.load(p)
        assert torch.equal(t.data, data)
        assert (t.n_layer, t.n_head_kv, t.head_dim) == (3, 4, 8)
        assert len(t.md5) == 32


def test_rejects_bad_files(tmp_path: Path) -> None:
    p = tmp_path / "bad.bin"
    p.write_bytes(b"XXXX" + struct.pack("<4i", 1, 1, 1, 2) + b"\0" * 8)
    with pytest.raises(ValueError, match="magic"):
        kvpm.load(p)
    p.write_bytes(b"KVPM" + struct.pack("<4i", 1, 1, 1, 2) + b"\0" * 4)
    with pytest.raises(ValueError, match="payload"):
        kvpm.load(p)
    p.write_bytes(b"KVPM" + struct.pack("<4i", 2, 1, 1, 2) + b"\0" * 8)
    with pytest.raises(ValueError, match="version"):
        kvpm.load(p)
    kvpm.write(p, torch.tensor([[[float("nan"), 0.0]]]))
    with pytest.raises(ValueError, match="non-finite"):
        kvpm.load(p)


@pytest.mark.parametrize(
    ("n_total", "n_local", "tp", "expect"),
    [
        (8, 8, 1, [[0, 1, 2, 3, 4, 5, 6, 7]]),
        (8, 4, 2, [[0, 1, 2, 3], [4, 5, 6, 7]]),
        (8, 2, 4, [[0, 1], [2, 3], [4, 5], [6, 7]]),
        (2, 1, 4, [[0], [0], [1], [1]]),
        (4, 1, 8, [[0], [0], [1], [1], [2], [2], [3], [3]]),
    ],
)
def test_heads_for_rank(n_total: int, n_local: int, tp: int, expect: list[list[int]]) -> None:
    assert [kvpm.heads_for_rank(n_total, n_local, r, tp) for r in range(tp)] == expect


def test_heads_for_rank_mismatch() -> None:
    with pytest.raises(ValueError):
        kvpm.heads_for_rank(8, 4, 0, 1)
    with pytest.raises(ValueError):
        kvpm.heads_for_rank(8, 3, 0, 2)
