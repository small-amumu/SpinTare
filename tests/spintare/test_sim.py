# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""spintare.sim quantisers against gguf-py's reference quantise / dequantise; Engine round trip."""

import numpy as np
import pytest
import torch

from spintare.sim import Engine, gguf_perm, quant


@pytest.mark.parametrize("fmt", ["q4_0", "q4_1", "q8_0"])
def test_quant_matches_gguf_py(fmt: str) -> None:
    gguf = pytest.importorskip("gguf")
    from gguf.quants import dequantize, quantize

    rng = np.random.default_rng(0)
    x = rng.standard_normal((512, 128)).astype(np.float32) * 3.0
    x += 40.0 * (rng.random((1, 128)) > 0.9)  # a shared offset on some channels
    x[0, :32] = 0.0  # an all-zero block
    t = getattr(gguf.GGMLQuantizationType, fmt.upper())
    ref = dequantize(quantize(x, t), t).reshape(x.shape)
    ours = quant(torch.from_numpy(x), fmt).numpy()
    assert np.array_equal(ref, ours)


@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("order", ["hf", "gguf"])
def test_engine_round_trip(d: int, order: str) -> None:
    e = Engine(d, order, "hadamard", torch.device("cpu"))
    k = torch.randn(3, 5, d)
    assert torch.allclose(e.from_engine(e.to_engine(k)), k, atol=1e-5)


def test_gguf_perm() -> None:
    assert gguf_perm(8).tolist() == [0, 4, 1, 5, 2, 6, 3, 7]
