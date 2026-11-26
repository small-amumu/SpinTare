# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""SpinTare for vLLM: pre-deployment check and postmean repair for low-bit KV caches.

Installed as a vLLM general plugin (entry point group `vllm.general_plugins`): `register()` adds two
attention backends. `SPINTARE`: static post-RoPE mean (postmean) subtracted at the cache write, on
when SPINTARE_KVPM points at a KVPM file. `SPINTARE_AB`: pre-RoPE mean with exact score-side
add-back and the exemption gate (needs the vLLM add-back patch), on when SPINTARE_KVMU points at a
KVMR file. Without their files both are the stock Triton backend.
"""

from __future__ import annotations

__all__ = ["register"]


def register() -> None:
    """vLLM general-plugin entry point. Safe to call more than once per process."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    from spintare_vllm import addback, backend

    for name, path in (
        (backend.BACKEND_NAME, "spintare_vllm.backend.SpinTareTritonBackend"),
        (addback.BACKEND_NAME, "spintare_vllm.addback.SpinTareAddbackBackend"),
    ):
        if name not in AttentionBackendEnum._member_map_:
            register_backend(name, path)
