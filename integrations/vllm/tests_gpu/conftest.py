# SPDX-License-Identifier: Apache-2.0
"""GPU test setup: optional machine workaround for drivers that cannot JIT vLLM's hadacore PTX."""

from __future__ import annotations

import os


def pytest_configure(config) -> None:  # type: ignore[no-untyped-def]  # noqa: ARG001
    if os.environ.get("SPINTARE_NO_HADACORE") == "1":
        import vllm.v1.attention.ops.int4_per_token_head as i4

        i4._HADACORE_AVAILABLE = False
