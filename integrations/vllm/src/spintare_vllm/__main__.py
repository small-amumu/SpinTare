# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""python -m spintare_vllm <command>: check | kld | calib-mu."""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("check", "kld", "calib-mu"):
        sys.exit("usage: python -m spintare_vllm {check|kld|calib-mu} ...")
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "check":
        from spintare_vllm.check import main as run
    elif cmd == "kld":
        from spintare_vllm.kld import main as run  # type: ignore[assignment]
    else:
        from spintare_vllm.calib_mu import main as run  # type: ignore[assignment]
    run(rest)


if __name__ == "__main__":
    main()
