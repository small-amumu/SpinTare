# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""The X5 run imported a byte-identical copy of spintare/ggml_ref.py (md5 c40c8eb1) from this
folder; in the repository this module re-exports the library one."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spintare.ggml_ref import gguf_perm, hadamard, n_rot_k, quant  # noqa: E402, F401
