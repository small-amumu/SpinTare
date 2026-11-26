# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Start-up of every `vllm serve` process of the vLLM task cells (run_vllm_cell.sh puts this
directory on PYTHONPATH).

Python imports one sitecustomize per interpreter, so this file does both jobs in every `vllm serve`
process:
1. VLLM_KV_POSTMEAN=<file> (static postmean arm only): install the kv_postmean import hook first, before anything
   imports the vLLM model modules;
2. SPINTARE_SERVE=1: apply SPINTARE_NO_HADACORE and register the SpinTare vLLM plugin.
"""

import os
import sys

if os.environ.get("VLLM_KV_POSTMEAN"):
    import kv_postmean

    kv_postmean.install()
    print(f"[rl-site] postmean hook installed in pid {os.getpid()}", file=sys.stderr, flush=True)

if os.environ.get("SPINTARE_SERVE") == "1":
    if os.environ.get("SPINTARE_NO_HADACORE") == "1":
        import vllm.v1.attention.ops.int4_per_token_head as _i4

        _i4._HADACORE_AVAILABLE = False
        print(
            "[spintare-serve] hadacore disabled (SPINTARE_NO_HADACORE=1)",
            file=sys.stderr,
            flush=True,
        )
    import spintare_vllm

    spintare_vllm.register()
    print(f"[spintare-serve] plugin registered in pid {os.getpid()}", file=sys.stderr, flush=True)
