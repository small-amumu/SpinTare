# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Command-line entry point: python -m spintare <command> [args].

Commands (each prints its own usage with -h, or with no arguments for the positional-argument
tools):
  check           pre-deployment check on llama.cpp
  restore-engine  layer-wise restoration in llama.cpp (LLAMA_KV_F16_LAYERS)
  calib           static post-RoPE key mean -> KVPM file (postmean repair)
  calib-vl        the same for the Qwen2.5-VL text decoder
  permute         reorder a KVPM file to the GGUF order of Llama models
  layout          verify the engine's K channel order against the weights
  offset          offset ratio per layer
  restore         layer-wise restoration (Hugging Face simulation)
  inject          inject a shared key offset into a checkpoint
  inject-verify   verify an injected checkpoint
  probe           baseline features from one 16-bit forward
  rules           baseline decision rules on the probe features
  attn-kl-vllm    simulated attention KL baseline (vLLM int4 format)
  longbench       LongBench subtasks through llama-server
  paired-ci       paired bootstrap verdict on LongBench results
"""

from __future__ import annotations

import runpy
import sys

COMMANDS = {
    "check": "spintare.check.check",
    "restore-engine": "spintare.check.restore_engine",
    "calib": "spintare.fix.calib",
    "calib-vl": "spintare.fix.calib_vl",
    "permute": "spintare.fix.permute",
    "layout": "spintare.fix.layout",
    "offset": "spintare.measure.offset",
    "restore": "spintare.measure.restore",
    "inject": "spintare.measure.inject",
    "inject-verify": "spintare.measure.inject_verify",
    "probe": "spintare.baselines.probe",
    "rules": "spintare.baselines.rules",
    "attn-kl-vllm": "spintare.baselines.attn_kl_vllm",
    "longbench": "spintare.tasks.longbench",
    "paired-ci": "spintare.tasks.paired_ci",
}


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    sys.argv = [f"spintare {cmd}"] + sys.argv[2:]
    runpy.run_module(COMMANDS[cmd], run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
