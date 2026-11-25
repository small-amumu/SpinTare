# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""Evidence constants for the KV-cache selector.

Every number here comes from an existing measurement (SpinTare runs in results/spintare/) or from
the patch sources in patches/. The SOURCE dict names the file each value was read from; rules that
have no measurement behind them are marked TO_TEST in select.py and carry no constant here.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- locations of the reused code
SPINTARE_REPO = Path(__file__).resolve().parents[1]  # repository root
CONFIG_CHECK_PY = SPINTARE_REPO / "experiments" / "spintare" / "detection_gate" / "config_check.py"
KVPROBE_PY = SPINTARE_REPO / "experiments" / "spintare" / "detection_gate" / "kvprobe_unified.py"

# ---------------------------------------------------------------- SpinTare: configuration check
# tau_b is imported from config_check.py at run time (frozen 2.0280476994784027); TAU_ALLOW is the
# engine-KL allow threshold of `spintare check` (spintare/check/check.py).
TAU_ALLOW = 0.1016
CALIB_TOKENS = 2048  # static-mean / mu calibration window (calib_and_diag/mu_calib.py --T)

# bits per element of the flat formats (llama.cpp block formats; vLLM per-token-head stores one
# fp32 scale per (token, head); the int4 zero point lives in the scale's low mantissa bits)
FLAT_BPW = {
    "q4_0": 4.5,
    "q4_1": 5.0,
    "q8_0": 8.5,
    "f16": 16.0,
}


def pth_bpw(bits: int, head_dim: int, n_meta: int) -> float:
    """vLLM per-token-head bits per element: payload + n_meta fp32 values per (token, head)."""
    return bits + 32.0 * n_meta / head_dim


# kernel coverage by (engine, repair): which HF model_type graphs a patch is wired into, and which
# RoPE scaling types its decode path accepts. Read from the patch sources (patches/):
#   llama.cpp spintare-rotary-addback-exemption-gate.patch touches
#     src/models/{gptneox,qwen2,qwen3}.cpp;
#     its gated decode asserts the kqpos / kadd path, and kqpos is switched off by YaRN-style
#     ext_factor / attn_factor (kv-mu.h, "plain NEOX rope, no YaRN"); rope_mu passes no freq
#     factors.
#   llama.cpp postmean-static-mean.patch touches src/models/{llama,qwen2,qwen2vl,qwen3}.cpp.
#   vLLM plugin (spintare_vllm): attention backend, any model, but calib-mu refuses non-NeoX,
#     partial rotation and attention_scaling != 1 (integrations/vllm, calib-mu); a
#     length-dependent inv_freq
#     (dynamic NTK) or mrope is not representable in its KVMR file either.
KERNEL_MODEL_TYPES: dict[tuple[str, str], frozenset[str] | None] = {
    ("llama.cpp", "spintare"): frozenset({"qwen2", "qwen3", "gpt_neox"}),
    # gpt_neox (Pythia) is wired but its partial rotary never reaches the gated path (kq_bias_ok
    # needs n_rot == head_dim); the partial-RoPE check in select.py refuses it.
    # postmean: HF MistralForCausalLM converts to MODEL_ARCH.LLAMA (llama.cpp conversion/llama.py,
    # ModelBase.register list; checked in the patched llama.cpp tree), i.e. the patched llama
    # graph; the patched
    # qwen2vl graph (M-RoPE) was never run, so it is left out.
    ("llama.cpp", "postmean"): frozenset({"llama", "mistral", "qwen2", "qwen3"}),
    ("vllm", "spintare"): None,  # model-agnostic backend
}
# RoPE scaling types (HF rope_scaling / rope_parameters "rope_type") each add-back kernel handles;
# None / "default" = no scaling. llama.cpp: linear uses freq_scale, which the kqpos path evaluates;
# YaRN turns kqpos into kqbias, which the gate asserts against. vLLM: llama3 / linear only change
# inv_freq, which calib-mu writes into the KVMR file (covered by the F2 engine check, not measured).
# Postmean is a static post-RoPE vector applied only inside its 2K calibration window (rule F4),
# so it has no RoPE-type entry.
KERNEL_ROPE_TYPES: dict[tuple[str, str], frozenset[str | None]] = {
    ("llama.cpp", "spintare"): frozenset({None, "default", "linear"}),
    ("vllm", "spintare"): frozenset({None, "default", "linear", "llama3"}),
}

# decode cost of the repairs (llama.cpp RTX 5090 D; vLLM RTX PRO 5000 throughput ratio vs stock)
COST = {
    "llamacpp_spintare_decode": (0.049, 0.068),  # +4.9 % / +6.8 % per token (Qwen3-8B / 4B)
    "vllm_int4_addback_tput": 0.877,
    "vllm_fp8_addback_tput": 0.841,
}

SOURCE = {
    "tau_b": "experiments/spintare/detection_gate/config_check.py (frozen)",
    "TAU_ALLOW": "spintare/check/check.py",
    "FLAT_BPW": "spintare/check/check.py BPW; serving policy in analysis/kvsel/offline_check.py",
    "KERNEL_*": "patches/llama.cpp/*.patch (diff headers, kv-mu.h asserts); "
                "integrations/vllm (calib-mu)",
    "COST": "results/spintare/decode_overhead/res_ov2_time_agg.json, "
            "results/spintare/vllm_figure/vpd_analysis.txt",
}
