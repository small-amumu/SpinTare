# Decode overhead of the add-back and the exemption gate

Drivers of the registered timing, accuracy and profiling runs on one RTX 5090 D (batch 1), behind
`fig_decode_cost` and the reported decode-cost numbers.

| File | Run | What |
|---|---|---|
| `ov2_time.sh` | OV2 G4 | `llama-bench` tg128 decode time without add-back and with the table (`kqbias`) or position (`kqpos`) add-back, K4V4 / K4V8, depth 0 / 8192 / 32768, 5 repetitions x 3 rounds (forward / reverse / forward) |
| `ov2_acc.sh` | OV2 G2 / G3 | output KL against the 16-bit cache: the table path against the October add-back tree (bit-identical) and the position path against the table path |
| `sg2_time.sh` | SG2 T1 | the same timing protocol for the position path with and without the fused exemption gate |
| `sg2_acc.sh` | SG2 E1 / E2 | output KL with the gate off and on (decode and prefill graphs); compared with the unfused gate (SG1), whose output it reproduces |
| `sg2_prof.sh`, `sg2b_prof.sh` | SG2 profiles | `nsys` breakdown of the gate's cost (descriptive) |

The engine is the SpinTare tree (`scripts/build_llama_cpp.sh spintare`, `B6` in the scripts). The
OV2 accuracy and timing scripts also read the October add-back tree (`B3`, OV1) as a reference arm; that
tree is not part of this repository. The scripts check the md5 of their original inputs (GGUFs, texts,
binaries) before they run, so a re-run needs the same files or edited checks.

The figure is re-derived from the shipped aggregates (`results/spintare/decode_overhead/*.json`) by
`analysis/spintare/figures/make_figs.py`; the per-round logs are not shipped.
