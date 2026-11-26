# Project history: shipped vs. superseded work

This repository ships the final implementation of each component and the exact scripts behind the reported numbers.
Earlier prototypes and exploratory branches remain in the authors' archive and are summarised below, not shipped.

| Line of work | Status | Outcome / rationale |
|---|---|---|
| Static post-RoPE mean ("postmean", KVPM files, `spintare calib`) | kept as the static-mean baseline and as step 2 of `spintare check` | `q . m` cancels in the softmax only for a static mean; RoPE rotates the offset, so the static mean drifts beyond its calibration window (LC1, E2) |
| Score-side add-back: table path (`kqbias`) -> position path (`kqpos`) | `kqpos` shipped | computing `q . R(p) mu` from positions instead of a table lowers the decode cost by 4-43 %; the earlier `kadd` path (rebuild K) stays as the prefill path |
| Alternative FlashAttention add-back kernels (fused, favec, register, two-head packing) | not shipped | slower than the shared table / position kernel |
| Exemption gate SG1 (ggml ops) -> SG2 (fused `GGML_OP_KV_MU_GATE`) | SG2 shipped | SG2 reproduces SG1 exactly and cuts the gate's cost by 20-42 %; the gate still costs decode time |
| vLLM: static-mean import hook -> exact add-back in the Triton kernel | both shipped | the add-back covers models without a key bias (Qwen3) and the sink rows |
| FlashAttention key-bias debias (v6b -> v6d) | not shipped | bias-only (k_proj) path; superseded by the calibrated-mean add-back, which also covers learned offsets |
| Earlier add-back tree (mu3) and its timing run OV1 | not shipped | superseded; the OV2 scripts still name it as a reference arm |
| Predictor candidates (mean bits, worst layer, top-k energy, attention KL, key MSE, the bias rule of vLLM #63081, cosine) | `b_top2` shipped | selected by nested leave-one-family-out cross-validation (ablations Z1-Z9 in `experiments/analysis/spintare/z_analyses.py`) |
| Learned K rotation (krot) and its research llama.cpp tree | not shipped | a side line. The original probes used that tree only for `gguf-py`, which its patch does not touch |
| Jetson Xavier tensor-core decode kernel | not shipped | a side line |
| Comparisons with turboQuantDC kvcheck, KIVI simulations, the vLLM per-token-head census, the Qwen3.5 set | not shipped | side comparisons. The running-mean baseline of the same comparison line is shipped (`long_context_lc1/t2_runmean.py`) |
| HELMET / BABILong runs at 32K | not shipped | dropped |
| Residual-stream propagation model and effective-bit approximation formula | not shipped | dropped |
