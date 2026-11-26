# LC1 registration: static mean-centring of post-RoPE keys at long context

Written 2026-11-12 (UTC), before any LC1 output on real text exists. SpinTare line (then called TareKV): the claim under test is one link of the
storyline "mean-centring is treated as neutral; it is not".

## Question

Published repairs of the shared key offset subtract a static per-channel mean of the cached (post-RoPE) keys. The
offset is constant before RoPE, so after RoPE it is R(t) mu: it turns with the position, at each rotary pair's own
frequency. A static mean is the average of R(t) mu over the calibration positions. Two things can follow:
(a) pairs that turn fast average to nearly zero, so their part of the offset is never removed;
(b) pairs that turn slowly are removed inside the calibration range, and come back once the position has turned
    them away from the calibrated direction.
LC1 measures which of the two happens, whether it is predicted by the rotary frequencies alone, and whether it
shows in the model's output as the context grows.

At 4K (C27, same quantizer, no rotation) the static mean was 0.97-1.16x the rotary form's KL in nine models (1.61x
in OLMo), with calibration and test both on positions 0..4095. So the 4K data do not show a failure of the static
mean; the question is open for longer contexts and for a calibration shorter than the test.

## Code and protocol

`lc1_long.py` (new, self-contained): harness of `c27_kl.py` (one teacher-forced pass per arm, every layer's keys
replaced, values exact) at L = 32768, with the key formats of `eg1_engine.reconstruct` (Q4_0 on 32-blocks, with
and without the engine's Hadamard rotation, channel order per model as in the EG1 probes). bfloat16 weights,
float32 attention; one RTX 5090 D; torch 2.13.0, transformers 5.18.0 (the versions of the other SpinTare runs),
weights verified file by file against the copies those runs used.
Texts: qasper and mfqa from token 0; each is the other's calibration text. Calibration statistics: pre-RoPE mean
mu and post-RoPE mean over positions [0, 2048) (B2, D, E), post-RoPE mean over [0, L) (BL).
Arms: A (none), B2 (static, 2K calibration), BL (static, calibration as long as the test), D (rotary), E (rotary,
rows that centring lengthens coded uncentred), each in `nr` (no rotation) and `ro` (rotation); X (exact keys through
the arm path, must give KL exactly 0).
KL is read in 512-position buckets ending at 1536, 4096, 8192, 16384 and 32768.
Models: the ten of C27 (Llama-3.1-8B, Mistral-7B-v0.3, OLMo-2-7B, Qwen2.5-1.5B, Qwen3-8B, Qwen2.5-7B, Granite-3.1-8B,
Llama-3.2-3B, Yi-1.5-6B, deepseek-llm-7b) plus the collapsing DeepSeek-R1-Distill-Qwen-7B and Qwen3-0.6B.
"In range" = max_position_embeddings >= 32768 (from the config). OLMo-2, Yi-1.5-6B and deepseek-llm-7b are trained
to 4096 positions: they are run and reported, and every verdict is given for all models and for in-range models.

## Readings (fixed now; frozen aggregation `lc1_long.py agg`; a model's value = mean over the two texts)

- L1 mechanism. rho = offset energy left after subtracting the B2 mean, as a fraction of the bucket's own offset
  energy; median over layers and KV heads. Per model: "recurs" if rho <= 0.5 in the first bucket (inside the
  calibration range) and rho in the last bucket >= 1.5x the first; "never removed" if rho > 0.5 in the first bucket.
  Verdict RECURS / NEVER REMOVED if at least 75% of the models fall in that class, else MIXED.
- L1 fit. The prediction of rho from the calibration text and the positions alone (no test keys) orders the
  measured rho over all (layer, head, bucket): Spearman >= 0.8 in at least 75% of the models. The closed form over
  the rotary pairs is reported next to it.
- L2 output, per format. Among models that need centring (KL of A >= 2x KL of D in the first bucket):
  g(b) = KL(B2) / KL(D) in bucket b; RECURS if g(last) >= 1.5 g(first) in at least 75% of them, NO if it fails in at
  least 75%, else MIXED; fewer than 3 such models: UNDECIDABLE.
- L3 "a longer calibration does not fix it", per format, same models: KL(BL) / KL(D) >= 1.5 in the last bucket in at
  least 75%.
- L4 sink exemption at length (descriptive): KL(E) / KL(D) per bucket and format; KL(D) / KL(A) and KL(B2) / KL(A) per
  bucket (is centring still worse than nothing in the Llamas at 32K, with and without rotation).
- Checks: arm X exactly 0; the exact pass reproducible bit for bit; complete JSON, exit code 0.

## Expectations (mine; not registered claims)

L1: MIXED or NEVER REMOVED rather than RECURS, because RoPE's frequencies are geometric and about a third of the
pairs complete a turn within 2048 positions; the share of the offset energy on those pairs decides it and is not
known. L2: uncertain; in Qwen2.5 the collapse is so large that any partial removal may suffice. L3: likely holds
where L2 holds. L4: the exemption gain persists; whether it persists under rotation is unknown (C27 had no rotation).

## What this cannot show

Teacher forcing; one context (up to 32K; 64K and 128K need longer texts than the two available); values exact;
emulated formats (EG1: emulation and engine agree, Spearman 0.997, on 4-chunk 2K runs, not at 32K); the static mean
is ours (mean over a calibration window of another text), not a reimplementation of any published method's
calibration; KL, not task scores.

## Presentation rule (stated before the data)

All models go into the full table. A summary figure may show a subset chosen by a rule stated in advance
(e.g. the models that need centring and are in range); no result is dropped because of its direction.

## A1 (2026-11-12, after a code review, before any real-text output)

Review: no blocker. The copied format code was compared line by line with eg1_engine / kvprobe_unified, the arms
with c27_kl.py, and the rotary algebra and the closed form were re-derived; all found correct. Adopted:
- **De-rotation check restored (MAJOR).** c27_kl.py checks its rotary tables against the pre-RoPE keys of the model;
  lc1 had dropped it and runs bfloat16 weights, where the model applies the rotation in bfloat16 and the tables are
  ideal. The check is back (layer 0, relative error, overall and per bucket, stored in the JSON; the run stops above
  0.05). Because a bfloat16 rotation error would count against the rotary arms D and E at large positions, three
  small models (Qwen3-0.6B, Qwen2.5-1.5B, Llama-3.2-3B) are also run with float32 weights as a control; those runs
  are reported next to the bfloat16 ones and are not aggregated.
- **Spread (MAJOR).** The verdicts act on point estimates. The aggregate now also gives every ratio per text, the
  median KL per bucket next to the mean, and the within-bucket rank correlation of predicted and measured rho (the
  pooled correlation of the L1 fit can be carried by the common trend over buckets). The verdict rules are unchanged;
  a verdict whose two texts disagree in direction is reported as such. The 75% rule on three to five models is not a
  statistical test.
- The aggregation lists the twelve planned models and names the missing ones (a crashed run writes no JSON).
- Wording: arm X returns the exact keys without the arithmetic of the other arms, so "KL exactly 0" checks
  determinism and the equality of the attention code between passes, nothing more. The quantizer runs on the GPU and
  is not checked against ggml bit for bit as c27 was. The closed form rho_cf assumes the static mean is exactly
  a∘mu (no content term); it enters no verdict.
Noted, unchanged: position 0 is inside both calibration windows (as in c27); with every key quantized the error of
earlier positions propagates into later buckets, so a growth of KL(B2) / KL(D) with position mixes the direct effect
with accumulated error (D carries the same accumulation, which is why the ratio to D is used); the selection of
"models that need centring" and g(first) share KL(D) of the first bucket.

## A2 (2026-11-12, after a second code review, before any real-text output)

- **Granite logits (confirmed in the model config and in transformers 5.18).** The script takes the logits as
  lm_head(hidden states), which skips the division by `config.logits_scaling` (16 for Granite-3.1-8B) that
  GraniteForCausalLM applies. The division is now applied (1 for every other model). c27_kl.py used the full forward
  and was not affected.
- A1 said more than the aggregation prints. What holds: the aggregate gives, per text, g(first), g(last), BL/D and E/D
  in the last bucket, rho of B2 and the within-bucket rank correlation. Per-text medians (`kl_med`), per-text D/A and
  B2/A, and the float32 control runs are in the result files only; I read them from there for the Outcome, and a
  disagreement of the two texts in the direction of a verdict is stated there by hand. A model whose directory is
  missing is logged in the queue log (rc=97) and listed under `missing`.
- torch.median takes the lower middle value of the 512 rows of a bucket; kl_med enters no verdict.

## A3 (2026-11-12, correction of the Question section, before any real-text output)

The sentence "the question is open for longer contexts" is wrong. Earlier results exist and LC1 does not replace them:
- E2 (2026-11-07, llama.cpp on V100; outline v6 sections 3.5 and 5.4): static mean calibrated on 2048 tokens,
  output KL at the longest length over the 2K value: R1-Distill-Qwen-7B 224x (to 64K), Qwen2.5-7B 5.0x, Qwen3-0.6B
  1.9x (to 32K); exact add-back 2.2x, 1.02x, 0.85x. Llama-3.1-8B: static mean 1.5-1.6x worse than no repair at every
  length up to 128K.
- C13b (line C, 10 models): the energy the static mean leaves is 1.2-4.1x the residual of the rotation-aware form.
- RoPE panel of 2026-11-02: the offset sits on slowly turning pairs (8 of 9 families); static mean KL 1.06-1.95x the
  per-position form in 13 of 13 models, with the gap in sink models carried mostly by key 0.
What LC1 adds: twelve models instead of three, the left-over offset energy per position bucket and its prediction
from the rotary frequencies alone, a calibration as long as the test, and the same arms with the engine rotation.
Readings and verdict rules are unchanged.
