
## 2026-10-25T23:04:50Z GATE-PRED a1 and MECH-D outcome (scored by gp_score.py 6efc2f02 and mechd_score.py 7e8a710b)
- probes: 44 probe.json (every model of models_gp.txt; three passes as recorded in run notes 1-5: stablelm2,
  starcoder2_7b and olmo2_7b with BOS overrides, gemma2_2b_inj2 / inj3 from the _ob0 dirs; all layers captured).
  Voids: ChatGLM3-6B (3 cells), GLM-4-9B-eps (1), Llama-3.1-8B-inj32L0 (1), as registered. Outputs in
  <work>/gatepred/ (gp_score.out / .json, mechd_score.out / .json).
- TRAIN-I class x family (probed): Llama-3.1-8B 1 lossless / 19 damaged, Mistral-7B 2 / 10, gemma-2-2b 7 / 0.
  Fitted: b* = 2.662 (equal-BA interval 2.633-2.662; COMMON BA 0.615 at its lower end, middle and upper end alike),
  attention-KL threshold 0.0784, key-MSE threshold 0.00615.
- COMMON (31 cells of TEST-N out-of-sample + TEST-P decided by every predictor; 13 damaged):
| predictor | misclear (collapse) | false alarm | bal. acc. | cost |
|---|---|---|---|---|
| P_check (frozen) | 1 (0) | 3 | 0.878 | 8 |
| P_beff (b_meas >= b*) | 10 (9) | 0 | 0.615 | 50 |
| P_bias (vLLM PR 63081) | 4 (3) | 0 | 0.846 | 20 |
| P_mlx mean band | 0 (0) | 10 | 0.722 | 10 |
| P_cos (>= 0.998) | 0 (0) | 17 | 0.528 | 17 |
| P_akl (fitted) | 1 (0) | 1 | 0.934 | 6 |
| P_kmse (fitted) | 3 (3) | 12 | 0.551 | 27 |
  P_ratio decides 17 of these cells (0 misclear, 4 false alarms). TEST-P alone: P_check, P_bias and P_akl all 21 / 20
  cells right (BA 1.000); P_beff misclears all 9 collapse cells of TEST-P.
- G1' holds (P_check misclears 0 damaged TEST-P cells). G2' fails (P_beff BA 0.615 vs 0.878, 9 collapse cells
  allowed: the threshold fitted on the injected families lies below the b_meas of the natural Qwen2-family collapse
  cells, about 2.85-3.19). G3' fails (n = 3: P_bias allows all three by definition and P_check refuses all three, but
  P_beff allows Qwen3-0.6B q4_0). G4' fails (12 of 17 ordered pairs, 14 needed; P_ratio puts all eight models in its
  fall-back band as predicted).
- McNemar: no comparison reaches the registered 30 paired models (per model n <= 19), so none is a significance claim.
- MECH-D (29 models, 11 collapse): collapse AUC, hypothesised direction (permutation p): attention pattern
  diffuse_head_frac 0.682 (0.056), norm_entropy 0.641, top10_mass 0.677, sink_mass 0.626; key bias has_k_bias 0.871
  (0.0001), PR 63081 statistic 0.937; size 0.571; offset b_meas 0.808 (0.003), b_eff 0.707, registered r_max 0.991
  (n 21, secondary).
  M1 fails: full sample 0.808 vs 0.682 (margin 0.126); b_meas is above every attention feature in all 13 usable
  left-out-family subsets, but the 0.1 margin holds in 9 (69%, 80% needed; the gemma, granite, OLMo and Pythia
  subsets give 0.06-0.10). M2 holds (no attention feature reaches 0.8, full sample or any usable subset; max 0.722).
  M3 fails on one dose: base vs inj3 ... inj16 every attention feature within 5% and a tenth of its range (largest
  change 3.3%), inj32 changes diffuse_head_frac by +13.3% (a count over heads); labels go lossless -> damaged (base 0,
  inj3 partial, inj4 ... inj32 collapse). M4 fails (has_k_bias AUC 0.871 >= 0.8).
- readings as fixed: G2' fails -> the single-forward effective-bit predictor does not replace the engine check; the
  check stays the gate. MECH-D: M1 fails -> both properties carry information and SpinTare does not claim the offset
  explanation predicts better; M2 holds -> no layer-mean attention statistic separates the failing models (AUC <=
  0.72); M3 holds for doses up to 16x and fails at 32x on one coarse feature, so "an approximately softmax-invisible
  offset reproduces the failure" is stated for doses up to inj16 only. Descriptive: the best weights-only and
  measured statistics are the key-bias size (0.937) and the offset ratio (0.991, fewer models), not attention
  sparsity; P_akl (a fitted K-only attention KL) is the best probe predictor (BA 0.934) but needs a quantised
  forward and a threshold chosen here.
