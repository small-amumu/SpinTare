
## 2026-10-26T05:18:26Z PRED-P outcome (scored by pp_rules.py score, f8ca672f, and pp_amend_score.py for amendment 1)
- run: pp_host.sh f50c7b75, 02:33:48Z-05:15:20Z; 40 task cells, all rc 0; decisions appended 03:00:12Z before any task
  run; amendment 1 (natural-only thresholds) appended 04:05:15Z. Unusable labels (f16 < 5 or loops >= 20%): every cell
  of Qwen3-1.7B-Base (f16 1.19, loops 65%) and Qwen3-4B-Base (f16 4.30). Undecidable: Falcon3-3B q4_0 (-2.07
  [-4.00, -0.19]) and q4_0norot (-1.49), granite-3.3-2b q4_0norot (-5.05 [-7.38, -2.75]) and q4_1 (-1.84 [-3.35,
  -0.36]).
- usable decided cells: 20 (9 collapse, all Qwen2 family: Qwen2-0.5B / Qwen2-1.5B / Qwen2.5-Coder-3B x 3 formats,
  drops 14.9-36.3; 11 lossless: Falcon3-3B q4_1, granite q4_0, OLMo-2-1B x 3, SmolLM2-360M x 3, stablelm-2-zephyr x 3).
| rule | collapse allowed | false alarms | BA |
|---|---|---|---|
| P_check (engine) | 0 | 1 (stablelm-zephyr q4_0norot, D 0.178 fall back; lossless -1.23) | 0.955 |
| P_top2 (pooled 2.343) | 0 | 1 (same cell, b_top2 2.200) | 0.955 |
| P_klm (pooled 0.0945) | 2 (Coder-3B q4_0 kl 0.078 / -29.3; q4_1 0.083 / -22.0) | 1 | 0.843 |
| P_top2n (natural 2.028, amendment 1) | 0 | 0 | 1.000 |
| P_klmn (natural 0.0558, amendment 1) | 0 | 2 (stablelm-zephyr and SmolLM2 q4_0norot) | 0.909 |
| family baseline (Qwen2 < 14B refused) | 0 | 0 | 1.000 |
- PP1 holds (P_top2 allows none of the 9 collapse cells). PP2 holds (0.955 vs 0.955). PP3 fails (P_klm 0.843, two
  collapse cells allowed).
- PP4 (amendment 1, prospective cells only: Falcon3-3B, granite, OLMo-2-1B, SmolLM2-360M, Qwen3-4B-Base q4_0norot /
  q4_1): 8 labelled prospective cells, all lossless, 0 collapse; P_top2n 0 false alarms (P_check 0); holds for false
  alarms, UNTESTED for misses.
- the near-threshold cell granite q4_0norot (b_top2 2.163, D 0.076) is undecidable with point estimate -5.05 and an
  upper bound of -2.75: allowed by P_check and P_top2n, refused by P_top2 (pooled). If it is damaged, P_check and P_top2n
  miss it and the pooled threshold catches it; it is not scored and is reported with that reading.
- reading as fixed: on these new checkpoints the worst-two-layer effective bits allowed no collapse cell and produced
  no more false alarms than the engine check (pooled thresholds: equal; natural thresholds: one fewer, the
  stablelm-zephyr q4_0norot cell where the engine check also errs); every damaged cell is in the Qwen2 family, where
  the trivial family rule is equally perfect, so the test does not show that b_top2 beats the family rule. The simulated
  attention KL is not kept as a gate (PP3 fails).
