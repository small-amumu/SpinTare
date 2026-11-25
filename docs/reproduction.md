# Reproduction: every result and figure

Conventions: `. scripts/env.sh` first. **Re-derive** = runs on a CPU from the shipped frozen results in
`results/spintare/` (figures go next to the script, or to `FIG_OUT`). **Run** = needs models and a GPU;
the script is the one behind the number, with the settings of the registered run. A *cell* is one model
under one K format.

## Re-derive everything that can be re-derived

```bash
pip install -e ".[figures]"
scripts/stage_results.sh                                  # links results/ under the aggregators' names
python analysis/spintare/z_analyses.py                    # configuration check: tau_b, LOFO BA, ablations
for f in make_figs make_figs_ch3 make_figs_col make_fig_length_kl_col make_fig_row_bars_col \
         make_figs_eval_bars make_ch5_figs make_ch5_bars; do
    FIG_OUT=work/figures python analysis/spintare/figures/$f.py
done
```

`make_ch5_figs.py` / `make_ch5_bars.py` regenerate `ch5_numbers.json` and the generated tables
(`tab5_*.tex`, `tab5_*.md`) from the frozen results.

## Core results

| Result | Run | Re-derive (script -> frozen input) | GPU |
|---|---|---|---|
| Output KL along 32K, five models that need centring | `experiments/spintare/long_context_lc1/run_lc1.sh` (`lc1_long.py`) | `make_fig_length_kl_col.py` <- `long_context_lc1/res_lc1_agg.json` | RTX 5090 D |
| Mechanism illustration | drawn (`docs/figures/fig_mech_wide.png`) | - | - |
| LongBench collapse across engines | `tasks/lc_lb.py` (llama.cpp), `tasks/lc_lb_vllm.py` (vLLM); intervals `calib_and_diag/paired_ci.py` | - | - |
| Effective bits per layer, 28 models | `detection_gate/kvprobe_unified.py` | `make_figs_col.py` / `make_figs_ch3.py` <- `figure_inputs/layers.csv` (`export_layers.py` <- `detection_gate/results/*/probe.json`) | - |
| Ordinary rows vs the sink row | `analysis/spintare/figures/dump_keys.py` | `make_fig_row_bars_col.py` <- `figure_inputs/keys3d/*.npz` | - |
| Configuration check and serving | `detection_gate/`, `patches/llama.cpp/`, `integrations/vllm/` | - | - |
| LongBench 8K, llama.cpp | `tasks/run_llamacpp_cell.sh longbench <arm>` | - | RTX 5090 D |
| RULER 32K, llama.cpp | `tasks/run_llamacpp_cell.sh ruler <arm>` | - | RTX 5090 D |
| RULER 32K, vLLM | `tasks/run_vllm_cell.sh ruler <arm>` | - | RTX PRO 5000 |
| GSM8K, MATH-500, R1-Distill-Qwen-7B | `tasks/run_llamacpp_cell.sh gsm8k\|math500 <arm>` | per-item records in `reasoning_r1/` | RTX PRO 5000 |
| KL by position, 12 models | LC1 | `make_ch5_figs.py` <- `res_lc1_agg.json` | RTX 5090 D |
| Each repair step at equal bits | LC1 | `make_ch5_bars.py` <- `res_lc1_agg.json` | RTX 5090 D |
| Accuracy vs KV memory per generation | reasoning cells | `make_ch5_figs.py` <- `reasoning_r1/` | RTX PRO 5000 |
| Configuration check numbers, bias rule | `detection_gate/`, `prospective/` | `analysis/spintare/z_analyses.py` | - |

## Supporting results

| Result | Run | Re-derive | GPU |
|---|---|---|---|
| Every natural census cell | `detection_gate/` probes + LongBench labels (`tasks/lc_lb.py`, `paired_ci.py`) | cell data in `detection_gate/gp_score.json` | - |
| 8-bit keys; loop suppression | `tasks/lc_lb.py` (`--sampling` for the penalties), `paired_ci.py` | - | - |
| Collapsed cache on 919 GSM8K items | `tasks/lc_reason.py --start 400 --n 919` | - | - |
| Configuration check vs baselines; score ablation | `detection_gate/kvprobe_unified.py`, `spintare/baselines/`, `prospective/*` | `z_analyses.py` <- `detection_gate/`, `prospective/` | - |
| LongBench 8K, vLLM | `tasks/run_vllm_cell.sh longbench <arm>` | - | RTX PRO 5000 |
| 32K output KL, 12 models, incl. running mean | LC1 + `long_context_lc1/run_t2.sh` | 32K bucket of `res_lc1_agg.json`, `res_t2_agg.json` | RTX 5090 D |
| KL inside / beyond the calibration window, R1-Distill-Qwen-7B | `tasks/kl_by_context.sh` | - | RTX PRO 5000 |
| LongBench with / without the gate | `tasks/run_llamacpp_cell.sh longbench spintare\|nogate` | - | RTX 5090 D |
| KL by context length in llama.cpp | `long_context_llamacpp/run_all.sh` | `make_ch5_figs.py` <- `long_context_llamacpp/e2_final.json` (`e2_analyze.py` <- `run_logs/`); V100 replication in `v100_replication/` | RTX PRO 5000 |
| Residual energy rho: prediction vs measurement | LC1 | `make_ch5_bars.py` <- `res_lc1_agg.json` | RTX 5090 D |
| Exemption gate vs no gate / first row only | `exemption_gate_c23/run_c23.sh` | `make_ch5_figs.py` <- `exemption_gate_c23/res_c23_agg.json` | RTX PRO 5000 |
| RULER 32K with / without the gate | `tasks/run_llamacpp_cell.sh ruler spintare\|nogate` | - | RTX 5090 D |
| Where long reasoning fails | reasoning cells | `make_ch5_figs.py` <- `reasoning_r1/` | RTX PRO 5000 |
| vLLM output KL, kernel time, throughput | `vllm_figure/vpd_chain.sh` (`integrations/vllm`: `kld`, `pf_bench`) | `make_figs_eval_bars.py` <- `vllm_figure/vpd_result.json` | RTX PRO 5000 |
| Left-over offset after a static mean | LC1 | `make_figs.py` <- `res_lc1_agg.json` | RTX 5090 D |
| Full layer bits | as the effective-bits figure | `make_figs_ch3.py` | - |
| Decode slowdown of the add-back and the gate | `decode_overhead/ov2_time.sh`, `sg2_time.sh` | `make_figs.py` <- `decode_overhead/*.json` | RTX 5090 D |
| Further ablations | LC1, C23 | `res_lc1_agg.json`, `res_c23_agg.json` | - |
| Exemption gate in simulation (a) and engine (b) | LC1; `decode_overhead/sg2_acc.sh` | `make_figs.py` <- `res_lc1_agg.json`, `decode_overhead/res_sg1_acc_agg.json` | RTX 5090 D |

Paths in the tables are relative to `experiments/spintare/`, `results/spintare/` and
`analysis/spintare/figures/`. "-" in the GPU column: CPU only, or several cards of the census.

## Notes

- Mean files: the SpinTare cells used the pre-RoPE mean of the first 2048 tokens of `data/long_text.txt`
  (`calib_and_diag/mu_calib.py`); the static-mean cells used the post-RoPE mean of the same window
  (`calib_and_diag/pm_calib.py`, or `python -m spintare calib` for the KVPM header).
- Intervals: every LongBench interval comes from `calib_and_diag/paired_ci.py` (one bootstrap index shared
  by the tasks). `python -m spintare paired-ci` resamples each task separately; its bounds differ by up to
  0.3 points with the same point estimates and verdicts.
- Script headers cite the records of the experiment registry (`REGISTRATION_*.md`, `REG_*`, amendment
  and review ids such as `A2` or `r31 S-5`) and, in a few places, driver scripts of the original queues
  (`ov1_time.sh`, `n1_host.sh`, `run_n1_probe.sh`). The registry and those queue drivers are not part of
  this repository; `REGISTRATION_LC1.md` is included as an example. `prospective/n1/n1r_repair.sh` is the
  record of a repair step that ran inside the N1 queue and does not run on its own.
- Raw per-run files (server logs, per-item records of the task tables other than reasoning, reference
  log-probabilities) are not shipped; every shipped aggregate was produced from them by the scripts named
  here. Model weights, GGUFs and folded checkpoints are not shipped either.
