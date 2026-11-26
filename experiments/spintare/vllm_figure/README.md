# vLLM figure: accuracy, kernel time and throughput on one RTX PRO 5000

Source of every number in the vLLM figure, measured on 2026-11-23 on one RTX PRO 5000
(vLLM main 5cb0fd265d, Triton attention backend).

| File | Role |
|---|---|
| `vpd_chain.sh` | the run: builds the PR #59674 tree, then output KL (3 models x 6 arms), an eager profiled pass and CUDA-graph throughput (Llama-3.1-8B, 6 arms, 3 repeats) |
| `vpd_analyze.py` | controls and panel values; writes `vpd_result.json` |
| `pf1_trace.py` | GPU kernel time by category from a torch-profiler trace (used by the analyzer) |
| `pr59674_applied.diff` | PR #59674 (vLLM, key-bias handling for INT4 caches) as applied to 5cb0fd265d |
| `test_vpd_analyze.py` | synthetic test of the analyzer (one passing tree, nine broken ones) |

Frozen output: `experiments/results/spintare/vllm_figure/` (per-run JSON files with paths replaced, `vpd_result.json`,
`vpd_analysis.txt`). Controls, all passed:
- every KL arm present and finite; add-back arms built;
- on Llama-3.1-8B (no key bias) the PR arm equals stock INT4 exactly;
- on both Qwen2.5 models the PR arm differs from stock INT4 (the PR is applied);
- every profiled run has one trace with attention kernels;
- every throughput run is present and full length, with equal KV blocks across the repeats of an arm.

The profiler traces (86 MB) are not shipped, so `vpd_analyze.py` cannot be re-run on the frozen output;
`vpd_result.json` is its report from the run (analyzer md5 2e04721d; the copy here differs only in
formatting and its first docstring line).

Panel values (`vpd_analysis.txt`):
- (a) KL to the bf16 cache, INT4 stock / PR #59674 / SpinTare: Qwen2.5-1.5B 7.12 / 0.0141 / 0.0102,
  Qwen2.5-7B 6.71 / 0.0172 / 0.0149, Llama-3.1-8B 0.0168 / 0.0168 / 0.00708; FP8 stock / SpinTare:
  1.74 / 0.00170, 4.86 / 0.00210, 0.00244 / 0.00129.
- (b) attention kernel time (eager, 128 requests 512 / 64), stock / add-back v0 + torch write /
  SpinTare (v1 tf32x3 read + fused write): INT4 630 / 3106 / 1159 ms, FP8 855 / 3179 / 1524 ms.
- (c) throughput relative to stock (CUDA graphs, 1000 requests 512 / 256, median of 3, spread <= 1.3 %):
  add-back v0 0.65 (INT4) / 0.64 (FP8), SpinTare 0.88 / 0.84.

INT4 arms use vLLM's Hadamard path without the hadacore kernel (`SPINTARE_NO_HADACORE=1`), which this
GPU cannot run. An earlier run of the same protocol on an RTX 4090 (with hadacore) gave KL within 6 %
of these values, kernel times of 601 / 2531 / 1135 ms (INT4) and 722 / 2785 / 1265 ms (FP8), and
throughput ratios 0.74 / 0.69 (v0) and 0.90 / 0.90 (SpinTare).
