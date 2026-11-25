# Task evaluation in the engines

Clients and runners behind the task results. Each runner starts one server for one K-cache
arm, runs the client against it and stops the server; results land under `$KVREL_WORK/results/`.

| Result | Runner | Client | Engine build | GPU |
|---|---|---|---|---|
| LongBench 8K, llama.cpp | `run_llamacpp_cell.sh longbench <arm> <name>` | `lc_lb.py` (+ `lb.py`) | SpinTare tree; `pm`: postmean tree | RTX 5090 D |
| RULER 32K, llama.cpp; with / without the gate | `run_llamacpp_cell.sh ruler <arm> <name>` | `lc_ruler.py` | same | RTX 5090 D |
| GSM8K, MATH-500; failure causes, accuracy vs memory | `run_llamacpp_cell.sh gsm8k\|math500 <arm> <name>` | `lc_reason.py` | same | RTX PRO 5000 |
| KL inside / beyond the calibration window | `kl_by_context.sh` | `llama-perplexity` | SpinTare and postmean trees | RTX PRO 5000 |
| LongBench with / without the gate | `run_llamacpp_cell.sh longbench spintare\|nogate <name>` | `lc_lb.py` | SpinTare tree | RTX 5090 D |
| RULER 32K, vLLM | `run_vllm_cell.sh ruler <arm> <name>` | `lc_ruler_vllm.py` | vLLM 5cb0fd265 (+ add-back patch for `spintare`) | RTX PRO 5000 |
| LongBench 8K, vLLM | `run_vllm_cell.sh longbench <arm> <name>` | `lc_lb_vllm.py` | same | RTX PRO 5000 |

Arms (llama.cpp): `f16`, `plain` (`q4_0`, no repair), `pm` (static post-RoPE mean), `sa` (SmoothAttention,
folded GGUF), `spintare` (rotary centring with exact add-back and the exemption gate), `nogate` (the same
without the gate). Arms (vLLM): `bf16`, `int4`, `pm`, `sa`, `spintare`. `DRY_RUN=1` prints the commands of
a cell without starting anything; each runner's header lists its environment variables.

Fixed settings, as registered: V `q8_0` (llama.cpp) and flash attention in every arm; the llama.cpp K cache
uses the engine's default Hadamard rotation; LongBench prompts truncated in the middle to 8000 tokens, greedy;
RULER at 32768 tokens, 50 items per task, greedy (llama.cpp prints niah_multikey_3, niah_multiquery, vt,
qa_1; vLLM runs seven tasks and the table prints the same four); GSM8K items 0-199 and MATH-500 items 0-499
of R1-Distill-Qwen-7B with temperature 0.6, top-p 0.95, seed 1000 + item, at most 8192 new tokens, four
parallel slots. The reasoning client keeps a per-record journal, so a run cut by a server crash resumes
with the same seeds.

Inputs:
- mean files: `MU` from `../calib_and_diag/mu_calib.py` (llama.cpp, KVMU) or `python -m spintare_vllm
  calib-mu` (vLLM, KVMR), both on the first 2048 tokens of `data/long_text.txt`; `PM` from `python -m
  spintare calib` (KVPM, llama.cpp postmean tree) or `../calib_and_diag/pm_calib.py` (KVMU, vLLM hook);
- SmoothAttention checkpoints: `../smoothattention/`;
- RULER data: NVIDIA RULER `scripts/data/prepare.py` (commit c3f5e3b, 50 samples, seed 42,
  `--model_template_type base`) per model tokenizer at length 32768;
- LongBench: `scripts/get_longbench.sh "$KVREL_WORK/data/data"`; reasoning data: GSM8K (main, test parquet)
  and MATH-500 under `$KVREL_WORK/reason/` (or `KVT_REASON_ROOT`).

Paired intervals: `../calib_and_diag/paired_ci.py` (one bootstrap index shared by the tasks) and
`../calib_and_diag/reason_ci.py`. The per-item records of the reasoning cells are shipped in
`results/spintare/reasoning_r1/`; the figure scripts read them.

Other clients here: `lc_lb_ho.py` (held-out models, `trust_remote_code`), `lc_probe.py` (needle probe
baseline), `lc_reason_vllm.py`, `lc_lb_vllm.py`, `me_sim.py` (simulated attention KL baseline).
`vllm_site/` is put on `PYTHONPATH` of every vLLM server: it installs the static-mean import hook when
`VLLM_KV_POSTMEAN` is set and registers the SpinTare plugin when `SPINTARE_SERVE=1`.
