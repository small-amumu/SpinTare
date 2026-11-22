# SGLang NVFP4: collapse and the postmean fallback (LongBench 8K)

Source of the SGLang rows of the engine tables (§3 Table engine_collapse, §5 Table main):
Qwen2.5-7B bf16 48.65; NVFP4 18.22, -30.43 [-33.26, -27.52] (collapse); NVFP4 + postmean 47.83, -0.82 [-1.96, +0.29]
(lossless). Qwen3-0.6B: NVFP4 -3.66 [-5.87, -1.45] (undecidable), + postmean -0.71 [-2.65, +1.22].

| File | Role |
|---|---|
| `run_nvsg.sh` | serves each model with SGLang 0.5.22 (`--kv-cache-dtype nvfp4`) and runs `tasks/lc_lb_vllm.py` (4 tasks x 200) |
| `sg_pm/sitecustomize.py` | import hook: subtracts a static post-RoPE key mean in Qwen2 / Qwen3 attention before the cache write (on `PYTHONPATH` only for the postmean cells; inert without `SGLANG_KV_POSTMEAN`) |

Frozen results: `results/spintare/engine_sglang/<model>/` (per-question files and summaries), `run.log` (the original
log of both passes, paths replaced by `$KVREL_WORK`; server logs stay on the run machine). The postmean files are the E2 ones (`long_context_llamacpp/kvmu/pm_*`, same md5).

Re-derive the intervals on CPU with the script that produced them (one bootstrap index shared by the four tasks):

```bash
PYTHONUTF8=1 python experiments/spintare/calib_and_diag/paired_ci.py results/spintare/engine_sglang/sg_qwen25_7b bf16_n200 nvfp4_n200
```

All four intervals reproduce exactly (checked 2026-11-15). `python -m spintare paired-ci` resamples each task
separately (stratified), so its bounds differ by up to 0.3 points (same point estimates and verdicts); the paper uses
the original intervals.

Original run: one RTX 5090 in a container, 2026-10-21 (`run.log`). The first Qwen3-0.6B postmean cell failed to
serve (the hook put the mean on the CPU, which breaks CUDA graph capture); with the hook fixed (shipped here) both
postmean cells were rerun. The Qwen2.5-7B rerun reproduced its first run exactly (47.83; the first run is kept in
`sg_qwen25_7b/run1_hook_unverified/`).
